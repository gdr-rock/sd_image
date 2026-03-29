import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)


DEFAULT_TXT2IMG_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
DEFAULT_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"


def has_model_weights(path: Path) -> bool:
    weight_names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any((path / weight_name).exists() for weight_name in weight_names)


def is_complete_diffusers_snapshot(path: Path) -> bool:
    required_dirs = ["unet", "vae", "text_encoder"]
    if not (path / "model_index.json").exists():
        return False
    for directory in required_dirs:
        if not has_model_weights(path / directory):
            return False
    return True


def resolve_model_source(model_id_or_path: str) -> tuple[str, bool]:
    candidate = Path(model_id_or_path)
    if candidate.exists():
        return str(candidate), is_complete_diffusers_snapshot(candidate) if candidate.is_dir() else True

    try:
        cached_path = snapshot_download(repo_id=model_id_or_path, local_files_only=True)
        cached_dir = Path(cached_path)
        if is_complete_diffusers_snapshot(cached_dir):
            return cached_path, True
        return model_id_or_path, False
    except Exception:
        return model_id_or_path, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal local Stable Diffusion runner for txt2img, img2img, and inpainting."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser, *, include_input: bool, include_mask: bool) -> None:
        subparser.add_argument("--prompt", required=True, help="Main text prompt.")
        subparser.add_argument("--negative-prompt", default="", help="Optional negative prompt.")
        subparser.add_argument("--model-id", default=None, help="HF model id to override the default.")
        subparser.add_argument("--output", default="outputs", help="Output folder.")
        subparser.add_argument("--steps", type=int, default=28, help="Number of denoising steps.")
        subparser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale.")
        subparser.add_argument("--seed", type=int, default=None, help="Random seed. Default is random.")
        subparser.add_argument("--batch-size", type=int, default=1, help="Number of images to generate.")
        subparser.add_argument("--width", type=int, default=None, help="Target width in pixels.")
        subparser.add_argument("--height", type=int, default=None, help="Target height in pixels.")
        subparser.add_argument(
            "--max-side",
            type=int,
            default=768,
            help="If width/height are omitted for img2img/inpaint, scale the longest side to at most this value.",
        )
        subparser.add_argument("--cpu-offload", action="store_true", help="Use CPU offload for tighter VRAM budgets.")
        subparser.add_argument(
            "--control-image",
            default=None,
            help="Optional control image path for ControlNet. If provided, ControlNet is enabled.",
        )
        subparser.add_argument(
            "--controlnet-model",
            default=DEFAULT_CONTROLNET_MODEL,
            help="ControlNet model id. Change this for depth, pose, or segmentation workflows.",
        )
        subparser.add_argument(
            "--control-scale",
            type=float,
            default=1.0,
            help="Strength of the ControlNet conditioning.",
        )
        subparser.add_argument(
            "--control-preprocess",
            choices=["none", "canny"],
            default="none",
            help="Optional preprocessing for the control image. Use canny with the default ControlNet model.",
        )
        if include_input:
            subparser.add_argument("--input", required=True, help="Source image path.")
            subparser.add_argument(
                "--strength",
                type=float,
                default=0.75,
                help="How strongly to transform the source image. Higher means more change.",
            )
        if include_mask:
            subparser.add_argument("--mask", required=True, help="Mask image path. White = editable, black = keep.")

    add_common(subparsers.add_parser("txt2img", help="Generate new images from text."), include_input=False, include_mask=False)
    add_common(subparsers.add_parser("img2img", help="Transform an input image with a text prompt."), include_input=True, include_mask=False)
    add_common(subparsers.add_parser("inpaint", help="Edit masked regions of an input image."), include_input=True, include_mask=True)
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_mask(path: str) -> Image.Image:
    mask = Image.open(path)
    mask = ImageOps.exif_transpose(mask)
    mask = mask.convert("L")
    return mask.point(lambda px: 255 if px >= 128 else 0)


def round_to_multiple_of_8(value: int) -> int:
    return max(64, int(round(value / 8.0) * 8))


def resolve_size(
    image: Optional[Image.Image],
    width: Optional[int],
    height: Optional[int],
    max_side: int,
    mode: str,
) -> Tuple[int, int]:
    if width and height:
        return round_to_multiple_of_8(width), round_to_multiple_of_8(height)

    if image is None:
        base = 512 if mode == "txt2img" else 640
        return round_to_multiple_of_8(width or base), round_to_multiple_of_8(height or base)

    src_w, src_h = image.size
    scale = 1.0
    longest = max(src_w, src_h)
    if longest > max_side:
        scale = max_side / float(longest)

    target_w = width or int(src_w * scale)
    target_h = height or int(src_h * scale)
    return round_to_multiple_of_8(target_w), round_to_multiple_of_8(target_h)


def resize_image(image: Image.Image, size: Tuple[int, int], *, is_mask: bool = False) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.LANCZOS
    return image.resize(size, resample=resample)


def preprocess_control_image(image: Image.Image, method: str) -> Image.Image:
    if method == "none":
        return image
    if method == "canny":
        np_image = np.array(image)
        edges = cv2.Canny(np_image, threshold1=100, threshold2=200)
        edges = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges)
    raise ValueError(f"Unsupported control preprocessing: {method}")


def build_pipeline(args: argparse.Namespace, torch_dtype: torch.dtype):
    use_controlnet = bool(args.control_image)
    model_id = args.model_id
    if model_id is None:
        model_id = DEFAULT_INPAINT_MODEL if args.mode == "inpaint" else DEFAULT_TXT2IMG_MODEL
    model_source, model_is_local = resolve_model_source(model_id)

    common_kwargs = {
        "torch_dtype": torch_dtype,
        "safety_checker": None,
        "local_files_only": model_is_local,
    }

    if use_controlnet:
        controlnet_source, controlnet_is_local = resolve_model_source(args.controlnet_model)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_source,
            torch_dtype=torch_dtype,
            local_files_only=controlnet_is_local,
        )
        if args.mode == "txt2img":
            pipe = StableDiffusionControlNetPipeline.from_pretrained(model_source, controlnet=controlnet, **common_kwargs)
        elif args.mode == "img2img":
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_source, controlnet=controlnet, **common_kwargs)
        else:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(model_source, controlnet=controlnet, **common_kwargs)
    else:
        if args.mode == "txt2img":
            pipe = StableDiffusionPipeline.from_pretrained(model_source, **common_kwargs)
        elif args.mode == "img2img":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_source, **common_kwargs)
        else:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_source, **common_kwargs)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=False)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe, model_id


def configure_device(pipe, cpu_offload: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")
    return device


def save_images(images, output_dir: Path, prefix: str, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for idx, image in enumerate(images, start=1):
        image_path = output_dir / f"{prefix}_{timestamp}_seed{seed}_{idx:02d}.png"
        image.save(image_path)
        print(f"Saved: {image_path}")


def main() -> None:
    args = parse_args()

    source_image = load_image(args.input) if hasattr(args, "input") else None
    mask_image = load_mask(args.mask) if hasattr(args, "mask") else None
    control_image = load_image(args.control_image) if args.control_image else None

    width, height = resolve_size(source_image, args.width, args.height, args.max_side, args.mode)
    target_size = (width, height)

    if source_image is not None:
        source_image = resize_image(source_image, target_size)
    if mask_image is not None:
        mask_image = resize_image(mask_image, target_size, is_mask=True)
    if control_image is not None:
        control_image = resize_image(control_image, target_size)
        control_image = preprocess_control_image(control_image, args.control_preprocess)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe, model_id = build_pipeline(args, dtype)
    device = configure_device(pipe, args.cpu_offload)

    seed = args.seed if args.seed is not None else torch.seed() % (2**31)
    generator = torch.Generator(device="cuda" if device == "cuda" else "cpu").manual_seed(seed)

    print(f"Mode: {args.mode}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Size: {width}x{height}")
    print(f"Seed: {seed}")
    if args.control_image:
        print(f"ControlNet: {args.controlnet_model} (scale={args.control_scale}, preprocess={args.control_preprocess})")

    common_run_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt or None,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "generator": generator,
        "num_images_per_prompt": args.batch_size,
    }

    if args.mode == "txt2img":
        common_run_kwargs.update({"width": width, "height": height})
    else:
        common_run_kwargs.update({"image": source_image, "strength": args.strength})
    if args.mode == "inpaint":
        common_run_kwargs["mask_image"] = mask_image
    if args.control_image:
        common_run_kwargs["controlnet_conditioning_scale"] = args.control_scale
        if args.mode == "txt2img":
            common_run_kwargs["image"] = control_image
        else:
            common_run_kwargs["control_image"] = control_image

    with torch.inference_mode():
        result = pipe(**common_run_kwargs)

    save_images(result.images, Path(args.output), args.mode, seed)


if __name__ == "__main__":
    main()
