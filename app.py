import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import requests
import torch
from PIL import Image, ImageChops

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms._functional_tensor as functional_tensor

    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from simple_lama_inpainting import SimpleLama

from sd_generate import (
    DEFAULT_CONTROLNET_MODEL,
    DEFAULT_INPAINT_MODEL,
    DEFAULT_TXT2IMG_MODEL,
    build_pipeline,
    configure_device,
    load_image,
    load_mask,
    preprocess_control_image,
    resize_image,
    resolve_size,
)


REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
OUTPUT_DIR = Path("outputs")
WEIGHTS_DIR = Path("weights")


def save_output(image: Image.Image, prefix: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    image.save(output_path)
    return str(output_path)


def ensure_pil_image(value) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return Image.open(value)
    raise ValueError("Unsupported image value from editor.")


def extract_background_and_mask(editor_value) -> tuple[Image.Image, Image.Image]:
    if not editor_value or not editor_value.get("background"):
        raise gr.Error("Upload an image into the mask editor first.")

    background = ensure_pil_image(editor_value["background"]).convert("RGB")
    layers = editor_value.get("layers") or []
    mask = Image.new("L", background.size, 0)

    for layer in layers:
        layer_image = ensure_pil_image(layer).convert("RGBA")
        mask = ImageChops.lighter(mask, layer_image.getchannel("A"))

    if not layers and editor_value.get("composite"):
        composite = ensure_pil_image(editor_value["composite"]).convert("RGB")
        diff = ImageChops.difference(background, composite).convert("L")
        mask = diff

    mask = mask.point(lambda px: 255 if px >= 16 else 0)
    return background, mask


@lru_cache(maxsize=12)
def get_sd_pipeline(
    mode: str,
    model_id: str,
    controlnet_model: Optional[str],
    use_controlnet: bool,
    cpu_offload: bool,
):
    args = argparse.Namespace(
        mode=mode,
        model_id=model_id,
        control_image="enabled" if use_controlnet else None,
        controlnet_model=controlnet_model or DEFAULT_CONTROLNET_MODEL,
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe, resolved_model_id = build_pipeline(args, dtype)
    device = configure_device(pipe, cpu_offload)
    return pipe, resolved_model_id, device


@lru_cache(maxsize=1)
def get_lama():
    return SimpleLama()


def download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with target_path.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_handle.write(chunk)


@lru_cache(maxsize=4)
def get_upsampler(tile: int):
    realesrgan_weight = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
    download_file(REALESRGAN_URL, realesrgan_weight)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=str(realesrgan_weight),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )


@lru_cache(maxsize=4)
def get_face_restorer(tile: int, outscale: float):
    gfpgan_weight = WEIGHTS_DIR / "GFPGANv1.3.pth"
    download_file(GFPGAN_URL, gfpgan_weight)
    upsampler = get_upsampler(tile)
    return GFPGANer(
        model_path=str(gfpgan_weight),
        upscale=max(1, int(round(outscale))),
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )


def build_info_block(title: str, lines: list[str]) -> str:
    return "\n".join([f"### {title}", *[f"- {line}" for line in lines]])


def run_sd(
    mode: str,
    prompt: str,
    negative_prompt: str,
    input_path: Optional[str],
    inpaint_editor,
    control_path: Optional[str],
    model_id: str,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
    width: int,
    height: int,
    max_side: int,
    controlnet_model: str,
    control_scale: float,
    control_preprocess: str,
    cpu_offload: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt.strip():
        raise gr.Error("Prompt is required.")
    if mode == "img2img" and not input_path:
        raise gr.Error("Input image is required for img2img.")

    source_image = None
    mask_image = None
    if mode == "inpaint":
        source_image, mask_image = extract_background_and_mask(inpaint_editor)
    elif input_path:
        source_image = load_image(input_path)
    control_image = load_image(control_path) if control_path else None

    final_model = model_id.strip() or (DEFAULT_INPAINT_MODEL if mode == "inpaint" else DEFAULT_TXT2IMG_MODEL)
    use_controlnet = control_image is not None

    target_width = width or None
    target_height = height or None
    resolved_width, resolved_height = resolve_size(source_image, target_width, target_height, max_side, mode)
    target_size = (resolved_width, resolved_height)

    if source_image is not None:
        source_image = resize_image(source_image, target_size)
    if mask_image is not None:
        mask_image = resize_image(mask_image, target_size, is_mask=True)
    if control_image is not None:
        control_image = resize_image(control_image, target_size)
        control_image = preprocess_control_image(control_image, control_preprocess)

    progress(0.1, desc="Loading model pipeline")
    pipe, resolved_model, device = get_sd_pipeline(
        mode,
        final_model,
        controlnet_model,
        use_controlnet,
        cpu_offload,
    )

    actual_seed = int(seed) if seed and seed >= 0 else int(torch.seed() % (2**31))
    generator = torch.Generator(device="cuda" if device == "cuda" else "cpu").manual_seed(actual_seed)

    run_kwargs = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip() or None,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "generator": generator,
        "num_images_per_prompt": 1,
    }

    if mode == "txt2img":
        run_kwargs.update({"width": resolved_width, "height": resolved_height})
    else:
        run_kwargs.update({"image": source_image, "strength": float(strength)})
    if mode == "inpaint":
        run_kwargs["mask_image"] = mask_image
    if use_controlnet:
        run_kwargs["controlnet_conditioning_scale"] = float(control_scale)
        if mode == "txt2img":
            run_kwargs["image"] = control_image
        else:
            run_kwargs["control_image"] = control_image

    progress(0.2, desc="Generating image")
    with torch.inference_mode():
        result = pipe(**run_kwargs)

    output = result.images[0]
    output_path = save_output(output, f"ui_{mode}")
    info = build_info_block(
        "Run summary",
        [
            f"Mode: `{mode}`",
            f"Model: `{resolved_model}`",
            f"Device: `{device}`",
            f"Size: `{resolved_width}x{resolved_height}`",
            f"Seed: `{actual_seed}`",
            f"ControlNet: `{'on' if use_controlnet else 'off'}`",
            f"Saved to: `{output_path}`",
        ],
    )
    return output, info


def run_lama(editor_value, progress=gr.Progress(track_tqdm=True)):
    progress(0.1, desc="Loading image and mask")
    image, mask = extract_background_and_mask(editor_value)
    lama = get_lama()

    progress(0.4, desc="Running LaMa cleanup")
    result = lama(image, mask)
    output_path = save_output(result, "ui_lama")
    info = build_info_block(
        "Run summary",
        [
            "Model: `LaMa clean erase`",
            "Purpose: remove masked objects while keeping nearby content stable",
            f"Saved to: `{output_path}`",
        ],
    )
    return result, info


def run_upscale(input_path: str, outscale: float, tile: int, face_enhance: bool, progress=gr.Progress(track_tqdm=True)):
    if not input_path:
        raise gr.Error("Input image is required.")

    progress(0.1, desc="Preparing upscaler")
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise gr.Error("Could not read the uploaded image.")

    tile_value = max(0, int(tile))
    if face_enhance:
        progress(0.3, desc="Running GFPGAN + Real-ESRGAN")
        restorer = get_face_restorer(tile_value, float(outscale))
        _, _, output = restorer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
    else:
        progress(0.3, desc="Running Real-ESRGAN")
        upsampler = get_upsampler(tile_value)
        output, _ = upsampler.enhance(image, outscale=float(outscale))

    if output.ndim == 3 and output.shape[2] == 4:
        rgb_output = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
    else:
        rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_output = Image.fromarray(rgb_output)
    output_path = save_output(pil_output, "ui_upscale")
    info = build_info_block(
        "Run summary",
        [
            f"Upscaler: `{'GFPGAN + Real-ESRGAN' if face_enhance else 'Real-ESRGAN'}`",
            f"Outscale: `{outscale}`",
            f"Tile: `{tile_value}`",
            f"Saved to: `{output_path}`",
        ],
    )
    return pil_output, info


def toggle_sd_fields(mode: str):
    is_txt2img = mode == "txt2img"
    is_inpaint = mode == "inpaint"
    return (
        gr.update(visible=mode == "img2img"),
        gr.update(visible=is_inpaint),
        gr.update(visible=not is_txt2img),
    )


def build_app():
    with gr.Blocks(title="Local SD Image Toolkit") as demo:
        gr.Markdown(
            """
            # Local SD Image Toolkit
            Small local UI for SD1.5 editing, LaMa cleanup, and Real-ESRGAN upscaling.

            Models load on demand and stay cached in memory where possible. On 8GB GPUs, start with `512` or `640` sized jobs, keep batch size at `1`, and enable CPU offload when ControlNet or larger sizes get tight.
            """
        )

        with gr.Tab("Generate / Edit"):
            with gr.Row():
                with gr.Column(scale=2):
                    mode = gr.Radio(
                        choices=["txt2img", "img2img", "inpaint"],
                        value="txt2img",
                        label="Mode",
                        info="txt2img creates from scratch. img2img restyles an image. inpaint edits only the white-masked region.",
                    )
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe what you want to generate or edit.")
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        lines=2,
                        placeholder="Optional: blurry, low quality, extra fingers, distorted face",
                        info="Things you want the model to avoid.",
                    )
                    input_image = gr.Image(
                        type="filepath",
                        label="Input image",
                        visible=False,
                    )
                    inpaint_editor = gr.ImageEditor(
                        type="pil",
                        label="Inpaint mask editor",
                        visible=False,
                        height=420,
                        image_mode="RGBA",
                        brush=gr.Brush(
                            colors=["#FFFFFF"],
                            default_color="#FFFFFF",
                            color_mode="fixed",
                            default_size=28,
                        ),
                        eraser=gr.Eraser(default_size=24),
                        layers=True,
                        transforms=["crop", "resize"],
                        info="Upload the source image here, then paint white over the region you want to edit. White changes, untouched areas stay protected.",
                    )
                    control_image = gr.Image(
                        type="filepath",
                        label="Control image (optional)",
                        visible=True,
                    )
                    with gr.Accordion("Advanced settings", open=False):
                        model_id = gr.Textbox(
                            label="Model ID",
                            value="",
                            placeholder=f"Default: {DEFAULT_TXT2IMG_MODEL} or {DEFAULT_INPAINT_MODEL}",
                            info="Leave blank to use the recommended default SD1.5 model for the selected mode.",
                        )
                        steps = gr.Slider(10, 60, value=28, step=1, label="Steps", info="More steps can improve detail a bit, but also make runs slower.")
                        guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance", info="Higher values follow the prompt more strongly. Too high can look unnatural.")
                        strength = gr.Slider(
                            0.1,
                            1.0,
                            value=0.75,
                            step=0.05,
                            label="Strength",
                            info="Only for img2img and inpaint. Lower preserves more of the source image, higher changes it more.",
                            visible=False,
                        )
                        seed = gr.Number(value=-1, precision=0, label="Seed", info="Use -1 for a random seed. Reuse a seed to repeat a composition.")
                        with gr.Row():
                            width = gr.Slider(0, 1536, value=0, step=8, label="Width", info="0 means auto-size.")
                            height = gr.Slider(0, 1536, value=0, step=8, label="Height", info="0 means auto-size.")
                        max_side = gr.Slider(
                            256,
                            1536,
                            value=768,
                            step=8,
                            label="Max side",
                            info="When width and height are 0, uploaded images are scaled down so the longest side stays near this value.",
                        )
                        cpu_offload = gr.Checkbox(
                            value=False,
                            label="Enable CPU offload",
                            info="Helps on 8GB VRAM when using larger sizes or ControlNet, but it is slower.",
                        )
                        with gr.Accordion("ControlNet", open=False):
                            controlnet_model = gr.Textbox(
                                label="ControlNet model",
                                value=DEFAULT_CONTROLNET_MODEL,
                                info="Default is canny edges. Swap this to depth, openpose, or segmentation if you already have the matching control image.",
                            )
                            control_scale = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Control scale", info="Higher values stick more closely to the control image.")
                            control_preprocess = gr.Dropdown(
                                choices=["none", "canny"],
                                value="none",
                                label="Control preprocess",
                                info="Use canny when the control image should be turned into edge guidance automatically.",
                            )
                    run_sd_button = gr.Button("Run generation", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        **Parameter hints**

                        - `Steps`: `24-32` is a good default range.
                        - `Guidance`: `6.5-8.5` is usually the sweet spot.
                        - `Strength`: `0.45-0.7` for controlled edits, `0.8+` for stronger changes.
                        - `Mask editor`: paint white on the area you want to edit.
                        - `ControlNet`: use only when you need stricter structure.
                        """
                    )
                    sd_output = gr.Image(label="Output")
                    sd_info = gr.Markdown()

        with gr.Tab("LaMa Clean Erase"):
            with gr.Row():
                with gr.Column(scale=2):
                    lama_editor = gr.ImageEditor(
                        type="pil",
                        label="LaMa mask editor",
                        height=420,
                        image_mode="RGBA",
                        brush=gr.Brush(
                            colors=["#FFFFFF"],
                            default_color="#FFFFFF",
                            color_mode="fixed",
                            default_size=28,
                        ),
                        eraser=gr.Eraser(default_size=24),
                        layers=True,
                        transforms=["crop", "resize"],
                        info="Upload the image here, then paint white over objects you want LaMa to remove.",
                    )
                    run_lama_button = gr.Button("Run LaMa cleanup", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        **When to use LaMa**

                        Use this when the job is simple object removal and you do not want diffusion to redesign nearby regions.
                        """
                    )
                    lama_output = gr.Image(label="Output")
                    lama_info = gr.Markdown()

        with gr.Tab("Upscale"):
            with gr.Row():
                with gr.Column(scale=2):
                    upscale_input = gr.Image(type="filepath", label="Input image")
                    outscale = gr.Slider(1.0, 4.0, value=4.0, step=0.5, label="Outscale", info="How much to enlarge the image.")
                    tile = gr.Slider(
                        0,
                        1024,
                        value=512,
                        step=32,
                        label="Tile size",
                        info="Lower tile size uses less VRAM. Try 256 if you hit memory issues.",
                    )
                    face_enhance = gr.Checkbox(
                        value=False,
                        label="Enable face enhancement",
                        info="Useful for portraits. Usually unnecessary for landscapes or product images.",
                    )
                    run_upscale_button = gr.Button("Run upscale", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        **Upscale tips**

                        - `Outscale 4` is the main high-quality mode.
                        - `Tile 512` is a good 8GB default.
                        - `Tile 256` is safer when memory gets tight.
                        """
                    )
                    upscale_output = gr.Image(label="Output")
                    upscale_info = gr.Markdown()

        mode.change(
            toggle_sd_fields,
            inputs=mode,
            outputs=[input_image, inpaint_editor, strength],
        )
        run_sd_button.click(
            run_sd,
            inputs=[
                mode,
                prompt,
                negative_prompt,
                input_image,
                inpaint_editor,
                control_image,
                model_id,
                steps,
                guidance,
                strength,
                seed,
                width,
                height,
                max_side,
                controlnet_model,
                control_scale,
                control_preprocess,
                cpu_offload,
            ],
            outputs=[sd_output, sd_info],
        )
        run_lama_button.click(run_lama, inputs=[lama_editor], outputs=[lama_output, lama_info])
        run_upscale_button.click(
            run_upscale,
            inputs=[upscale_input, outscale, tile, face_enhance],
            outputs=[upscale_output, upscale_info],
        )
    demo.queue()
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
