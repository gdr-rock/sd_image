import argparse
import sys
from pathlib import Path

import cv2
import requests

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms._functional_tensor as functional_tensor

    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from tqdm import tqdm


REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-ESRGAN upscaling with optional GFPGAN face restoration."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--outscale", type=float, default=4.0, help="Final output scale factor.")
    parser.add_argument(
        "--tile",
        type=int,
        default=512,
        help="Tile size for 8GB GPUs. Lower it if you hit CUDA OOM. Use 0 to disable tiling.",
    )
    parser.add_argument("--face-enhance", action="store_true", help="Apply GFPGAN face enhancement.")
    parser.add_argument("--weights-dir", default="weights", help="Folder for downloaded model weights.")
    return parser.parse_args()


def download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {target_path.name}")
    with target_path.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file_handle.write(chunk)
            progress.update(len(chunk))
    progress.close()


def build_upsampler(model_path: Path, tile: int) -> RealESRGANer:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=cv2.cuda.getCudaEnabledDeviceCount() > 0,
    )


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir)
    realesrgan_weight = weights_dir / "RealESRGAN_x4plus.pth"
    gfpgan_weight = weights_dir / "GFPGANv1.3.pth"

    stage_bar = tqdm(total=4 if args.face_enhance else 3, desc="Upscale pipeline", unit="stage")
    download_file(REALESRGAN_URL, realesrgan_weight)
    if args.face_enhance:
        download_file(GFPGAN_URL, gfpgan_weight)
    stage_bar.update(1)

    upsampler = build_upsampler(realesrgan_weight, args.tile)
    stage_bar.update(1)

    input_image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if input_image is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    if args.face_enhance:
        restorer = GFPGANer(
            model_path=str(gfpgan_weight),
            upscale=int(max(1, round(args.outscale))),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        _, _, output = restorer.enhance(
            input_image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
    else:
        output, _ = upsampler.enhance(input_image, outscale=args.outscale)
    stage_bar.update(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    stage_bar.update(1)
    stage_bar.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
