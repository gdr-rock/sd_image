import argparse
from pathlib import Path

from PIL import Image, ImageOps
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LaMa-based clean object removal. White mask pixels are removed and filled."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--mask", required=True, help="Binary mask path. White = remove, black = keep.")
    parser.add_argument("--output", required=True, help="Output image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = tqdm(total=3, desc="LaMa cleanup", unit="stage")

    image = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    mask = ImageOps.exif_transpose(Image.open(args.mask)).convert("L")
    mask = mask.point(lambda px: 255 if px >= 128 else 0)
    progress.update(1)

    lama = SimpleLama()
    result = lama(image, mask)
    progress.update(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    progress.update(1)
    progress.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
