"""
ImageFeaturesExtractor.py
-------------------------
Extracts fixed-dimension feature vectors from images using the SigLIP model
and writes them to a CSV file. Designed to run on large datasets in batches
(e.g., on a SLURM cluster), but also works locally on any folder of images.

Usage:
    python ImageFeaturesExtractor.py \
        --image-dir ./images \
        --output-path ./features.csv \
        --start-folder 0161 \
        --end-folder 0260 \
        --batch-size 8 \
        --cache-dir ./model_cache
"""

import argparse
import csv
import logging
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModel, AutoProcessor

# ── Constants ─────────────────────────────────────────────────
MODEL_NAME = "google/siglip-so400m-patch14-384"
IMAGE_SIZE = 384
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract SigLIP image features to CSV.")
    parser.add_argument("--image-dir", required=True,
                        help="Root directory of images (ImageFolder structure).")
    parser.add_argument("--output-path", required=True,
                        help="Destination CSV file path.")
    parser.add_argument("--start-folder", default=None,
                        help="First sub-folder name to include (e.g. '0161'). "
                             "Leave blank to process all folders.")
    parser.add_argument("--end-folder", default=None,
                        help="Last sub-folder name to include (e.g. '0260').")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="DataLoader batch size (default: 8).")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to cache downloaded model weights.")
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    """Return the preprocessing pipeline expected by SigLIP."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def load_model(cache_dir: str | None, device: torch.device):
    """Download (or load from cache) the SigLIP model and processor."""
    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        do_rescale=False,
    )
    model.to(device)
    model.eval()
    return model, processor


def build_dataloader(image_dir: str, start_folder: str | None,
                     end_folder: str | None, batch_size: int) -> tuple[DataLoader, list[int], ImageFolder]:
    """
    Create a DataLoader filtered to a sub-range of sub-folders.

    Returns the DataLoader, the filtered index list, and the full dataset
    (needed to resolve original file paths later).
    """
    dataset = ImageFolder(root=image_dir, transform=build_transform())

    if start_folder is not None and end_folder is not None:
        filtered_indices = [
            i for i, (path, _) in enumerate(dataset.imgs)
            if start_folder <= path.split(os.sep)[-2] <= end_folder
        ]
        logger.info("Filtered to %d images (folders %s–%s).",
                    len(filtered_indices), start_folder, end_folder)
    else:
        filtered_indices = list(range(len(dataset)))
        logger.info("Processing all %d images.", len(filtered_indices))

    subset = Subset(dataset, filtered_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader, filtered_indices, dataset


def extract_features(model, processor, dataloader: DataLoader,
                     filtered_indices: list[int], dataset: ImageFolder,
                     output_path: str, device: torch.device) -> None:
    """Run inference and write (image_path, feature_vector) rows to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "features"])

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(dataloader):
                logger.info("Processing batch %d / %d", batch_idx + 1, len(dataloader))

                # Normalise to [0, 1] before handing to the processor
                inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
                inputs = processor(images=inputs, return_tensors="pt").to(device)

                features = model.get_image_features(**inputs)

                start = batch_idx * dataloader.batch_size
                end = start + len(inputs["pixel_values"])
                batch_indices = filtered_indices[start:end]

                if len(inputs["pixel_values"]) != len(batch_indices):
                    logger.error("Batch size mismatch at batch %d — stopping.", batch_idx)
                    break

                for offset, global_idx in enumerate(batch_indices):
                    raw_path = dataset.imgs[global_idx][0]
                    # Store a relative path so the CSV is portable
                    rel_path = os.path.join("images", raw_path.split("/images/")[-1])
                    writer.writerow([rel_path, features[offset].tolist()])

                del inputs, features
                if device.type != "cpu":
                    torch.cuda.empty_cache()

    logger.info("Features saved to %s", output_path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model, processor = load_model(args.cache_dir, device)
    dataloader, filtered_indices, dataset = build_dataloader(
        args.image_dir, args.start_folder, args.end_folder, args.batch_size
    )
    extract_features(model, processor, dataloader, filtered_indices,
                     dataset, args.output_path, device)


if __name__ == "__main__":
    main()
