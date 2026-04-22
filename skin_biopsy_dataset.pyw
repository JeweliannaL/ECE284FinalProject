"""
Skin Biopsy Disease Classification — Dataset & DataLoader
==========================================================

Usage:
    from skin_biopsy_dataset import build_dataloaders

    train_loader, test_loader, classes = build_dataloaders(
        csv_path  = "SkinBiopsyAnnotation.csv",
        root_dir  = "/path/to/tiff/files",
        image_size = 224,
        batch_size = 8,
        num_workers = 4,
        # label_col (Jf not uploading an option, a menu will pop out to ask)
    )

Dependencies:
    pip install tifffile torch torchvision pandas numpy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Literal

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import tifffile


# ─────────────────────────────────────────────────────────────────
# Selection of Classification Criteria
# ─────────────────────────────────────────────────────────────────

_LABEL_OPTIONS = {
    "1": ("category", "category  — 3  types: BCC / SCC / negative"),
    "2": ("type",     "type      — 8  types: BCC / SCC / SK / nevus / AK / negative ..."),
    "3": ("subtype",  "subtype   — 13 types: BCC-nodular / SCC-welldiff / BCC-infiltrative ..."),
}

def _ask_label_col() -> str:
    print("\n" + "=" * 55)
    print("  Select the Classification Option (label_col):")
    print("=" * 55)
    for key, (col, desc) in _LABEL_OPTIONS.items():
        print(f"  [{key}]  {desc}")
    print("=" * 55)

    while True:
        choice = input("  Enter Code (1 / 2 / 3): ").strip()
        if choice in _LABEL_OPTIONS:
            col, desc = _LABEL_OPTIONS[choice]
            print(f"  ✓ Selected: {desc}\n")
            return col
        print("  Invalid input, please enter 1, 2, or 3")


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class SkinBiopsyDataset(Dataset):
    """
    Parameters
    ----------
    csv_path  : SkinBiopsyAnnotation.csv Path
    root_dir  : Save TIFF root directory (Name = slide Column value + .tif/.tiff)
    split     : "train" | "test" | "all"
    label_col : "category" | "type" | "subtype"
    transform : torchvision transform，input as float32 CHW tensor [0,1]
    """

    def __init__(
        self,
        csv_path:  str | Path,
        root_dir:  str | Path,
        split:     Literal["train", "test", "all"] = "train",
        label_col: Literal["category", "type", "subtype"] = "category",
        transform: Optional[Callable] = None,
    ):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.label_col = label_col

        df = pd.read_csv(csv_path)

        # Fix the inconsistency of SK in types
        if label_col == "type":
            df["type"] = df["type"].str.strip().replace({"Sk": "SK", "sk": "SK"})

        if split != "all":
            df = df[df["dataset"] == split].reset_index(drop=True)
        self.df = df

        # labels
        classes = sorted(df[label_col].unique().tolist())
        self.classes       = classes
        self.class_to_idx  = {c: i for i, c in enumerate(classes)}
        self.idx_to_class  = {i: c for c, i in self.class_to_idx.items()}

        # Print the distribution of data
        self._print_distribution(split)

    # ------------------------------------------------------------------
    def _print_distribution(self, split: str):
        counts = self.df[self.label_col].value_counts().sort_index()
        print(f"\n[SkinBiopsyDataset | split='{split}' | label_col='{self.label_col}']")
        print(f"  Total sample: {len(self.df)}")
        print(f"  Caregory Distribution:")
        for cls, n in counts.items():
            idx = self.class_to_idx[cls]
            bar = "█" * (n // 3)
            print(f"    [{idx}] {cls:<25} {n:>4} {bar}")

    # ------------------------------------------------------------------
    def _find_tiff(self, slide: str) -> Path:
        for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
            p = self.root_dir / f"{slide}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Cannot find slide='{slide}'  TIFF file\n"
            f"Search root dir: {self.root_dir}\n"
            f"Expected File: {slide}.tif or {slide}.tiff"
        )

    # ------------------------------------------------------------------
    def _load_tiff(self, path: Path) -> torch.Tensor:
        """Return float32 CHW tensor，value range [0, 1]"""
        img = tifffile.imread(str(path))   # (H,W) or (H,W,C)

        if img.ndim == 2:                 
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] > 3:
            img = img[..., :3]         

        # Normalize
        dtype_max = {np.uint8: 255.0, np.uint16: 65535.0}.get(
            img.dtype.type, float(img.max()) or 1.0
        )
        img = img.astype(np.float32) / dtype_max

        return torch.from_numpy(img).permute(2, 0, 1)  # → CHW

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        img   = self._load_tiff(self._find_tiff(row["slide"]))
        label = self.class_to_idx[row[self.label_col]]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ─────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=90),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

def get_val_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────
# DataLoader 
# ─────────────────────────────────────────────────────────────────

def build_dataloaders(
    csv_path:    str | Path,
    root_dir:    str | Path,
    label_col:   Optional[Literal["category", "type", "subtype"]] = None,
    image_size:  int = 224,
    batch_size:  int = 8,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Return (train_loader, test_loader, class_names)

    When label_col is null, ask user to enter
    """
    if label_col is None:
        label_col = _ask_label_col()

    train_ds = SkinBiopsyDataset(
        csv_path, root_dir, split="train",
        label_col=label_col,
        transform=get_train_transform(image_size),
    )
    test_ds = SkinBiopsyDataset(
        csv_path, root_dir, split="test",
        label_col=label_col,
        transform=get_val_transform(image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n✓ DataLoader Constructed successfully")
    print(f"  label_col  : {label_col}")
    print(f"  classes    : {train_ds.classes}")
    print(f"  image_size : {image_size}×{image_size}")
    print(f"  batch_size : {batch_size}")
    print(f"  train      : {len(train_ds)} samples → {len(train_loader)} batches")
    print(f"  test       : {len(test_ds)}  samples → {len(test_loader)} batches\n")

    return train_loader, test_loader, train_ds.classes


# ─────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, test_loader, classes = build_dataloaders(
        csv_path   = "SkinBiopsyAnnotation.csv",
        root_dir   = "/path/to/tiff/files",  
        image_size = 224,
        batch_size = 8,
        num_workers = 0, 
    )

    # Use a batch to verify
    imgs, labels = next(iter(train_loader))
    print(f"Batch image shape : {imgs.shape}")   # (B, 3, 224, 224)
    print(f"Batch labels      : {labels}")
    print(f"Label → class     : { {i: classes[i] for i in labels.unique().tolist()} }")
