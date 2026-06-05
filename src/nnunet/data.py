"""BraTS2023 dataset preparation for single-modality nnUNet segmentation.

Converts the BraTS2023 folder structure into the datalist JSON format
expected by ``monai.apps.nnunet.nnUNetV2Runner.convert_dataset()``.

BraTS2023 layout (per subject)::

    BraTS-GLI-XXXXX-XXX/
        t1c.nii.gz
        t1n.nii.gz
        t2f.nii.gz
        t2w.nii.gz
        seg.nii.gz          (training subjects only)

After calling :func:`prepare_datalist`, the output JSON has the shape::

    {"training": [{"image": "<relative_path>", "label": "<relative_path>"}],
     "test":     [{"image": "<relative_path>"}]}
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

__all__ = ["prepare_datalist", "prepare_inference_folder"]

# Modality filename stems shipped with BraTS2023.
MODALITY_NAMES: dict[str, str] = {
    "t1c": "t1c.nii.gz",
    "t1n": "t1n.nii.gz",
    "t2f": "t2f.nii.gz",
    "t2w": "t2w.nii.gz",
}


def prepare_datalist(
    data_root: str,
    modality: str = "t2f",
    train_ratio: float = 1.0,
    output_json: str | None = None,
    seed: int = 42,
) -> dict:
    """Scan *data_root* for BraTS2023 subjects and build a datalist dict.

    Parameters
    ----------
    data_root : str
        Root directory containing subject folders (e.g.
        ``/data/BraTS2023/GLI_train``).
    modality : str
        One of ``"t1c"``, ``"t1n"``, ``"t2f"``, ``"t2w"``.  The single
        modality to use for segmentation.
    train_ratio : float
        Fraction of subjects with a ``seg.nii.gz`` to assign to training.
        Default ``1.0`` — all labeled subjects go to ``"training"`` and
        nnUNet handles the cross-validation split internally.  Set to a
        value < 1.0 to hold out labeled subjects as unlabeled test cases
        (their labels will **not** be included in the datalist).
    output_json : str | None
        If given, write the datalist to this path as JSON.
    seed : int
        Random seed for the train/test split.

    Returns
    -------
    dict
        Datalist compatible with ``nnUNetV2Runner.convert_dataset()``.
    """
    if modality not in MODALITY_NAMES:
        raise ValueError(
            f"Unknown modality {modality!r}. Choose from {list(MODALITY_NAMES)}"
        )
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0.0, 1.0], got {train_ratio}")

    modality_file = MODALITY_NAMES[modality]
    data_root = Path(data_root)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Collect subjects: training subjects have seg.nii.gz, test subjects don't.
    training_entries: list[dict[str, str]] = []
    test_entries: list[dict[str, str]] = []

    # Separate folders: those with seg.nii.gz are training, those without are test.
    train_subjects: list[str] = []
    test_subjects: list[str] = []

    # Support both naming conventions:
    #   t2f.nii.gz              (post-2023 unpacked)
    #   BraTS-GLI-00000-000-t2f.nii.gz  (original tarball)
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        modality_path = _find_modality_file(entry, modality_file)
        if modality_path is None:
            continue
        has_label = _find_seg_file(entry) is not None
        if has_label:
            train_subjects.append(entry.name)
        else:
            test_subjects.append(entry.name)

    if not train_subjects and not test_subjects:
        raise ValueError(
            f"No BraTS2023 subjects with modality {modality!r} found in {data_root}"
        )

    # Shuffle and split training subjects into train/val ("test" in nnUNet datalist).
    rng = random.Random(seed)
    rng.shuffle(train_subjects)
    split_idx = int(len(train_subjects) * train_ratio)
    train_fold = train_subjects[:split_idx]
    val_fold = train_subjects[split_idx:]

    for subj in train_fold:
        img_rel = _find_modality_file(data_root / subj, modality_file)
        lbl_rel = _find_seg_file(data_root / subj)
        assert img_rel is not None and lbl_rel is not None
        training_entries.append({"image": img_rel, "label": lbl_rel})

    for subj in val_fold:
        img_rel = _find_modality_file(data_root / subj, modality_file)
        assert img_rel is not None
        test_entries.append({"image": img_rel})

    for subj in test_subjects:
        img_rel = _find_modality_file(data_root / subj, modality_file)
        assert img_rel is not None
        test_entries.append({"image": img_rel})

    datalist: dict = {"training": training_entries, "test": test_entries}

    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(datalist, f, indent=4)

    return datalist


def _find_modality_file(subject_dir: Path, modality_file: str) -> str | None:
    """Return relative path to the modality file inside *subject_dir*, or None."""
    # Try direct name first: subject_dir / t2f.nii.gz
    direct = subject_dir / modality_file
    if direct.exists():
        return os.path.join(subject_dir.name, modality_file)
    # Try prefixed name: subject_dir / BraTS-GLI-XXXXX-XXX-t2f.nii.gz
    for f in subject_dir.iterdir():
        if f.name.endswith("-" + modality_file):
            return os.path.join(subject_dir.name, f.name)
    return None


def _find_seg_file(subject_dir: Path) -> str | None:
    """Return relative path to seg.nii.gz inside *subject_dir*, or None."""
    direct = subject_dir / "seg.nii.gz"
    if direct.exists():
        return os.path.join(subject_dir.name, "seg.nii.gz")
    for f in subject_dir.iterdir():
        if f.name.endswith("-seg.nii.gz"):
            return os.path.join(subject_dir.name, f.name)
    return None


def prepare_inference_folder(
    data_root: str,
    output_folder: str,
    modality: str = "t2f",
) -> str:
    """Symlink BraTS2023 subjects into a flat folder with nnUNet naming.

    nnUNet ``predict_from_files`` expects inputs named
    ``<case_id>_0000.nii.gz`` (channel 0).  This function creates
    symlinks from each subject's chosen modality file into *output_folder*
    using that convention.

    Parameters
    ----------
    data_root : str
        Root directory containing subject sub-folders.
    output_folder : str
        Flat folder to create with symlinks.
    modality : str
        One of ``"t1c"``, ``"t1n"``, ``"t2f"``, ``"t2w"``.

    Returns
    -------
    str
        The *output_folder* path (for convenience).
    """
    if modality not in MODALITY_NAMES:
        raise ValueError(f"Unknown modality {modality!r}. Choose from {list(MODALITY_NAMES)}")

    modality_file = MODALITY_NAMES[modality]
    data_root = Path(data_root).resolve()
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        rel = _find_modality_file(entry, modality_file)
        if rel is None:
            continue
        src = (data_root / rel).resolve()
        dst = out / f"{entry.name}_0000.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    return str(out)
