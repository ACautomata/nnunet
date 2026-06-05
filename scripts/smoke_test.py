#!/usr/bin/env python3
"""Smoke test for BraTS2023 nnUNet segmentation pipeline.

Verifies the full pipeline (prepare -> plan -> train -> validate)
on a small subset of BraTS2023-GLI training data.

Creates the nnUNet raw dataset manually (bypassing MONAI's convert_dataset
which has a naming quirk), then uses MONAI's nnUNetV2Runner for planning,
training, and validation.

Environment variables:
    SMOKE_RUNTIME      - Working directory for all outputs (required)
    SMOKE_DATA_ROOT    - BraTS2023-GLI training data root (required)
    SMOKE_NUM_SUBJECTS - Number of subjects to use (default: 10)
    SMOKE_EPOCHS       - Training epochs (default: 20)
    SMOKE_CONFIG       - nnUNet config: "2d" or "3d_fullres" (default: "2d")
    SMOKE_GPU_ID       - GPU device (default: 0)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    cfg = get_config()
    print_banner(cfg)

    t0 = time.time()

    setup_nnunet_env(cfg)
    create_data_subset(cfg)
    create_nnunet_raw_dataset(cfg)
    install_custom_trainer(cfg)
    plan_and_preprocess(cfg)
    train(cfg)
    validate(cfg)
    report_results(cfg)

    elapsed = time.time() - t0
    print(f"\n=== Smoke test PASSED in {elapsed:.0f}s ({elapsed / 60:.1f} min) ===")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config() -> dict:
    runtime = os.environ.get("SMOKE_RUNTIME")
    data_root = os.environ.get("SMOKE_DATA_ROOT")
    if not runtime or not data_root:
        print("ERROR: Set SMOKE_RUNTIME and SMOKE_DATA_ROOT environment variables")
        sys.exit(1)

    return {
        "runtime": Path(runtime),
        "data_root": Path(data_root),
        "num_subjects": int(os.environ.get("SMOKE_NUM_SUBJECTS", "10")),
        "epochs": int(os.environ.get("SMOKE_EPOCHS", "20")),
        "config": os.environ.get("SMOKE_CONFIG", "2d"),
        "fold": 0,
        "gpu_id": int(os.environ.get("SMOKE_GPU_ID", "0")),
        "modality": "t2f",
        "dataset_id": 1001,
    }


def print_banner(cfg: dict) -> None:
    print("=" * 60)
    print("BraTS2023 nnUNet Smoke Test")
    print("=" * 60)
    for k, v in [
        ("Data root", cfg["data_root"]),
        ("Runtime", cfg["runtime"]),
        ("Subjects", cfg["num_subjects"]),
        ("Modality", cfg["modality"]),
        ("Config", cfg["config"]),
        ("Epochs", cfg["epochs"]),
        ("Fold", cfg["fold"]),
        ("GPU", cfg["gpu_id"]),
        ("Dataset ID", cfg["dataset_id"]),
    ]:
        print(f"  {k + ':':<14}{v}")
    print()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def setup_nnunet_env(cfg: dict) -> None:
    """Create working directories and set nnUNet environment variables."""
    rt = cfg["runtime"]
    dirs = {
        "smoke_data": rt / "smoke_data",
        "nnunet_raw": rt / "nnunet_raw",
        "nnunet_preprocessed": rt / "nnunet_preprocessed",
        "nnunet_results": rt / "nnunet_results",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(dirs["nnunet_raw"])
    os.environ["nnUNet_preprocessed"] = str(dirs["nnunet_preprocessed"])
    os.environ["nnUNet_results"] = str(dirs["nnunet_results"])
    os.environ["OMP_NUM_THREADS"] = "1"

    cfg.update(dirs)


def create_data_subset(cfg: dict) -> None:
    """Create symlinks for a subset of BraTS subjects."""
    print("[1/7] Creating data subset...")
    src_root = cfg["data_root"]
    dst_root = cfg["smoke_data"]

    subjects = sorted(d.name for d in src_root.iterdir() if d.is_dir())
    selected = subjects[: cfg["num_subjects"]]

    for subj in selected:
        dst = dst_root / subj
        if not dst.exists() and not dst.is_symlink():
            os.symlink(src_root / subj, dst)

    cfg["subjects"] = selected
    print(f"  Linked {len(selected)} subjects to {dst_root}")


def _find_file(directory: Path, suffix: str) -> Path | None:
    """Find a file in directory matching a suffix (e.g. '-t2f.nii.gz')."""
    for f in directory.iterdir():
        if f.name.endswith(suffix):
            return f
    return None


def create_nnunet_raw_dataset(cfg: dict) -> None:
    """Manually create nnUNet raw dataset format (Dataset<ID>_<Name>/).

    Bypasses MONAI's ``convert_dataset`` which mangles the dataset ID.
    Creates symlinks into imagesTr/labelsTr and writes dataset.json.
    """
    print("[2/7] Creating nnUNet raw dataset...")
    did = cfg["dataset_id"]
    dataset_dir = cfg["nnunet_raw"] / f"Dataset{did:04d}_BraTS2023"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    modality = cfg["modality"]  # e.g. "t2f"
    mod_suffix = f"-{modality}.nii.gz"
    seg_suffix = "-seg.nii.gz"
    smoke_data = cfg["smoke_data"]

    count = 0
    for subj_name in cfg["subjects"]:
        subj_dir = smoke_data / subj_name
        img_file = _find_file(subj_dir, mod_suffix)
        lbl_file = _find_file(subj_dir, seg_suffix)

        if img_file is None or lbl_file is None:
            print(f"  WARNING: skipping {subj_name} (missing files)")
            continue

        # nnUNet naming: <case_id>_0000.nii.gz for images, <case_id>.nii.gz for labels
        case_id = f"case_{count:03d}"
        img_link = images_tr / f"{case_id}_0000.nii.gz"
        lbl_link = labels_tr / f"{case_id}.nii.gz"

        # Remove existing links before creating new ones
        if img_link.exists() or img_link.is_symlink():
            img_link.unlink()
        if lbl_link.exists() or lbl_link.is_symlink():
            lbl_link.unlink()

        # Resolve through symlinks to get the real file path
        img_link.symlink_to(img_file.resolve())
        lbl_link.symlink_to(lbl_file.resolve())
        count += 1

    # dataset.json — BraTS2023 has 3 foreground labels: NCR, ED, ET
    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "NCR_NET": 1,
            "ED": 2,
            "ET": 3,
        },
        "numTraining": count,
        "file_ending": ".nii.gz",
        "description": "BraTS2023 Smoke Test",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    cfg["dataset_name"] = dataset_dir.name
    print(f"  Created {dataset_dir.name} with {count} cases")


def install_custom_trainer(cfg: dict) -> None:
    """Install a custom nnUNet trainer with limited epochs.

    Creates the trainer file in nnUNet's ``variants/`` directory so the
    built-in ``recursive_find_python_class`` can discover it.

    The trainer must match the parent's ``__init__`` signature exactly
    because ``nnUNetTrainer.__init__`` uses ``inspect.signature(self.__init__)``
    to store init kwargs for reproducibility.
    """
    print("[3/7] Installing custom trainer...")
    import inspect
    import nnunetv2
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    # Inspect parent's __init__ signature to match it exactly
    import torch
    sig = inspect.signature(nnUNetTrainer.__init__)
    params = list(sig.parameters.values())  # includes 'self'

    def fmt_default(p):
        """Format a default value as valid Python source."""
        if p.default is inspect.Parameter.empty:
            return None
        v = p.default
        if isinstance(v, torch.device):
            return f"torch.device({v.type!r})"
        return repr(v)

    # Build parameter list string
    param_strs = ["self"]
    for p in params:
        if p.name == "self":
            continue
        default = fmt_default(p)
        if default is not None:
            param_strs.append(f"{p.name}={default}")
        else:
            param_strs.append(p.name)
    param_str = ", ".join(param_strs)

    # Build super() call with positional args (skip self)
    super_args = ", ".join(p.name for p in params if p.name != "self")

    epochs = cfg["epochs"]
    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    trainer_file = trainer_dir / "nnUNetTrainer_smoke.py"
    trainer_file.write_text(
        '"""Smoke test trainer."""\n'
        "import torch\n"
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n\n"
        f"class nnUNetTrainer_smoke(nnUNetTrainer):\n"
        f"    def __init__({param_str}):\n"
        f"        super().__init__({super_args})\n"
        f"        self.num_epochs = {epochs}\n",
    )

    # Verify discoverable
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    cls = recursive_find_python_class(
        Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer",
        "nnUNetTrainer_smoke",
        "nnunetv2.training.nnUNetTrainer",
    )
    if cls is None:
        print("  ERROR: Custom trainer not discoverable by nnUNet!")
        sys.exit(1)

    print(f"  Installed {trainer_file.name} ({epochs} epochs)")


def _get_runner(cfg: dict) -> object:
    """Create a configured nnUNetV2Runner for planning/training/validation."""
    from monai.apps.nnunet import nnUNetV2Runner

    input_config = {
        "datalist": "",  # placeholder — not used (dataset already created)
        "dataroot": "",
        "modality": "MRI",
        "nnunet_raw": str(cfg["nnunet_raw"]),
        "nnunet_preprocessed": str(cfg["nnunet_preprocessed"]),
        "nnunet_results": str(cfg["nnunet_results"]),
        "dataset_name_or_id": str(cfg["dataset_id"]),
    }
    return nnUNetV2Runner(
        input_config=input_config,
        trainer_class_name="nnUNetTrainer_smoke",
        export_validation_probabilities=True,
    )


def plan_and_preprocess(cfg: dict) -> None:
    """Run nnUNet fingerprinting, planning, and preprocessing."""
    print("[4/7] Planning and preprocessing (this takes a few minutes)...")
    runner = _get_runner(cfg)
    runner.plan_and_process(verify_dataset_integrity=True)
    print("  Done")


def train(cfg: dict) -> None:
    """Train a single model (1 fold)."""
    print(f"[5/7] Training ({cfg['config']}, fold {cfg['fold']}, {cfg['epochs']} epochs)...")
    runner = _get_runner(cfg)
    runner.train_single_model(
        config=cfg["config"],
        fold=cfg["fold"],
        gpu_id=cfg["gpu_id"],
    )
    print("  Done")


def validate(cfg: dict) -> None:
    """Run validation on the trained model."""
    print("[6/7] Validating...")
    runner = _get_runner(cfg)
    # Bypass runner.validate() which passes --only_run_validation (unrecognized
    # by nnunetv2>=2.7). Use train_single_model with val=True instead.
    runner.train_single_model(config=cfg["config"], fold=cfg["fold"], val=True)
    print("  Done")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_results(cfg: dict) -> None:
    """Find and print training/validation results."""
    print("[7/7] Collecting results...\n")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    results_dir = cfg["nnunet_results"]
    dataset_dirs = sorted(results_dir.glob("Dataset*"))
    if not dataset_dirs:
        print("  No results directory found!")
        return

    dataset_dir = dataset_dirs[0]
    print(f"  Results dir: {dataset_dir}")

    trainer_dir = dataset_dir / f"nnUNetTrainer_smoke__{cfg['config']}__nnUNetPlans"
    fold_dir = trainer_dir / f"fold_{cfg['fold']}"

    # Check checkpoint
    ckpt = fold_dir / "checkpoint_final.pth"
    if ckpt.exists():
        print(f"  Final checkpoint: {ckpt.stat().st_size / 1e6:.1f} MB")

    # Training log — nnUNet writes epoch metrics to progress.json
    progress_json = fold_dir / "progress.json"
    if progress_json.exists():
        with open(progress_json) as f:
            log = json.load(f)
        if isinstance(log, list) and log:
            first, last = log[0], log[-1]
            ep_first = first.get("epoch", "?")
            ep_last = last.get("epoch", "?")
            tl_first = first.get("train_loss", "?")
            tl_last = last.get("train_loss", "?")
            vl_first = first.get("val_loss", "?")
            vl_last = last.get("val_loss", "?")
            print(f"  Epochs:       {ep_first} -> {ep_last}")
            print(f"  Train loss:   {tl_first} -> {tl_last}")
            print(f"  Val loss:     {vl_first} -> {vl_last}")
            # Try common metric keys
            for key in ("dice", "mean_dice", "val_dice", "validation_dice"):
                if key in last:
                    print(f"  Val {key}: {last[key]}")

    # Validation summary (written by runner.validate)
    val_summary = trainer_dir / "validation" / "summary.json"
    if val_summary.exists():
        with open(val_summary) as f:
            summary = json.load(f)
        print("\n  Validation Summary:")
        if "mean" in summary:
            for k, v in summary["mean"].items():
                print(f"    mean {k}: {v}")
        if "results" in summary:
            print(f"    ({len(summary['results'])} cases evaluated)")

    # Check for training curves plot
    progress_png = fold_dir / "progress.png"
    if progress_png.exists():
        print(f"\n  Training curves: {progress_png}")


if __name__ == "__main__":
    main()
