"""Prediction / testing CLI for BraTS2023 single-modality nnUNet segmentation.

Usage examples::

    # Predict on test folder
    python -m nnunet.predict \\
        input_folder=/data/BraTS2023/GLI_test \\
        output_folder=/work/predictions \\
        model_folder=/work/nnunet_results/Dataset999_BraTS2023/nnUNetTrainer__3d_fullres__nnUNetPlans \\
        nnunet_raw=/work/nnunet_raw \\
        nnunet_preprocessed=/work/nnunet_preprocessed \\
        nnunet_results=/work/nnunet_results

    # Predict with specific folds and GPU
    python -m nnunet.predict \\
        input_folder=/data/test \\
        output_folder=/work/pred \\
        model_folder=/work/models/nnUNetTrainer__3d_fullres__nnUNetPlans \\
        use_folds=[0,1,2,3,4] \\
        gpu_id=0 \\
        nnunet_raw=/work/nnunet_raw \\
        nnunet_preprocessed=/work/nnunet_preprocessed \\
        nnunet_results=/work/nnunet_results
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _is_brats_subject_dir(folder: str) -> bool:
    """Heuristic: does *folder* contain BraTS-style subject sub-folders?"""
    p = Path(folder)
    if not p.is_dir():
        return False
    for entry in p.iterdir():
        if entry.is_dir():
            # BraTS sub-folders contain .nii.gz files
            nii_files = list(entry.glob("*.nii.gz"))
            if nii_files:
                return True
    return False


def _ensure_dirs(cfg: DictConfig) -> None:
    for key in ("nnunet_raw", "nnunet_preprocessed", "nnunet_results"):
        path = str(cfg[key])
        os.makedirs(path, exist_ok=True)
        os.environ[key] = path
    os.environ.setdefault("OMP_NUM_THREADS", "1")


@hydra.main(version_base=None, config_path="../../configs", config_name="brats2023")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Required path overrides
    for key in ("input_folder", "output_folder", "model_folder",
                "nnunet_raw", "nnunet_preprocessed", "nnunet_results"):
        val = cfg.get(key)
        if val is None or str(val) == "???":
            logger.error("Path %s is not set. Provide it via CLI: %s=/some/path", key, key)
            sys.exit(1)

    _ensure_dirs(cfg)

    from monai.apps.nnunet import nnUNetV2Runner

    from nnunet.data import prepare_inference_folder

    # If input_folder has BraTS2023 per-subject sub-folders, create a flat
    # nnUNet-compatible folder with _0000.nii.gz naming.
    modality_file = cfg.get("modality_file", "t2f")
    input_folder = str(cfg.input_folder)
    if _is_brats_subject_dir(input_folder):
        logger.info("Detected BraTS2023 subject layout, creating nnUNet-compatible folder...")
        flat_folder = os.path.join(str(cfg.nnunet_raw), "_inference_input")
        input_folder = prepare_inference_folder(
            data_root=str(cfg.input_folder),
            output_folder=flat_folder,
            modality=modality_file,
        )

    input_config = {
        "datalist": "",   # not needed for prediction
        "dataroot": "",
        "modality": cfg.get("modality", "MRI"),
        "nnunet_raw": cfg.nnunet_raw,
        "nnunet_preprocessed": cfg.nnunet_preprocessed,
        "nnunet_results": cfg.nnunet_results,
        "dataset_name_or_id": str(cfg.get("dataset_name_or_id", 1001)),
    }

    runner = nnUNetV2Runner(input_config=input_config)

    use_folds = cfg.get("use_folds", [0, 1, 2, 3, 4])
    # Normalize OmegaConf ListConfig / int / string -> tuple[int, ...]
    if isinstance(use_folds, ListConfig):
        use_folds = tuple(OmegaConf.to_container(use_folds))
    elif isinstance(use_folds, (list, tuple)):
        use_folds = tuple(use_folds)
    elif isinstance(use_folds, int):
        use_folds = (use_folds,)
    elif isinstance(use_folds, str):
        try:
            use_folds = tuple(int(x) for x in use_folds.strip("[]()").split(","))
        except ValueError as e:
            logger.error("Invalid use_folds value %r: %s", use_folds, e)
            sys.exit(1)

    # Validate input folder has actual nnUNet-format files
    if not _is_brats_subject_dir(input_folder):
        nii_files = list(Path(input_folder).glob("*_0000.nii.gz")) if Path(input_folder).is_dir() else []
        if not nii_files:
            logger.error(
                "Input folder %s contains no nnUNet-format files (*_0000.nii.gz) "
                "and no BraTS subject layout was detected.",
                input_folder,
            )
            sys.exit(1)

    logger.info(
        "Predicting:\n  input:  %s\n  output: %s\n  model:  %s\n  folds:  %s",
        input_folder, cfg.output_folder, cfg.model_folder, use_folds,
    )

    runner.predict(
        list_of_lists_or_source_folder=input_folder,
        output_folder=str(cfg.output_folder),
        model_training_output_dir=str(cfg.model_folder),
        use_folds=use_folds,
        tile_step_size=float(cfg.get("tile_step_size", 0.5)),
        use_gaussian=bool(cfg.get("use_gaussian", True)),
        use_mirroring=bool(cfg.get("use_mirroring", True)),
        save_probabilities=bool(cfg.get("save_probabilities", False)),
        checkpoint_name=cfg.get("checkpoint_name", "checkpoint_final.pth"),
        gpu_id=int(cfg.get("gpu_id", 0)),
    )

    logger.info("Prediction complete. Results saved to %s", cfg.output_folder)


if __name__ == "__main__":
    main()
