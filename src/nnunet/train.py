"""Training and validation CLI for BraTS2023 single-modality nnUNet segmentation.

Usage examples::

    # Full pipeline: data prep + plan + preprocess + train
    python -m nnunet.train \\
        data_root=/data/BraTS2023/GLI_train \\
        nnunet_raw=/work/nnunet_raw \\
        nnunet_preprocessed=/work/nnunet_preprocessed \\
        nnunet_results=/work/nnunet_results

    # Only train (skip data prep and planning)
    python -m nnunet.train mode=train \\
        data_root=/data/BraTS2023/GLI_train \\
        nnunet_raw=/work/nnunet_raw \\
        nnunet_preprocessed=/work/nnunet_preprocessed \\
        nnunet_results=/work/nnunet_results

    # Only validate
    python -m nnunet.train mode=validate \\
        data_root=/data/BraTS2023/GLI_train \\
        nnunet_raw=/work/nnunet_raw \\
        nnunet_preprocessed=/work/nnunet_preprocessed \\
        nnunet_results=/work/nnunet_results
"""

from __future__ import annotations

import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from nnunet.data import prepare_datalist

logger = logging.getLogger(__name__)


def _ensure_dirs(cfg: DictConfig) -> None:
    """Create nnUNet working directories and set env vars."""
    for key in ("nnunet_raw", "nnunet_preprocessed", "nnunet_results"):
        path = OmegaConf.to_container(cfg[key], resolve=True) if OmegaConf.is_config(cfg[key]) else cfg[key]
        path = str(path)
        os.makedirs(path, exist_ok=True)
        os.environ[key] = path
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _build_input_config(cfg: DictConfig) -> dict:
    """Build the dict consumed by ``nnUNetV2Runner``."""
    return {
        "datalist": cfg.datalist,
        "dataroot": cfg.dataroot,
        "modality": cfg.get("modality", "MRI"),
        "nnunet_raw": cfg.nnunet_raw,
        "nnunet_preprocessed": cfg.nnunet_preprocessed,
        "nnunet_results": cfg.nnunet_results,
        "dataset_name_or_id": str(cfg.get("dataset_name_or_id", 1001)),
    }


def run_prepare(cfg: DictConfig) -> None:
    """Step 1: generate datalist and convert to nnUNet format."""
    from monai.apps.nnunet import nnUNetV2Runner

    data_root = str(cfg.dataroot)
    modality = cfg.get("modality_file", "t2f")
    output_json = os.path.join(str(cfg.nnunet_raw), "brats2023_datalist.json")

    logger.info("Preparing BraTS2023 datalist for modality=%s", modality)
    prepare_datalist(
        data_root=data_root,
        modality=modality,
        output_json=output_json,
    )

    input_config = _build_input_config(cfg)
    input_config["datalist"] = output_json

    runner = nnUNetV2Runner(
        input_config=input_config,
        trainer_class_name=cfg.get("trainer_class_name", "nnUNetTrainer"),
        export_validation_probabilities=cfg.get("export_validation_probabilities", True),
    )
    logger.info("Converting dataset to nnUNet format...")
    runner.convert_dataset()


def run_plan(cfg: DictConfig) -> None:
    """Step 2: fingerprint extraction, planning, preprocessing."""
    from monai.apps.nnunet import nnUNetV2Runner

    input_config = _build_input_config(cfg)
    runner = nnUNetV2Runner(input_config=input_config)

    logger.info("Running experiment planning and preprocessing...")
    runner.plan_and_process()


def run_train(cfg: DictConfig) -> None:
    """Step 3: train models."""
    from monai.apps.nnunet import nnUNetV2Runner

    input_config = _build_input_config(cfg)
    trainer_name = cfg.get("trainer_class_name", "nnUNetTrainer")
    export_val = cfg.get("export_validation_probabilities", True)

    runner = nnUNetV2Runner(
        input_config=input_config,
        trainer_class_name=trainer_name,
        export_validation_probabilities=export_val,
    )

    # Support training a single config + fold or the full suite.
    train_config = cfg.get("train_config", None)
    gpu_id = cfg.get("gpu_id", 0)
    num_folds = cfg.get("num_folds", 5)

    if train_config is not None:
        fold = cfg.get("fold", 0)
        logger.info("Training single model: config=%s fold=%s gpu=%s", train_config, fold, gpu_id)
        runner.train_single_model(config=train_config, fold=fold, gpu_id=gpu_id)
    else:
        configs = cfg.get("configs", ("3d_fullres",))
        logger.info("Training configs=%s on gpu=%s", configs, gpu_id)
        runner.train(configs=configs, gpu_id_for_all=gpu_id)


def run_validate(cfg: DictConfig) -> None:
    """Step 3b: validate trained models."""
    from monai.apps.nnunet import nnUNetV2Runner

    input_config = _build_input_config(cfg)
    trainer_name = cfg.get("trainer_class_name", "nnUNetTrainer")
    export_val = cfg.get("export_validation_probabilities", True)

    runner = nnUNetV2Runner(
        input_config=input_config,
        trainer_class_name=trainer_name,
        export_validation_probabilities=export_val,
    )

    configs = cfg.get("configs", ("3d_fullres",))
    num_folds = cfg.get("num_folds", 5)

    logger.info("Validating configs=%s", configs)
    # Bypass runner.validate() which passes --only_run_validation (unrecognized
    # by nnunetv2>=2.7). Use train_single_model with val=True instead, which
    # correctly generates the --val flag.
    for config in configs:
        for fold in range(num_folds):
            runner.train_single_model(config=config, fold=fold, val=True)


# ---------- Hydra entry point ----------

_MODES = {
    "prepare": run_prepare,
    "plan": run_plan,
    "train": run_train,
    "validate": run_validate,
}


@hydra.main(version_base=None, config_path="../../configs", config_name="brats2023")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    mode = cfg.get("mode", "all")

    # Validate that paths are overridden (not left as ???)
    always_required = ("dataroot", "nnunet_raw", "nnunet_preprocessed", "nnunet_results")
    if mode in ("all", "prepare"):
        always_required = ("datalist",) + always_required
    for key in always_required:
        val = cfg.get(key)
        if val is None or str(val) == "???":
            logger.error(
                "Path %s is not set. Provide it via CLI, e.g. %s=/some/path",
                key, key,
            )
            sys.exit(1)

    _ensure_dirs(cfg)

    mode = cfg.get("mode", "all")
    if mode == "all":
        # Full pipeline
        run_prepare(cfg)
        run_plan(cfg)
        run_train(cfg)
    elif mode in _MODES:
        _MODES[mode](cfg)
    else:
        logger.error("Unknown mode %r. Choose from: all, %s", mode, ", ".join(_MODES))
        sys.exit(1)


if __name__ == "__main__":
    main()
