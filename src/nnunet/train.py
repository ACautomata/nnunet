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

from nnunet.data import (
    create_nnunet_raw_dataset,
    create_random_modality_dataset,
    generate_grouped_splits,
    prepare_datalist,
)

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
    """Build the dict consumed by ``nnUNetV2Runner``.

    ``datalist`` is only consumed by ``convert_dataset`` (which the
    pipeline no longer uses) and by ``MONAI``'s ``analyze_data`` during
    conversion.  Use an empty string as a safe placeholder for
    prepare-free modes (plan / train / validate).
    """
    datalist_val = cfg.get("datalist", "")
    if datalist_val is None or str(datalist_val) == "???":
        datalist_val = ""
    return {
        "datalist": str(datalist_val),
        "dataroot": str(cfg.get("dataroot", "")),
        "modality": cfg.get("modality", "MRI"),
        "nnunet_raw": cfg.nnunet_raw,
        "nnunet_preprocessed": cfg.nnunet_preprocessed,
        "nnunet_results": cfg.nnunet_results,
        "dataset_name_or_id": str(cfg.get("dataset_name_or_id", 1001)),
    }


def run_prepare(cfg: DictConfig) -> None:
    """Step 1: generate datalist and build the nnUNet raw dataset.

    Bypasses ``nnUNetV2Runner.convert_dataset()`` because MONAI 1.5.2's
    ID-naming formula (``str(int(dataset_id) + 1000)[-3:]``) mis-maps
    four-digit IDs such as ``1001`` to ``Dataset001_``, while the
    rest of the runner keeps looking up ID ``1001``.  We build the
    raw dataset directory directly and let planning / training work
    against the correct folder.
    """
    data_root = str(cfg.dataroot)
    modality = cfg.get("modality_file", "t2f")
    output_json = os.path.join(str(cfg.nnunet_raw), "brats2023_datalist.json")

    logger.info("Preparing BraTS2023 datalist for modality=%s", modality)
    prepare_datalist(
        data_root=data_root,
        modality=modality,
        output_json=output_json,
    )

    dataset_id = int(cfg.get("dataset_name_or_id", 1001))
    dataset_dir = create_nnunet_raw_dataset(
        data_root=data_root,
        nnunet_raw=str(cfg.nnunet_raw),
        modality=modality,
        dataset_id=dataset_id,
    )
    logger.info("Raw dataset created at %s", dataset_dir)


def run_prepare_random_modality(cfg: DictConfig) -> None:
    """Step 1 for random-modality pipeline: build augmented nnUNet raw dataset.

    Creates one case per subject×modality combination.  nnUNet's shuffling
    produces random modality sampling during training.  Uses dataset ID 2001
    by default to avoid collision with single-modality datasets.

    Also generates a grouped ``splits_final.json`` so that all modality-derived
    cases from the same subject stay in the same fold, preventing subject-level
    leakage in cross-validation.
    """
    import json as _json

    data_root = str(cfg.dataroot)

    # Accept modalities as a list or a space-separated string.
    modalities_cfg = cfg.get("modalities", "t1c t1n t2f t2w")
    if isinstance(modalities_cfg, str):
        modalities = modalities_cfg.split()
    elif hasattr(modalities_cfg, "__iter__"):
        modalities = list(modalities_cfg)
    else:
        modalities = ["t1c", "t1n", "t2f", "t2w"]

    dataset_id = int(cfg.get("dataset_name_or_id", 2001))

    logger.info(
        "Preparing random-modality dataset: modalities=%s dataset_id=%s",
        modalities, dataset_id,
    )
    dataset_dir, subject_to_cases = create_random_modality_dataset(
        data_root=data_root,
        nnunet_raw=str(cfg.nnunet_raw),
        modalities=modalities,
        dataset_id=dataset_id,
    )
    logger.info("Random-modality raw dataset created at %s", dataset_dir)

    # Write grouped splits so nnUNet keeps all cases from the same subject
    # in the same fold.  This must go into the preprocessed directory which
    # nnUNet creates during planning.  Since planning hasn't run yet, write
    # it to a temporary location and let run_plan move it after preprocessing.
    n_folds = int(cfg.get("num_folds", 5))
    splits = generate_grouped_splits(subject_to_cases, n_folds=n_folds)
    splits_tmp = os.path.join(str(cfg.nnunet_raw), f"_grouped_splits_{dataset_id}.json")
    with open(splits_tmp, "w") as f:
        _json.dump(splits, f, indent=2)
    logger.info(
        "Grouped splits (%d folds) written to %s. "
        "Will be copied to preprocessed dir after planning.",
        n_folds, splits_tmp,
    )


def _install_grouped_splits(cfg: DictConfig) -> None:
    """Copy grouped splits into the nnUNet preprocessed directory.

    nnUNet reads ``splits_final.json`` from the preprocessed dataset folder.
    Planning creates the folder, so this must run *after* :func:`run_plan`.
    If nnUNet already generated its own splits, our grouped version
    overwrites them to enforce subject-level grouping.
    """
    import shutil

    dataset_id = int(cfg.get("dataset_name_or_id", 2001))
    splits_tmp = os.path.join(str(cfg.nnunet_raw), f"_grouped_splits_{dataset_id}.json")
    if not os.path.isfile(splits_tmp):
        logger.warning("Grouped splits file not found at %s — skipping install", splits_tmp)
        return

    # Find the preprocessed dataset directory.
    preprocessed_base = Path(str(cfg.nnunet_preprocessed))
    # nnUNet names it Dataset{ID:04d}_<name> but may use a different suffix
    # than our raw dataset.  Search by ID prefix.
    candidates = list(preprocessed_base.glob(f"Dataset{dataset_id:04d}_*"))
    if not candidates:
        logger.warning(
            "Preprocessed dataset dir not found under %s for ID %d — skipping splits install",
            preprocessed_base, dataset_id,
        )
        return

    target = candidates[0] / "splits_final.json"
    shutil.copy2(splits_tmp, target)
    logger.info("Installed grouped splits at %s", target)


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
    train_config = cfg.get("train_config", None)
    if train_config is not None:
        # Match run_train: with train_config, train only the single fold
        # the user requested. Validating the full 5 folds when only one
        # was trained would be meaningless (no checkpoint exists for the
        # others) and slow.
        configs = (train_config,)
        folds = range(cfg.get("fold", 0), cfg.get("fold", 0) + 1)
    else:
        folds = range(cfg.get("num_folds", 5))

    logger.info("Validating configs=%s", configs)
    # Bypass runner.validate() which passes --only_run_validation (unrecognized
    # by nnunetv2>=2.7). Use train_single_model with val=True instead, which
    # correctly generates the --val flag.
    for config in configs:
        for fold in folds:
            runner.train_single_model(config=config, fold=fold, val=True)


# ---------- Hydra entry point ----------

_MODES = {
    "prepare": run_prepare,
    "plan": run_plan,
    "train": run_train,
    "validate": run_validate,
    "prepare_random_modality": run_prepare_random_modality,
}

# Modes that skip the datalist step (random-modality pipeline doesn't use it).
_RANDOM_MODALITY_MODES = {"random_modality", "random_modality_train"}


@hydra.main(version_base=None, config_path="../../configs", config_name="brats2023")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    mode = cfg.get("mode", "all")

    # Validate that paths are overridden (not left as ???).
    # ``datalist`` is only needed by mode=prepare; the new prepare step
    # generates the nnUNet raw dataset directly so it is optional.
    required = ("dataroot", "nnunet_raw", "nnunet_preprocessed", "nnunet_results")
    for key in required:
        val = cfg.get(key)
        if val is None or str(val) == "???":
            logger.error(
                "Path %s is not set. Provide it via CLI, e.g. %s=/some/path",
                key, key,
            )
            sys.exit(1)

    _ensure_dirs(cfg)

    if mode == "all":
        # Full pipeline
        run_prepare(cfg)
        run_plan(cfg)
        run_train(cfg)
    elif mode == "random_modality":
        # Random-modality full pipeline: prepare → plan → copy splits → train
        run_prepare_random_modality(cfg)
        run_plan(cfg)
        _install_grouped_splits(cfg)
        run_train(cfg)
    elif mode == "random_modality_train":
        # Skip data prep and planning, only train
        run_train(cfg)
    elif mode in _MODES:
        _MODES[mode](cfg)
    else:
        logger.error(
            "Unknown mode %r. Choose from: all, random_modality, random_modality_train, %s",
            mode, ", ".join(_MODES),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
