"""Tests for nnunet.train module (mocked — does not require nnUNet/MONAI runtime)."""
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nnunet.train import run_validate


def _make_cfg(**overrides):
    """Build a minimal cfg DictConfig for run_validate."""
    base = {
        "trainer_class_name": "nnUNetTrainer",
        "export_validation_probabilities": True,
        "configs": ["3d_fullres"],
        "train_config": None,
        "num_folds": 5,
        "nnunet_raw": "/tmp/raw",
        "nnunet_preprocessed": "/tmp/prep",
        "nnunet_results": "/tmp/res",
        "dataset_name_or_id": "1001",
        "modality": "MRI",
    }
    base.update(overrides)
    return OmegaConf.create(base)


@patch("monai.apps.nnunet.nnUNetV2Runner")
def test_run_validate_honors_train_config(MockRunner):
    """P2 fix: if train_config is set, run_validate should use that single config
    rather than the default configs list (which would silently validate 3d_fullres
    on a model trained with e.g. 2d)."""
    mock = MagicMock()
    MockRunner.return_value = mock

    cfg = _make_cfg(train_config="2d", configs=["3d_fullres"])
    run_validate(cfg)

    calls = mock.train_single_model.call_args_list
    configs_used = sorted({c.kwargs["config"] for c in calls})
    assert configs_used == ["2d"]


@patch("monai.apps.nnunet.nnUNetV2Runner")
def test_run_validate_falls_back_to_configs(MockRunner):
    """When train_config is null, run_validate should validate the full configs list."""
    mock = MagicMock()
    MockRunner.return_value = mock

    cfg = _make_cfg(train_config=None, configs=["3d_fullres", "2d"])
    run_validate(cfg)

    calls = mock.train_single_model.call_args_list
    configs_used = sorted({c.kwargs["config"] for c in calls})
    assert configs_used == ["2d", "3d_fullres"]


@patch("monai.apps.nnunet.nnUNetV2Runner")
def test_run_validate_passes_val_flag(MockRunner):
    """run_validate must pass val=True to bypass MONAI's broken --only_run_validation."""
    mock = MagicMock()
    MockRunner.return_value = mock

    # No train_config -> full configs list, multiple folds.
    cfg = _make_cfg(train_config=None, num_folds=3)
    run_validate(cfg)

    # 1 config * 3 folds = 3 calls
    assert mock.train_single_model.call_count == 3
    for call in mock.train_single_model.call_args_list:
        assert call.kwargs["val"] is True


@patch("monai.apps.nnunet.nnUNetV2Runner")
def test_run_validate_honors_single_fold_when_train_config_set(MockRunner):
    """P2 fix (round 3): with train_config set, run_train only trains one fold.
    run_validate must validate that same single fold, not the full 5."""
    mock = MagicMock()
    MockRunner.return_value = mock

    cfg = _make_cfg(train_config="2d", fold=2, num_folds=5)
    run_validate(cfg)

    folds_validated = [c.kwargs["fold"] for c in mock.train_single_model.call_args_list]
    assert folds_validated == [2]


@patch("monai.apps.nnunet.nnUNetV2Runner")
def test_run_validate_default_fold_when_train_config_set(MockRunner):
    """When train_config is set but fold is unset, validate fold 0."""
    mock = MagicMock()
    MockRunner.return_value = mock

    cfg = _make_cfg(train_config="2d", num_folds=5)
    # cfg.fold is not in _make_cfg defaults -> cfg.get("fold") returns None
    run_validate(cfg)

    folds_validated = [c.kwargs["fold"] for c in mock.train_single_model.call_args_list]
    assert folds_validated == [0]
