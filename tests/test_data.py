"""Tests for nnunet.data module."""
import json
import os
import tempfile

import pytest

from nnunet.data import _find_modality_file, _find_seg_file, prepare_datalist, prepare_inference_folder


def _make_brats_subjects(root, n_train=4, n_test=2):
    """Create fake BraTS2023 subject folders under *root*."""
    for i in range(n_train):
        subj = os.path.join(root, f"BraTS-GLI-{i:05d}-000")
        os.makedirs(subj, exist_ok=True)
        for mod in ("t1c", "t1n", "t2f", "t2w"):
            open(os.path.join(subj, f"{mod}.nii.gz"), "w").close()
        open(os.path.join(subj, "seg.nii.gz"), "w").close()
    for i in range(n_train, n_train + n_test):
        subj = os.path.join(root, f"BraTS-GLI-{i:05d}-000")
        os.makedirs(subj, exist_ok=True)
        for mod in ("t1c", "t1n", "t2f", "t2w"):
            open(os.path.join(subj, f"{mod}.nii.gz"), "w").close()


class TestPrepareDatalist:
    def test_default_train_ratio_puts_all_labeled_in_training(self, tmp_path):
        """With default train_ratio=1.0, all labeled subjects go to training."""
        _make_brats_subjects(tmp_path, n_train=4, n_test=2)
        result = prepare_datalist(data_root=str(tmp_path), modality="t2f")
        assert len(result["training"]) == 4
        assert len(result["test"]) == 2  # only unlabeled subjects
        # Test entries should NOT have "label" key
        for entry in result["test"]:
            assert "label" not in entry
            assert "image" in entry

    def test_custom_train_ratio(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=4, n_test=2)
        result = prepare_datalist(data_root=str(tmp_path), modality="t2f", train_ratio=0.75, seed=0)
        assert len(result["training"]) == 3  # 4 * 0.75 = 3
        assert len(result["test"]) == 3  # 1 held-out (no label) + 2 unlabeled
        # Held-out subjects in test should NOT have labels
        for entry in result["test"]:
            assert "label" not in entry

    def test_training_entries_have_image_and_label(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        result = prepare_datalist(data_root=str(tmp_path), modality="t2f")
        for entry in result["training"]:
            assert "image" in entry
            assert "label" in entry

    def test_all_modalities_work(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        for mod in ("t1c", "t1n", "t2f", "t2w"):
            result = prepare_datalist(data_root=str(tmp_path), modality=mod)
            assert len(result["training"]) == 2

    def test_writes_json(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        out_json = str(tmp_path / "out.json")
        result = prepare_datalist(data_root=str(tmp_path), modality="t2f", output_json=out_json)
        assert os.path.isfile(out_json)
        with open(out_json) as f:
            data = json.load(f)
        # Verify JSON round-trips correctly
        assert data == result
        assert "training" in data
        assert "test" in data
        # Verify schema
        for entry in data["training"]:
            assert "image" in entry
            assert "label" in entry

    def test_invalid_modality_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown modality"):
            prepare_datalist(data_root=str(tmp_path), modality="invalid")

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            prepare_datalist(data_root="/nonexistent/path", modality="t2f")

    def test_empty_root_raises(self, tmp_path):
        """Empty data_root should raise ValueError, not silently return empty datalist."""
        with pytest.raises(ValueError, match="No BraTS2023 subjects"):
            prepare_datalist(data_root=str(tmp_path), modality="t2f")

    def test_train_ratio_zero_raises(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        with pytest.raises(ValueError, match="train_ratio must be in"):
            prepare_datalist(data_root=str(tmp_path), modality="t2f", train_ratio=0.0)

    def test_train_ratio_negative_raises(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        with pytest.raises(ValueError, match="train_ratio must be in"):
            prepare_datalist(data_root=str(tmp_path), modality="t2f", train_ratio=-0.5)

    def test_train_ratio_one(self, tmp_path):
        """train_ratio=1.0 is valid: all labeled subjects in training, no held-out."""
        _make_brats_subjects(tmp_path, n_train=3, n_test=1)
        result = prepare_datalist(data_root=str(tmp_path), modality="t2f", train_ratio=1.0)
        assert len(result["training"]) == 3
        assert len(result["test"]) == 1  # only unlabeled subjects


class TestFindFiles:
    def test_direct_naming_returns_relative_path(self, tmp_path):
        subj = tmp_path / "BraTS-GLI-00000-000"
        subj.mkdir()
        (subj / "t2f.nii.gz").touch()
        result = _find_modality_file(subj, "t2f.nii.gz")
        assert result == os.path.join("BraTS-GLI-00000-000", "t2f.nii.gz")

    def test_prefixed_naming_returns_relative_path(self, tmp_path):
        subj = tmp_path / "BraTS-GLI-00000-000"
        subj.mkdir()
        (subj / "BraTS-GLI-00000-000-t2f.nii.gz").touch()
        result = _find_modality_file(subj, "t2f.nii.gz")
        assert result == os.path.join("BraTS-GLI-00000-000", "BraTS-GLI-00000-000-t2f.nii.gz")

    def test_no_match_returns_none(self, tmp_path):
        subj = tmp_path / "BraTS-GLI-00000-000"
        subj.mkdir()
        assert _find_modality_file(subj, "t2f.nii.gz") is None

    def test_seg_file_returns_relative_path(self, tmp_path):
        subj = tmp_path / "sub-001"
        subj.mkdir()
        (subj / "seg.nii.gz").touch()
        result = _find_seg_file(subj)
        assert result == os.path.join("sub-001", "seg.nii.gz")

    def test_prefixed_seg_file(self, tmp_path):
        subj = tmp_path / "BraTS-GLI-00000-000"
        subj.mkdir()
        (subj / "BraTS-GLI-00000-000-seg.nii.gz").touch()
        result = _find_seg_file(subj)
        assert result == os.path.join("BraTS-GLI-00000-000", "BraTS-GLI-00000-000-seg.nii.gz")

    def test_no_seg_returns_none(self, tmp_path):
        subj = tmp_path / "sub-001"
        subj.mkdir()
        (subj / "t2f.nii.gz").touch()
        assert _find_seg_file(subj) is None


class TestPrepareInferenceFolder:
    def test_creates_flat_symlinks(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=0, n_test=3)
        out = str(tmp_path / "flat")
        result = prepare_inference_folder(data_root=str(tmp_path), output_folder=out, modality="t2f")
        assert result == out
        files = os.listdir(out)
        assert len(files) == 3
        for f in files:
            assert f.endswith("_0000.nii.gz")
            # Verify symlink target resolves correctly
            target = os.path.realpath(os.path.join(out, f))
            assert os.path.isfile(target)

    def test_invalid_modality_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown modality"):
            prepare_inference_folder(
                data_root=str(tmp_path), output_folder=str(tmp_path / "out"), modality="bad"
            )

    def test_replaces_existing_symlinks(self, tmp_path):
        """If output folder already has stale symlinks, they should be replaced."""
        _make_brats_subjects(tmp_path, n_train=0, n_test=2)
        out = str(tmp_path / "flat")
        os.makedirs(out, exist_ok=True)
        # Create a stale symlink for the first subject
        stale = os.path.join(out, "BraTS-GLI-00000-000_0000.nii.gz")
        os.symlink("/nonexistent/file.nii.gz", stale)
        assert os.path.islink(stale)

        prepare_inference_folder(data_root=str(tmp_path), output_folder=out, modality="t2f")
        # Symlink should now point to a valid target
        assert os.path.islink(stale)
        assert os.path.isfile(os.path.realpath(stale))

    def test_empty_root_produces_empty_folder(self, tmp_path):
        """Empty data_root returns an empty output folder (no error)."""
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        out = str(tmp_path / "flat")
        result = prepare_inference_folder(data_root=str(empty_root), output_folder=out, modality="t2f")
        assert result == out
        assert len(os.listdir(out)) == 0
