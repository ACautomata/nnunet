"""Tests for nnunet.data module."""
import json
import os
import tempfile

import pytest

from nnunet.data import (
    _find_modality_file,
    _find_seg_file,
    create_nnunet_raw_dataset,
    create_random_modality_dataset,
    generate_grouped_splits,
    prepare_datalist,
    prepare_inference_folder,
)


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

    def test_clears_stale_subjects_from_previous_run(self, tmp_path):
        """P2 fix: re-running on a smaller input set removes links for dropped subjects."""
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        out = str(tmp_path / "flat")
        # First pass with both subjects.
        prepare_inference_folder(data_root=str(tmp_path), output_folder=out, modality="t2f")
        first = sorted(os.listdir(out))
        assert len(first) == 2

        # Drop one subject entirely (with its files) so the second pass won't see it.
        import shutil
        shutil.rmtree(str(tmp_path / "BraTS-GLI-00001-000"))
        prepare_inference_folder(data_root=str(tmp_path), output_folder=out, modality="t2f")
        second = sorted(os.listdir(out))
        assert len(second) == 1
        assert second[0].startswith("BraTS-GLI-00000-000")


class TestCreateNnunetRawDataset:
    """P1 fix: build nnUNet raw dataset directly, bypassing MONAI's broken ID mapping."""

    def test_creates_dataset_directory(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=3, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        result = create_nnunet_raw_dataset(
            data_root=str(tmp_path), nnunet_raw=raw,
            modality="t2f", dataset_id=1001,
        )
        assert os.path.isdir(result)
        assert os.path.basename(result) == "Dataset1001_BraTS2023"

    def test_creates_image_and_label_symlinks(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=3, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        result = create_nnunet_raw_dataset(
            data_root=str(tmp_path), nnunet_raw=raw,
            modality="t2f", dataset_id=1001,
        )
        images = sorted(os.listdir(os.path.join(result, "imagesTr")))
        labels = sorted(os.listdir(os.path.join(result, "labelsTr")))
        assert images == [
            "case_000_0000.nii.gz",
            "case_001_0000.nii.gz",
            "case_002_0000.nii.gz",
        ]
        assert labels == ["case_000.nii.gz", "case_001.nii.gz", "case_002.nii.gz"]

    def test_writes_dataset_json(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        result = create_nnunet_raw_dataset(
            data_root=str(tmp_path), nnunet_raw=raw,
            modality="t2f", dataset_id=1001,
        )
        with open(os.path.join(result, "dataset.json")) as f:
            ds = json.load(f)
        assert ds["channel_names"] == {"0": "T2F"}
        assert ds["labels"] == {"background": 0, "NCR_NET": 1, "ED": 2, "ET": 3}
        assert ds["numTraining"] == 2
        assert ds["file_ending"] == ".nii.gz"

    def test_no_subjects_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        raw = str(tmp_path / "nnunet_raw")
        with pytest.raises(ValueError, match="No BraTS2023 subjects"):
            create_nnunet_raw_dataset(
                data_root=str(empty), nnunet_raw=raw, modality="t2f",
            )

    def test_four_digit_id_is_preserved(self, tmp_path):
        """The whole point of the fix: a 4-digit ID stays in the folder name."""
        _make_brats_subjects(tmp_path, n_train=1, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        result = create_nnunet_raw_dataset(
            data_root=str(tmp_path), nnunet_raw=raw,
            modality="t2f", dataset_id=1001,
        )
        # nnUNet looks up by ID 1001 -> Dataset1001_*
        # If we had used MONAI's convert_dataset, this would be Dataset001_*
        assert os.path.basename(result) == "Dataset1001_BraTS2023"

    def test_clears_stale_cases_from_previous_run(self, tmp_path):
        """P2 fix (round 2): re-running with fewer subjects drops the old case links."""
        _make_brats_subjects(tmp_path, n_train=3, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_nnunet_raw_dataset(
            data_root=str(tmp_path), nnunet_raw=raw,
            modality="t2f", dataset_id=1001,
        )
        first_images = sorted(os.listdir(os.path.join(raw, "Dataset1001_BraTS2023", "imagesTr")))
        assert len(first_images) == 3

        # Drop one subject and re-run.
        import shutil
        shutil.rmtree(str(tmp_path / "BraTS-GLI-00002-000"))
        create_nnunet_raw_dataset(
            data_root=str(tmp_path), nnunet_raw=raw,
            modality="t2f", dataset_id=1001,
        )
        second_images = sorted(os.listdir(os.path.join(raw, "Dataset1001_BraTS2023", "imagesTr")))
        assert second_images == ["case_000_0000.nii.gz", "case_001_0000.nii.gz"]


class TestCreateRandomModalityDataset:
    """Tests for the random-modality dataset builder."""

    def test_creates_dataset_with_all_modalities(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=3, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        result, subject_map = create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        assert os.path.isdir(result)
        assert os.path.basename(result) == "Dataset2001_BraTS2023_RandomModality"
        assert len(subject_map) == 3

    def test_case_count_equals_subjects_times_modalities(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=3, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        images = os.listdir(os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "imagesTr"))
        labels = os.listdir(os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "labelsTr"))
        # 3 subjects × 4 modalities = 12 cases
        assert len(images) == 12
        assert len(labels) == 12

    def test_subset_of_modalities(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw,
            modalities=["t1c", "t2f"], dataset_id=2001,
        )
        images = os.listdir(os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "imagesTr"))
        # 2 subjects × 2 modalities = 4 cases
        assert len(images) == 4

    def test_dataset_json_channel_is_generic_mri(self, tmp_path):
        """Channel name should be generic 'MRI', not a specific modality."""
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        with open(os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "dataset.json")) as f:
            ds = json.load(f)
        assert ds["channel_names"] == {"0": "MRI"}
        assert ds["labels"] == {"background": 0, "NCR_NET": 1, "ED": 2, "ET": 3}
        assert ds["numTraining"] == 8  # 2 subjects × 4 modalities

    def test_case_naming_is_sequential(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        images = sorted(os.listdir(os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "imagesTr")))
        # Should use 4-digit case IDs: case_0000 through case_0007
        expected = [f"case_{i:04d}_0000.nii.gz" for i in range(8)]
        assert images == expected

    def test_labels_symlink_to_valid_targets(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        labels_dir = os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "labelsTr")
        for lbl in os.listdir(labels_dir):
            target = os.path.realpath(os.path.join(labels_dir, lbl))
            assert os.path.isfile(target), f"Label symlink {lbl} -> {target} is broken"

    def test_images_symlink_to_valid_targets(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        images_dir = os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "imagesTr")
        for img in os.listdir(images_dir):
            target = os.path.realpath(os.path.join(images_dir, img))
            assert os.path.isfile(target), f"Image symlink {img} -> {target} is broken"

    def test_no_labeled_subjects_raises(self, tmp_path):
        """Subjects without seg.nii.gz should be skipped, and all-skipped raises."""
        _make_brats_subjects(tmp_path, n_train=0, n_test=2)
        raw = str(tmp_path / "nnunet_raw")
        with pytest.raises(ValueError, match="No BraTS2023 subjects"):
            create_random_modality_dataset(
                data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
            )

    def test_invalid_modality_raises(self, tmp_path):
        _make_brats_subjects(tmp_path, n_train=1, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        with pytest.raises(ValueError, match="Unknown modality"):
            create_random_modality_dataset(
                data_root=str(tmp_path), nnunet_raw=raw,
                modalities=["invalid"], dataset_id=2001,
            )

    def test_clears_stale_cases_from_previous_run(self, tmp_path):
        """Re-running with fewer subjects drops old case symlinks."""
        _make_brats_subjects(tmp_path, n_train=3, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        first = sorted(os.listdir(os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "imagesTr")))
        assert len(first) == 12

        import shutil
        shutil.rmtree(str(tmp_path / "BraTS-GLI-00002-000"))
        create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        second = sorted(os.listdir(os.path.join(raw, "Dataset2001_BraTS2023_RandomModality", "imagesTr")))
        assert len(second) == 8  # 2 subjects × 4 modalities

    def test_subject_map_has_correct_cases(self, tmp_path):
        """subject_to_cases maps each subject to its case IDs."""
        _make_brats_subjects(tmp_path, n_train=2, n_test=0)
        raw = str(tmp_path / "nnunet_raw")
        _, subject_map = create_random_modality_dataset(
            data_root=str(tmp_path), nnunet_raw=raw, dataset_id=2001,
        )
        assert len(subject_map) == 2
        for subject, cases in subject_map.items():
            assert len(cases) == 4  # 4 modalities per subject
            for case_id in cases:
                assert case_id.startswith("case_")


class TestGenerateGroupedSplits:
    """Tests for grouped fold splits that prevent subject leakage."""

    def test_all_cases_accounted_for(self):
        subject_to_cases = {
            "sub_0": ["case_0000", "case_0001", "case_0002", "case_0003"],
            "sub_1": ["case_0004", "case_0005", "case_0006", "case_0007"],
            "sub_2": ["case_0008", "case_0009", "case_0010", "case_0011"],
            "sub_3": ["case_0012", "case_0013", "case_0014", "case_0015"],
        }
        splits = generate_grouped_splits(subject_to_cases, n_folds=2)
        assert len(splits) == 2
        all_cases = {"case_{:04d}".format(i) for i in range(16)}
        for fold in splits:
            assert len(fold["train"]) + len(fold["val"]) == 16
        # Each fold's val should be disjoint from its train.
        for fold in splits:
            assert set(fold["train"]).isdisjoint(set(fold["val"]))

    def test_subject_cases_stay_in_same_fold(self):
        """No subject should have cases split across train and val."""
        subject_to_cases = {
            f"sub_{i}": [f"case_{i * 4 + j:04d}" for j in range(4)]
            for i in range(10)
        }
        splits = generate_grouped_splits(subject_to_cases, n_folds=5)
        for fold in splits:
            train_set = set(fold["train"])
            val_set = set(fold["val"])
            for subject, cases in subject_to_cases.items():
                cases_in_train = sum(1 for c in cases if c in train_set)
                cases_in_val = sum(1 for c in cases if c in val_set)
                # Each subject's cases must be entirely in train OR entirely in val.
                assert cases_in_train == 0 or cases_in_val == 0, (
                    f"Subject {subject} leaked across train/val: "
                    f"{cases_in_train} in train, {cases_in_val} in val"
                )

    def test_reproducible_with_same_seed(self):
        subject_to_cases = {
            f"sub_{i}": [f"case_{i:04d}"] for i in range(20)
        }
        s1 = generate_grouped_splits(subject_to_cases, n_folds=5, seed=42)
        s2 = generate_grouped_splits(subject_to_cases, n_folds=5, seed=42)
        for f1, f2 in zip(s1, s2):
            assert f1["train"] == f2["train"]
            assert f1["val"] == f2["val"]

    def test_different_seeds_produce_different_splits(self):
        subject_to_cases = {
            f"sub_{i}": [f"case_{i:04d}"] for i in range(20)
        }
        s1 = generate_grouped_splits(subject_to_cases, n_folds=5, seed=1)
        s2 = generate_grouped_splits(subject_to_cases, n_folds=5, seed=2)
        # At least one fold should differ.
        any_different = any(
            f1["val"] != f2["val"] for f1, f2 in zip(s1, s2)
        )
        assert any_different
