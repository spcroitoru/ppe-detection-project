"""
Unit tests for src/data_utils.py (preprocessing functions).
Uses temporary directories with synthetic data, so tests are fast, isolated,
and don't depend on the real dataset.
"""
import os
import shutil
import tempfile

import pytest

import src.data_utils as data_utils


@pytest.fixture
def temp_dataset():
    """Creates a minimal temporary dataset structure for testing."""
    tmp_dir = tempfile.mkdtemp()
    original_data_root = data_utils.DATA_ROOT
    data_utils.DATA_ROOT = tmp_dir

    for split in ["train", "val"]:
        os.makedirs(os.path.join(tmp_dir, split, "images"))
        os.makedirs(os.path.join(tmp_dir, split, "labels"))

    yield tmp_dir

    data_utils.DATA_ROOT = original_data_root
    shutil.rmtree(tmp_dir)


def _img_dir(tmp_dir, split="train"):
    return os.path.join(tmp_dir, split, "images")


def _lbl_dir(tmp_dir, split="train"):
    return os.path.join(tmp_dir, split, "labels")


def test_list_images_finds_all_supported_extensions(temp_dataset):
    tmp_dir = temp_dataset
    for name in ["a.jpg", "b.jpeg", "c.png", "d.webp"]:
        open(os.path.join(_img_dir(tmp_dir), name), "w").close()

    assert len(data_utils.list_images("train")) == 4


def test_check_data_quality_detects_clean_data(temp_dataset):
    tmp_dir = temp_dataset
    open(os.path.join(_img_dir(tmp_dir), "img1.jpg"), "w").close()
    with open(os.path.join(_lbl_dir(tmp_dir), "img1.txt"), "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    report = data_utils.check_data_quality("train")
    assert report["images_without_label"] == 0
    assert report["labels_without_image"] == 0
    assert report["invalid_bounding_boxes"] == 0


def test_check_data_quality_detects_image_without_label(temp_dataset):
    tmp_dir = temp_dataset
    open(os.path.join(_img_dir(tmp_dir), "orphan.jpg"), "w").close()

    report = data_utils.check_data_quality("train")
    assert report["images_without_label"] == 1


def test_check_data_quality_detects_invalid_bounding_box(temp_dataset):
    tmp_dir = temp_dataset
    open(os.path.join(_img_dir(tmp_dir), "img1.jpg"), "w").close()
    with open(os.path.join(_lbl_dir(tmp_dir), "img1.txt"), "w") as f:
        f.write("0 1.5 0.5 0.2 0.2\n")  # xc=1.5 is outside [0,1]

    report = data_utils.check_data_quality("train")
    assert report["invalid_bounding_boxes"] == 1


def test_check_data_quality_detects_empty_label(temp_dataset):
    tmp_dir = temp_dataset
    open(os.path.join(_img_dir(tmp_dir), "img1.jpg"), "w").close()
    open(os.path.join(_lbl_dir(tmp_dir), "img1.txt"), "w").close()  # empty

    report = data_utils.check_data_quality("train")
    assert report["empty_labels"] == 1


def test_clean_labels_removes_orphan_labels(temp_dataset):
    tmp_dir = temp_dataset
    open(os.path.join(_lbl_dir(tmp_dir), "orphan.txt"), "w").close()

    report = data_utils.clean_labels("train")
    assert report["orphan_labels_removed"] == 1
    assert not os.path.exists(os.path.join(_lbl_dir(tmp_dir), "orphan.txt"))


def test_convert_jfif_to_jpg_renames_files(temp_dataset):
    tmp_dir = temp_dataset
    open(os.path.join(_img_dir(tmp_dir), "photo.jfif"), "w").close()

    renamed = data_utils.convert_jfif_to_jpg("train")
    assert renamed == 1
    assert os.path.exists(os.path.join(_img_dir(tmp_dir), "photo.jpg"))
    assert not os.path.exists(os.path.join(_img_dir(tmp_dir), "photo.jfif"))


def test_resplit_train_val_creates_list_files(temp_dataset):
    tmp_dir = temp_dataset
    for split in ["train", "val"]:
        for i in range(5):
            open(os.path.join(_img_dir(tmp_dir, split), f"img{split}{i}.jpg"), "w").close()

    result = data_utils.resplit_train_val(train_ratio=0.8, seed=0)

    assert result["total_images"] == 10
    assert result["new_train_count"] == 8
    assert result["new_val_count"] == 2


def test_resplit_train_val_is_reproducible_with_same_seed(temp_dataset):
    tmp_dir = temp_dataset
    for split in ["train", "val"]:
        for i in range(10):
            open(os.path.join(_img_dir(tmp_dir, split), f"img{split}{i}.jpg"), "w").close()

    result1 = data_utils.resplit_train_val(train_ratio=0.9, seed=42)
    with open(result1["train_list_path"]) as f:
        first_run = f.read()

    result2 = data_utils.resplit_train_val(train_ratio=0.9, seed=42)
    with open(result2["train_list_path"]) as f:
        second_run = f.read()

    assert first_run == second_run