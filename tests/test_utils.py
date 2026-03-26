"""Tests for utility functions."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from asr_pipeline.utils import (
    compute_dataset_stats,
    ensure_dir,
    load_config,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def test_ensure_dir():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "nested" / "dir"
        result = ensure_dir(test_path)
        assert result.exists()
        assert result.is_dir()


def test_save_and_load_manifest_parquet():
    """Test saving and loading parquet manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "audio_path": ["a1.wav", "a2.wav", "a3.wav"],
            "duration_sec": [1.0, 2.0, 3.0],
        })
        
        path = Path(tmpdir) / "test.parquet"
        save_manifest(df, path)
        
        loaded = load_manifest(path)
        assert len(loaded) == 3
        assert list(loaded.columns) == list(df.columns)


def test_save_and_load_manifest_csv():
    """Test saving and loading CSV manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({
            "sample_id": ["s1", "s2"],
            "audio_path": ["a1.wav", "a2.wav"],
        })
        
        path = Path(tmpdir) / "test.csv"
        save_manifest(df, path)
        
        loaded = load_manifest(path)
        assert len(loaded) == 2


def test_validate_manifest():
    """Test manifest validation."""
    df = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "audio_path": ["a1.wav", "a2.wav"],
    })
    
    # Should pass with default required columns
    assert validate_manifest(df) is True
    
    # Should pass with explicit columns
    assert validate_manifest(df, ["sample_id", "audio_path"]) is True
    
    # Should raise with missing columns
    with pytest.raises(ValueError):
        validate_manifest(df, ["sample_id", "nonexistent"])


def test_compute_dataset_stats():
    """Test dataset statistics computation."""
    df = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3"],
        "duration_sec": [1.0, 2.0, 3.0],
        "is_valid": [True, True, False],
        "split": ["train", "val", "test"],
    })
    
    stats = compute_dataset_stats(df)
    
    assert stats["total_samples"] == 3
    assert stats["valid_samples"] == 2
    assert stats["total_duration_hours"] == 6.0 / 3600
    assert stats["mean_duration_sec"] == 2.0
    assert stats["split_counts"]["train"] == 1


def test_load_config():
    """Test config loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_content = """
test_key: test_value
nested:
  key: value
"""
        config_path = Path(tmpdir) / "test.yaml"
        with open(config_path, "w") as f:
            f.write(config_content)
        
        cfg = load_config(config_path)
        assert cfg.test_key == "test_value"
        assert cfg.nested.key == "value"
