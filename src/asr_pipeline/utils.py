"""Utility functions for the ASR pipeline."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with Rich formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True,
        )
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(message)s",
            datefmt="[%X]",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load a YAML config file using OmegaConf.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Loaded config as DictConfig
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    return cfg


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save a config to YAML file.
    
    Args:
        config: Config to save
        output_path: Output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def load_manifest(manifest_path: Union[str, Path]) -> pd.DataFrame:
    """Load a manifest file (CSV or Parquet).
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        Manifest as DataFrame
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    suffix = manifest_path.suffix.lower()
    
    if suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
    elif suffix == ".csv":
        df = pd.read_csv(manifest_path)
    elif suffix == ".json":
        df = pd.read_json(manifest_path, lines=True)
    else:
        raise ValueError(f"Unsupported manifest format: {suffix}")
    
    return df


def save_manifest(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """Save a manifest to file.
    
    Args:
        df: DataFrame to save
        output_path: Output path
        format: Output format (parquet, csv, json). Auto-detected from path if None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format is None:
        format = output_path.suffix.lower().lstrip(".")
    
    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    console.print(f"[green]Saved manifest to {output_path}[/green]")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if needed.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """Get audio duration in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    import soundfile as sf
    
    info = sf.info(audio_path)
    return info.duration


def get_audio_info(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """Get audio file information.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dict with audio info
    """
    import soundfile as sf
    
    info = sf.info(audio_path)
    return {
        "duration_sec": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "format": info.format,
    }


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "01:23:45")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def merge_manifests(
    manifest_paths: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Merge multiple manifest files into one.
    
    Args:
        manifest_paths: List of manifest paths
        output_path: Optional output path
        
    Returns:
        Merged DataFrame
    """
    dfs = [load_manifest(p) for p in manifest_paths]
    merged = pd.concat(dfs, ignore_index=True)
    
    if output_path:
        save_manifest(merged, output_path)
    
    return merged


def validate_manifest(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> bool:
    """Validate manifest has required columns.
    
    Args:
        df: Manifest DataFrame
        required_cols: List of required column names
        
    Returns:
        True if valid
    """
    if required_cols is None:
        required_cols = ["sample_id", "audio_path"]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return True


def compute_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics for a dataset manifest.
    
    Args:
        df: Manifest DataFrame
        
    Returns:
        Dict with statistics
    """
    stats = {
        "total_samples": len(df),
        "valid_samples": df["is_valid"].sum() if "is_valid" in df.columns else len(df),
    }
    
    if "duration_sec" in df.columns:
        stats["total_duration_hours"] = df["duration_sec"].sum() / 3600
        stats["mean_duration_sec"] = df["duration_sec"].mean()
        stats["median_duration_sec"] = df["duration_sec"].median()
        stats["min_duration_sec"] = df["duration_sec"].min()
        stats["max_duration_sec"] = df["duration_sec"].max()
    
    if "split" in df.columns:
        stats["split_counts"] = df["split"].value_counts().to_dict()
    
    if "domain" in df.columns:
        stats["domain_counts"] = df["domain"].value_counts().to_dict()
    
    return stats


def print_dataset_stats(df: pd.DataFrame, title: str = "Dataset Statistics") -> None:
    """Print dataset statistics in a nice format.
    
    Args:
        df: Manifest DataFrame
        title: Title for the output
    """
    from rich.table import Table
    
    stats = compute_dataset_stats(df)
    
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 50)
    
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Samples", f"{stats['total_samples']:,}")
    table.add_row("Valid Samples", f"{stats['valid_samples']:,}")
    
    if "total_duration_hours" in stats:
        table.add_row("Total Duration", f"{stats['total_duration_hours']:.2f} hours")
        table.add_row("Mean Duration", f"{stats['mean_duration_sec']:.2f} sec")
        table.add_row("Median Duration", f"{stats['median_duration_sec']:.2f} sec")
        table.add_row("Duration Range", f"{stats['min_duration_sec']:.1f} - {stats['max_duration_sec']:.1f} sec")
    
    if "split_counts" in stats:
        table.add_row("Split Distribution", str(stats["split_counts"]))
    
    console.print(table)
    console.print()
