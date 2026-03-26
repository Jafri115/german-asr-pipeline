#!/usr/bin/env python3
"""Run train/val/test split creation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.split import run_split
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run split creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--input", help="Input manifest (override config)")
    parser.add_argument("--output-dir", help="Output directory (override config)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    input_path = args.input or cfg.paths.manifests.preprocessed
    output_dir = args.output_dir or cfg.paths.splits_dir
    
    split_cfg = cfg.pipeline.split
    
    logger.info(f"Creating splits from {input_path}")
    
    run_split(
        manifest_path=input_path,
        output_dir=output_dir,
        train_ratio=split_cfg.train_ratio,
        val_ratio=split_cfg.val_ratio,
        test_ratio=split_cfg.test_ratio,
        split_strategy=split_cfg.strategy,
        stratify_by=split_cfg.get("stratify_by", None),
        group_by=split_cfg.get("group_by", None),
        random_seed=split_cfg.seed,
    )
    
    logger.info("Split creation complete")


if __name__ == "__main__":
    main()
