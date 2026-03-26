#!/usr/bin/env python3
"""Run fine-tuning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.finetune import run_finetune
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fine-tuning")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--model", help="Model name (override config)")
    parser.add_argument("--train-manifest", help="Train manifest (override config)")
    parser.add_argument("--val-manifest", help="Val manifest (override config)")
    parser.add_argument("--output-dir", help="Output directory (override config)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Get model from selection if available
    model = args.model
    if not model:
        import json
        selection_path = cfg.paths.artifacts.model_selection
        if Path(selection_path).exists():
            with open(selection_path, "r") as f:
                selection = json.load(f)
                model = selection.get("primary", {}).get("model", cfg.finetune.model_name)
        else:
            model = cfg.finetune.model_name
    
    train_manifest = args.train_manifest or str(Path(cfg.paths.splits_dir) / "train.parquet")
    val_manifest = args.val_manifest or str(Path(cfg.paths.splits_dir) / "val.parquet")
    output_dir = args.output_dir or cfg.paths.artifacts.finetuned_model
    
    ft_cfg = cfg.finetune
    
    logger.info(f"Running fine-tuning with model: {model}")
    logger.info(f"Train: {train_manifest}")
    logger.info(f"Val: {val_manifest}")
    
    run_finetune(
        model_name=model,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        num_epochs=ft_cfg.num_epochs,
        batch_size=ft_cfg.batch_size,
        learning_rate=ft_cfg.learning_rate,
    )
    
    logger.info("Fine-tuning complete")


if __name__ == "__main__":
    main()
