#!/usr/bin/env python3
"""Run evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.evaluation import run_evaluation
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--model", help="Model path (override config)")
    parser.add_argument("--test-manifest", help="Test manifest (override config)")
    parser.add_argument("--output", help="Output path (override config)")
    parser.add_argument("--baseline-results", help="Baseline results for comparison")
    parser.add_argument("--device", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    model = args.model or cfg.paths.artifacts.finetuned_model
    test_manifest = args.test_manifest or str(Path(cfg.paths.splits_dir) / "test.parquet")
    output_path = args.output or cfg.paths.artifacts.evaluation_results
    
    baseline_results = args.baseline_results or cfg.paths.artifacts.get("baseline_results", None)
    device = args.device or cfg.evaluation.get("device", None)
    
    logger.info(f"Running evaluation on {test_manifest}")
    
    run_evaluation(
        model_path=model,
        test_manifest=test_manifest,
        output_path=output_path,
        baseline_results_path=baseline_results,
        device=device,
    )
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
