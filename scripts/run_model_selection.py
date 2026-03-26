#!/usr/bin/env python3
"""Run model selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.selection import run_model_selection
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run model selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model selection")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--benchmark-results", help="Benchmark results (override config)")
    parser.add_argument("--output", help="Output path (override config)")
    parser.add_argument("--report", help="Report path")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    benchmark_results = args.benchmark_results or cfg.paths.artifacts.benchmark_results
    output_path = args.output or cfg.paths.artifacts.model_selection
    report_path = args.report or cfg.paths.reports.model_selection
    
    selection_cfg = cfg.pipeline.model_selection
    
    logger.info(f"Running model selection from {benchmark_results}")
    
    run_model_selection(
        benchmark_results_path=benchmark_results,
        output_path=output_path,
        report_path=report_path,
        wer_weight=selection_cfg.wer_weight,
        speed_weight=selection_cfg.speed_weight,
    )
    
    logger.info("Model selection complete")


if __name__ == "__main__":
    main()
