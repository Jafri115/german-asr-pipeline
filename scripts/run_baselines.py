#!/usr/bin/env python3
"""Run baseline benchmarking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.benchmark import run_baseline_benchmark
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run baseline benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline benchmark")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--test-manifest", help="Test manifest (override config)")
    parser.add_argument("--output", help="Output path (override config)")
    parser.add_argument("--max-samples", type=int, help="Max samples for quick test")
    parser.add_argument("--device", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    test_manifest = args.test_manifest or str(Path(cfg.paths.splits_dir) / "test.parquet")
    output_path = args.output or cfg.paths.artifacts.benchmark_results
    
    models = cfg.benchmark.models
    device = args.device or cfg.benchmark.get("device", None)
    
    logger.info(f"Running baseline benchmark on {test_manifest}")
    logger.info(f"Models: {models}")
    
    run_baseline_benchmark(
        manifest_path=test_manifest,
        output_path=output_path,
        models=models,
        max_samples=args.max_samples,
        device=device,
    )
    
    logger.info("Baseline benchmarking complete")


if __name__ == "__main__":
    main()
