#!/usr/bin/env python3
"""Run data validation stage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.validation import run_validation
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data validation")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--input", help="Input manifest (override config)")
    parser.add_argument("--output", help="Output manifest (override config)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    input_path = args.input or cfg.paths.manifests.raw
    output_path = args.output or cfg.paths.manifests.validated
    
    validation_cfg = cfg.pipeline.validation
    
    logger.info(f"Running validation on {input_path}")
    
    run_validation(
        manifest_path=input_path,
        output_path=output_path,
        min_duration_sec=validation_cfg.min_duration_sec,
        max_duration_sec=validation_cfg.max_duration_sec,
        target_sample_rate=validation_cfg.target_sample_rate,
        require_transcript=validation_cfg.require_transcript,
    )
    
    logger.info("Validation complete")


if __name__ == "__main__":
    main()
