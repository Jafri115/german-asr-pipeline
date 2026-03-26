#!/usr/bin/env python3
"""Run data ingestion stage."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.ingestion import run_ingestion
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data ingestion")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--audio-dir", help="Audio directory (override config)")
    parser.add_argument("--output", help="Output manifest (override config)")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Get paths
    audio_dir = args.audio_dir or cfg.data.raw_audio_dir
    output_path = args.output or cfg.paths.manifests.raw
    transcript_dir = cfg.data.get("transcript_dir", None)
    
    logger.info(f"Running ingestion from {audio_dir}")
    
    run_ingestion(
        audio_dir=audio_dir,
        output_path=output_path,
        transcript_dir=transcript_dir,
        recursive=True,
    )
    
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
