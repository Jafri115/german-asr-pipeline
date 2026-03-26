#!/usr/bin/env python3
"""Run preprocessing stage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.preprocessing import run_preprocessing
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run preprocessing")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--input", help="Input manifest (override config)")
    parser.add_argument("--output-dir", help="Output directory (override config)")
    parser.add_argument("--output-manifest", help="Output manifest (override config)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    input_path = args.input or cfg.paths.manifests.validated
    output_dir = args.output_dir or cfg.paths.preprocessed_audio_dir
    output_manifest = args.output_manifest or cfg.paths.manifests.preprocessed
    
    prep_cfg = cfg.pipeline.preprocessing

    srt_dir = getattr(cfg.data, "srt_dir", None)
    ground_truth_dir = getattr(cfg.data, "ground_truth_dir", None)

    logger.info(f"Running preprocessing on {input_path}")
    
    run_preprocessing(
        manifest_path=input_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        target_sample_rate=prep_cfg.target_sample_rate,
        normalize_transcripts=prep_cfg.normalize_transcripts,
        trim_silence=False,
        srt_dir=srt_dir,
        srt_output_dir=ground_truth_dir,
        preprocess_audio=True,
        preserve_original_audio=False,
        normalize_volume=False,
    )
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
