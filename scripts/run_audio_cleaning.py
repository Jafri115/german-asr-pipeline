#!/usr/bin/env python3
"""Run audio cleaning stage (DeepFilterNet denoise + hum removal + loudnorm)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.preprocessing import AudioCleaner
from asr_pipeline.utils import get_logger, load_config


logger = get_logger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean audio with DeepFilterNet pipeline")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--input", help="Single input file or directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--mux-video", action="store_true", help="Mux cleaned audio back into video")
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate work files")
    args = parser.parse_args()

    cfg = load_config(args.config)

    clean_cfg = cfg.pipeline.audio_cleaning
    output_dir = args.output_dir or cfg.data.cleaned_audio_dir

    cleaner = AudioCleaner(
        deep_filter_bin=clean_cfg.deep_filter_bin,
        extract_sample_rate=clean_cfg.extract_sample_rate,
        gain_db=clean_cfg.gain_db,
        highpass_freq=clean_cfg.highpass_freq,
        dehum_freqs=[(f, g) for f, g in clean_cfg.dehum_freqs],
        target_lufs=clean_cfg.target_lufs,
        true_peak=clean_cfg.true_peak,
        lra=clean_cfg.lra,
        limiter=clean_cfg.limiter,
        deep_filter_pf=clean_cfg.deep_filter_pf,
    )

    mux = args.mux_video or clean_cfg.mux_video

    input_path = Path(args.input) if args.input else Path(cfg.data.raw_audio_dir)

    if input_path.is_file():
        ext = ".mp4" if mux else ".wav"
        out = Path(output_dir) / f"{input_path.stem}_clean{ext}"
        cleaner.clean(input_path, out, mux_video=mux, keep_work=args.keep_work)
    elif input_path.is_dir():
        cleaner.clean_directory(
            input_path, output_dir, mux_video=mux, keep_work=args.keep_work,
        )
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    logger.info("Audio cleaning complete. Output in %s", output_dir)


if __name__ == "__main__":
    main()
