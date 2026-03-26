#!/usr/bin/env python3
"""Run full inference on all data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_pipeline.inference import run_full_inference
from asr_pipeline.utils import get_logger, load_config

logger = get_logger(__name__)


def main():
    """Run full inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full inference")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file")
    parser.add_argument("--model", help="Model path (override config)")
    parser.add_argument("--manifest", help="Input manifest (override config)")
    parser.add_argument("--output", help="Output path (override config)")
    parser.add_argument("--fallback-model", help="Fallback model path")
    parser.add_argument("--device", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    model = args.model or cfg.paths.artifacts.finetuned_model
    manifest = args.manifest or cfg.paths.manifests.preprocessed
    output_path = args.output or cfg.paths.artifacts.inference_results
    
    fallback_model = args.fallback_model
    if not fallback_model:
        import json
        selection_path = cfg.paths.artifacts.model_selection
        if Path(selection_path).exists():
            with open(selection_path, "r") as f:
                selection = json.load(f)
                fallback_model = selection.get("fallback", {}).get("model", None)
    
    device = args.device or cfg.inference.get("device", None)
    
    logger.info(f"Running full inference on {manifest}")
    
    run_full_inference(
        model_path=model,
        manifest_path=manifest,
        output_path=output_path,
        fallback_model_path=fallback_model,
        device=device,
    )
    
    logger.info("Full inference complete")


if __name__ == "__main__":
    main()
