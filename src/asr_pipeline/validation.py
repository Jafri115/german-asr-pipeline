"""Data validation module for checking audio and transcript quality."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from asr_pipeline.utils import get_audio_info, get_logger, load_manifest, save_manifest

logger = get_logger(__name__)


class DataValidator:
    """Validate audio files and transcripts."""
    
    def __init__(
        self,
        min_duration_sec: float = 1.0,
        max_duration_sec: float = 60.0,
        target_sample_rate: int = 16000,
        require_transcript: bool = True,
        min_transcript_length: int = 2,
    ):
        """Initialize the validator.
        
        Args:
            min_duration_sec: Minimum allowed duration
            max_duration_sec: Maximum allowed duration
            target_sample_rate: Target sample rate
            require_transcript: Whether transcripts are required
            min_transcript_length: Minimum transcript length in characters
        """
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.target_sample_rate = target_sample_rate
        self.require_transcript = require_transcript
        self.min_transcript_length = min_transcript_length
    
    def validate_audio_file(self, audio_path: Path) -> Tuple[bool, Dict[str, str]]:
        """Validate a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, issues_dict)
        """
        issues = {}
        
        # Check file exists
        if not audio_path.exists():
            return False, {"file_exists": "File not found"}
        
        # Check file not empty
        if audio_path.stat().st_size == 0:
            return False, {"file_size": "File is empty"}
        
        try:
            info = get_audio_info(audio_path)
        except Exception as e:
            return False, {"audio_readable": f"Cannot read audio: {e}"}
        
        # Check duration
        duration = info["duration_sec"]
        if duration < self.min_duration_sec:
            issues["duration"] = f"Too short: {duration:.2f}s < {self.min_duration_sec}s"
        if duration > self.max_duration_sec:
            issues["duration"] = f"Too long: {duration:.2f}s > {self.max_duration_sec}s"
        
        # Check sample rate
        if info["sample_rate"] != self.target_sample_rate:
            issues["sample_rate"] = (
                f"Non-target SR: {info['sample_rate']} != {self.target_sample_rate}"
            )
        
        return len(issues) == 0, issues
    
    def validate_transcript(self, transcript: Optional[str]) -> Tuple[bool, Dict[str, str]]:
        """Validate a transcript.
        
        Args:
            transcript: Transcript text
            
        Returns:
            Tuple of (is_valid, issues_dict)
        """
        issues = {}
        
        if transcript is None:
            if self.require_transcript:
                return False, {"transcript": "Missing transcript"}
            return True, {}
        
        # Check length
        if len(transcript.strip()) < self.min_transcript_length:
            issues["transcript_length"] = (
                f"Too short: {len(transcript)} chars < {self.min_transcript_length}"
            )
        
        # Check for suspicious patterns
        if transcript.count("[") > 3 or transcript.count("(") > 3:
            issues["transcript_format"] = "Too many annotations/brackets"
        
        # Check if transcript is just silence markers
        silence_markers = ["[silence]", "[noise]", "[music]", "<unk>"]
        cleaned = transcript.lower()
        for marker in silence_markers:
            cleaned = cleaned.replace(marker, "")
        
        if len(cleaned.strip()) < self.min_transcript_length:
            issues["transcript_content"] = "Transcript appears to be only silence/noise markers"
        
        return len(issues) == 0, issues
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, List[str]]:
        """Validate a single manifest row.
        
        Args:
            row: DataFrame row
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate audio
        audio_path = Path(row["audio_path"])
        audio_valid, audio_issues = self.validate_audio_file(audio_path)
        issues.extend([f"audio:{k}:{v}" for k, v in audio_issues.items()])
        
        # Validate transcript
        transcript = row.get("transcript_raw")
        transcript_valid, transcript_issues = self.validate_transcript(transcript)
        issues.extend([f"transcript:{k}:{v}" for k, v in transcript_issues.items()])
        
        is_valid = audio_valid and transcript_valid
        
        return is_valid, issues
    
    def validate_manifest(
        self,
        manifest_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Validate entire manifest.
        
        Args:
            manifest_path: Path to manifest
            output_path: Optional path to save validated manifest
            
        Returns:
            Validated DataFrame with validation columns
        """
        df = load_manifest(manifest_path)
        logger.info(f"Validating {len(df)} records")
        
        validation_results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
            is_valid, issues = self.validate_row(row)
            validation_results.append({
                "is_valid": is_valid,
                "validation_issues": "; ".join(issues) if issues else None,
            })
        
        validation_df = pd.DataFrame(validation_results)
        df["is_valid"] = validation_df["is_valid"]
        df["validation_issues"] = validation_df["validation_issues"]
        
        # Statistics
        valid_count = df["is_valid"].sum()
        invalid_count = len(df) - valid_count
        
        logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
        
        if invalid_count > 0:
            issues_breakdown = df[~df["is_valid"]]["validation_issues"].value_counts().head(10)
            logger.info(f"Top issues:\n{issues_breakdown}")
        
        if output_path:
            save_manifest(df, output_path)
        
        return df
    
    def get_valid_manifest(
        self,
        manifest_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Get only valid records from manifest.
        
        Args:
            manifest_path: Path to manifest
            output_path: Optional path to save filtered manifest
            
        Returns:
            Filtered DataFrame with only valid records
        """
        df = self.validate_manifest(manifest_path)
        valid_df = df[df["is_valid"]].copy()
        
        logger.info(f"Filtered to {len(valid_df)} valid records from {len(df)} total")
        
        if output_path:
            save_manifest(valid_df, output_path)
        
        return valid_df


def run_validation(
    manifest_path: Union[str, Path],
    output_path: Union[str, Path],
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 60.0,
    target_sample_rate: int = 16000,
    require_transcript: bool = True,
) -> pd.DataFrame:
    """Run validation from CLI.
    
    Args:
        manifest_path: Path to input manifest
        output_path: Path to save validated manifest
        min_duration_sec: Minimum allowed duration
        max_duration_sec: Maximum allowed duration
        target_sample_rate: Target sample rate
        require_transcript: Whether transcripts are required
        
    Returns:
        Validated DataFrame
    """
    validator = DataValidator(
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        target_sample_rate=target_sample_rate,
        require_transcript=require_transcript,
    )
    return validator.validate_manifest(manifest_path, output_path)
