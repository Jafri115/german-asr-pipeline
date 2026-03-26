"""Data ingestion module for scanning and indexing audio/transcript files."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from asr_pipeline.utils import get_audio_info, get_logger, save_manifest
from asr_pipeline.preprocessing import SrtTranscriptParser

logger = get_logger(__name__)

_srt_parser = SrtTranscriptParser()


class DataIngester:
    """Ingest raw audio and transcript files to create a raw manifest."""
    
    def __init__(
        self,
        audio_extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
        transcript_extensions: Tuple[str, ...] = (".txt", ".transcript", ".json", ".srt"),
        audio_dir: Optional[Union[str, Path]] = None,
        transcript_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the data ingester.
        
        Args:
            audio_extensions: Tuple of valid audio file extensions
            transcript_extensions: Tuple of valid transcript file extensions
            audio_dir: Default audio directory
            transcript_dir: Default transcript directory
        """
        self.audio_extensions = audio_extensions
        self.transcript_extensions = transcript_extensions
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
    
    def scan_audio_files(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> List[Path]:
        """Scan directory for audio files.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            
        Returns:
            List of audio file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        audio_files = []
        pattern = "**/*" if recursive else "*"
        
        for ext in self.audio_extensions:
            audio_files.extend(directory.glob(f"{pattern}{ext}"))
        
        logger.info(f"Found {len(audio_files)} audio files in {directory}")
        return sorted(audio_files)
    
    def find_transcript(
        self,
        audio_path: Path,
        transcript_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Find transcript file for an audio file.
        
        Args:
            audio_path: Path to audio file
            transcript_dir: Optional directory to look for transcripts
            
        Returns:
            Path to transcript file or None
        """
        # Try same directory as audio
        base_name = audio_path.stem
        
        for ext in self.transcript_extensions:
            transcript_path = audio_path.parent / f"{base_name}{ext}"
            if transcript_path.exists():
                return transcript_path
        
        # Try transcript directory if provided
        if transcript_dir:
            for ext in self.transcript_extensions:
                transcript_path = transcript_dir / f"{base_name}{ext}"
                if transcript_path.exists():
                    return transcript_path
        
        # Try common patterns
        patterns = [
            audio_path.parent / "transcripts" / f"{base_name}.txt",
            audio_path.parent / "text" / f"{base_name}.txt",
            audio_path.parent / f"{base_name}.trans.txt",
        ]
        
        for pattern in patterns:
            if pattern.exists():
                return pattern
        
        return None
    
    def read_transcript(self, transcript_path: Path) -> Optional[str]:
        """Read transcript from file.
        
        Args:
            transcript_path: Path to transcript file
            
        Returns:
            Transcript text or None
        """
        if not transcript_path.exists():
            return None
        
        try:
            if transcript_path.suffix == ".json":
                import json
                with open(transcript_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("text", data.get("transcript", None))
            elif transcript_path.suffix == ".srt":
                return _srt_parser.parse(transcript_path)
            else:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read transcript {transcript_path}: {e}")
            return None
    
    def extract_speaker_id(self, audio_path: Path) -> Optional[str]:
        """Extract speaker ID from file path or name.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speaker ID or None
        """
        # Common patterns: speaker_001, spk_123, s001, etc.
        patterns = [
            r"speaker[_-]?([\w\d]+)",
            r"spk[_-]?([\w\d]+)",
            r"s([\d]{2,})",
            r"([\d]{3,})_",
        ]
        
        name = audio_path.stem
        for pattern in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Try parent directory name
        parent = audio_path.parent.name
        if "speaker" in parent.lower() or "spk" in parent.lower():
            return parent
        
        return None
    
    def extract_domain(self, audio_path: Path) -> Optional[str]:
        """Extract domain from file path.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Domain or None
        """
        # Common domain indicators in path
        path_str = str(audio_path).lower()
        domains = [
            "medical", "legal", "news", "podcast", "audiobook",
            "conversation", "meeting", "lecture", "broadcast",
            "callcenter", "customer_service", "interview",
        ]
        
        for domain in domains:
            if domain in path_str:
                return domain
        
        return "general"
    
    def create_manifest(
        self,
        audio_files: List[Path],
        transcript_dir: Optional[Union[str, Path]] = None,
        extract_metadata: bool = True,
    ) -> pd.DataFrame:
        """Create manifest from audio files.
        
        Args:
            audio_files: List of audio file paths
            transcript_dir: Optional directory for transcripts
            extract_metadata: Whether to extract audio metadata
            
        Returns:
            Manifest DataFrame
        """
        transcript_dir = Path(transcript_dir) if transcript_dir else None
        
        records = []
        for idx, audio_path in enumerate(tqdm(audio_files, desc="Processing audio files")):
            record = {
                "sample_id": f"sample_{idx:08d}",
                "audio_path": str(audio_path),
                "audio_path_absolute": str(audio_path.resolve()),
            }
            
            # Find and read transcript
            transcript_path = self.find_transcript(audio_path, transcript_dir)
            if transcript_path:
                record["transcript_path"] = str(transcript_path)
                record["transcript_raw"] = self.read_transcript(transcript_path)
            else:
                record["transcript_path"] = None
                record["transcript_raw"] = None
            
            # Extract metadata
            if extract_metadata:
                try:
                    info = get_audio_info(audio_path)
                    record.update(info)
                except Exception as e:
                    logger.warning(f"Failed to get audio info for {audio_path}: {e}")
                    record["duration_sec"] = None
                    record["sample_rate"] = None
                    record["channels"] = None
            
            # Extract speaker and domain
            record["speaker_id"] = self.extract_speaker_id(audio_path)
            record["domain"] = self.extract_domain(audio_path)
            record["language"] = "de"  # German
            
            # Validation flag (will be set properly in validation stage)
            record["is_valid"] = True
            
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Created manifest with {len(df)} records")
        
        return df
    
    def ingest_from_directory(
        self,
        audio_dir: Union[str, Path],
        output_path: Union[str, Path],
        transcript_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ) -> pd.DataFrame:
        """Ingest all audio files from a directory.
        
        Args:
            audio_dir: Directory containing audio files
            output_path: Path to save manifest
            transcript_dir: Optional directory containing transcripts
            recursive: Whether to scan recursively
            
        Returns:
            Manifest DataFrame
        """
        logger.info(f"Ingesting from {audio_dir}")
        
        audio_files = self.scan_audio_files(audio_dir, recursive)
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return pd.DataFrame()
        
        df = self.create_manifest(audio_files, transcript_dir)
        save_manifest(df, output_path)
        
        return df
    
    def ingest_from_csv_mapping(
        self,
        mapping_csv: Union[str, Path],
        output_path: Union[str, Path],
        audio_col: str = "audio_path",
        transcript_col: str = "transcript",
    ) -> pd.DataFrame:
        """Ingest from a CSV file with audio-transcript mappings.
        
        Args:
            mapping_csv: CSV file with mappings
            output_path: Path to save manifest
            audio_col: Column name for audio paths
            transcript_col: Column name for transcripts
            
        Returns:
            Manifest DataFrame
        """
        logger.info(f"Ingesting from mapping CSV: {mapping_csv}")
        
        mapping_df = pd.read_csv(mapping_csv)
        
        records = []
        for idx, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Processing"):
            audio_path = Path(row[audio_col])
            
            record = {
                "sample_id": f"sample_{idx:08d}",
                "audio_path": str(audio_path),
                "audio_path_absolute": str(audio_path.resolve()) if audio_path.exists() else None,
                "transcript_raw": row[transcript_col],
                "language": "de",
                "is_valid": audio_path.exists(),
            }
            
            # Extract audio info if available
            if audio_path.exists():
                try:
                    info = get_audio_info(audio_path)
                    record.update(info)
                except Exception as e:
                    logger.warning(f"Failed to get audio info: {e}")
            
            records.append(record)
        
        df = pd.DataFrame(records)
        save_manifest(df, output_path)
        
        return df


def run_ingestion(
    audio_dir: Union[str, Path],
    output_path: Union[str, Path],
    transcript_dir: Optional[Union[str, Path]] = None,
    recursive: bool = True,
) -> pd.DataFrame:
    """Run ingestion from CLI.
    
    Args:
        audio_dir: Directory containing audio files
        output_path: Path to save manifest
        transcript_dir: Optional directory containing transcripts
        recursive: Whether to scan recursively
        
    Returns:
        Manifest DataFrame
    """
    ingester = DataIngester()
    return ingester.ingest_from_directory(
        audio_dir=audio_dir,
        output_path=output_path,
        transcript_dir=transcript_dir,
        recursive=recursive,
    )
