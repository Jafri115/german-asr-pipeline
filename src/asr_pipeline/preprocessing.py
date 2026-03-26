"""Audio preprocessing and transcript normalization module."""

import re
import shlex
import shutil
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from asr_pipeline.utils import ensure_dir, get_logger, load_manifest, save_manifest

logger = get_logger(__name__)


def _clean_amberscript_text(text: str) -> str:
    """Shared cleaning logic for Amberscript transcript text.

    Strips speaker labels, inline timestamps, CAVE headers,
    parenthetical annotations, and normalises whitespace.
    """
    text = re.sub(r"CAVE\s*:.*", "", text)
    text = re.sub(r"^\s*\d{6}[\w\._\-]*\s*", "", text)
    text = re.sub(r"#\d{2}:\d{2}:\d{2}[#\-\d]*#?", "", text)
    text = re.sub(r"-\s*#?\d{2}:\d{2}:\d{2}[#\-\d]*,\s*Video beginnt\)", "", text)
    text = re.sub(r",?\s*Video beginnt\)", "", text)
    text = re.sub(r"\b[A-Z][A-Z0-9]*\s*\d*\s*:\s*", "", text)
    text = re.sub(r"[\(\[][\w\s\.\-\xE4\xF6\xFC\xC4\xD6\xDC\xDF]*[\)\]]", "", text)
    text = re.sub(r"(?:^|\s)-\s+", " ", text)
    text = re.sub(r"\s*/\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_session_key(stem: str) -> Optional[str]:
    """Extract a canonical ``{patient_id}_S{session_id}`` key from a messy filename stem."""
    s = stem
    s = re.sub(r"[\._ ]mp[34][\._ ]?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[\._ ]mov[\._ ]?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_\.]?(converted|merged|remuxed|unkonvertiert)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\(\d+\)", "", s)
    s = s.strip("_.- ")
    m = re.match(r"(\d{6})[_]?[sStT](\d+)", s)
    if m:
        return f"{m.group(1)}_S{m.group(2)}"
    return None


class SrtTranscriptParser:
    """Parse SRT caption files into clean plain text for ground truth transcripts."""

    def parse(self, srt_path: Union[str, Path]) -> str:
        """Convert an SRT file to clean plain text."""
        srt_content = Path(srt_path).read_text(encoding="utf-8-sig")

        lines: List[str] = []
        for line in srt_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.fullmatch(r"\d+", line):
                continue
            if re.match(
                r"\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}",
                line,
            ):
                continue
            line = re.sub(r"<[^>]+>", "", line)
            if line:
                lines.append(line)

        text = " ".join(lines)
        return _clean_amberscript_text(text)

    def convert_directory(
        self,
        srt_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = True,
    ) -> List[Tuple[Path, Path]]:
        """Batch-convert all .srt files in a directory to .txt files."""
        srt_dir = Path(srt_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pattern = "**/*.srt" if recursive else "*.srt"
        srt_files = sorted(srt_dir.glob(pattern))
        logger.info(f"Found {len(srt_files)} SRT files in {srt_dir}")

        converted: List[Tuple[Path, Path]] = []
        for srt_path in tqdm(srt_files, desc="Converting SRT transcripts"):
            try:
                clean_text = self.parse(srt_path)
                out_name = srt_path.stem.replace("_transcript", "") + ".txt"
                out_path = output_dir / out_name
                out_path.write_text(clean_text, encoding="utf-8")
                converted.append((srt_path, out_path))
            except Exception as e:
                logger.warning(f"Failed to convert {srt_path}: {e}")

        logger.info(f"Converted {len(converted)}/{len(srt_files)} SRT files to {output_dir}")
        return converted


class DocxTranscriptParser:
    """Parse DOCX Amberscript transcripts into clean plain text."""

    def parse(self, docx_path: Union[str, Path]) -> str:
        from docx import Document
        import zipfile

        docx_path = Path(docx_path)
        if not zipfile.is_zipfile(str(docx_path)):
            raw = docx_path.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", "", raw)
            return _clean_amberscript_text(text)

        doc = Document(str(docx_path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return _clean_amberscript_text(text)


class RtfTranscriptParser:
    """Parse RTF Amberscript transcripts into clean plain text."""

    def parse(self, rtf_path: Union[str, Path]) -> str:
        from striprtf.striprtf import rtf_to_text

        raw = Path(rtf_path).read_bytes().decode("latin-1")
        text = rtf_to_text(raw)
        return _clean_amberscript_text(text)


class TranscriptConverter:
    """Unified converter for SRT, DOCX, and RTF Amberscript transcripts."""

    def __init__(self) -> None:
        self._srt = SrtTranscriptParser()
        self._docx = DocxTranscriptParser()
        self._rtf = RtfTranscriptParser()

    _PARSERS = {".srt": "_srt", ".docx": "_docx", ".rtf": "_rtf"}

    def parse(self, path: Union[str, Path]) -> str:
        path = Path(path)
        attr = self._PARSERS.get(path.suffix.lower())
        if attr is None:
            raise ValueError(f"Unsupported transcript format: {path.suffix}")
        return getattr(self, attr).parse(path)

    def convert_directory(
        self,
        src_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Tuple[str, ...] = (".srt", ".docx", ".rtf"),
        recursive: bool = True,
        deduplicate: bool = True,
    ) -> List[Tuple[Path, Path]]:
        """Batch-convert transcript files to clean .txt, deduplicating by session key.

        When *deduplicate* is True and multiple source files map to the same
        session key, SRT is preferred over RTF over DOCX.
        """
        src_dir = Path(src_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files: List[Path] = []
        for ext in extensions:
            pat = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(src_dir.glob(pat))
        logger.info(f"Found {len(files)} transcript files in {src_dir}")

        priority = {".srt": 0, ".rtf": 1, ".docx": 2}
        key_map: Dict[Optional[str], List[Path]] = {}
        for f in files:
            key = _normalize_session_key(f.stem)
            key_map.setdefault(key, []).append(f)

        if deduplicate:
            best: Dict[str, Path] = {}
            for key, sources in key_map.items():
                if key is None:
                    continue
                sources.sort(key=lambda p: priority.get(p.suffix.lower(), 99))
                best[key] = sources[0]
            to_convert = list(best.items())
        else:
            to_convert = []
            for key, sources in key_map.items():
                for s in sources:
                    k = key or s.stem
                    to_convert.append((k, s))

        logger.info(f"Converting {len(to_convert)} unique transcripts")

        converted: List[Tuple[Path, Path]] = []
        for key, src_path in tqdm(sorted(to_convert), desc="Converting transcripts"):
            try:
                clean_text = self.parse(src_path)
                out_path = output_dir / f"{key}.txt"
                out_path.write_text(clean_text, encoding="utf-8")
                converted.append((src_path, out_path))
            except Exception as e:
                logger.warning(f"Failed to convert {src_path}: {e}")

        logger.info(f"Converted {len(converted)}/{len(to_convert)} transcripts to {output_dir}")
        return converted


class TranscriptNormalizer:
    """Normalize German transcripts."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        normalize_numbers: bool = True,
        remove_filler_words: bool = False,
        remove_annotations: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_numbers = normalize_numbers
        self.remove_filler_words = remove_filler_words
        self.remove_annotations = remove_annotations

        self.filler_words = [
            "ähm", "öhm", "ähhh", "öhhh", "äh", "öh",
            "also", "naja", "halt", "eben", "irgendwie",
            "quasi", "sozusagen", "irgendwie",
        ]

    def remove_bracket_annotations(self, text: str) -> str:
        """Remove bracket annotations like [noise], (laughs)."""
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"<.*?>", "", text)
        return text

    def normalize_numbers_german(self, text: str) -> str:
        """Normalize simple digit representations to German words."""
        digit_map = {
            "0": "null",
            "1": "eins",
            "2": "zwei",
            "3": "drei",
            "4": "vier",
            "5": "fünf",
            "6": "sechs",
            "7": "sieben",
            "8": "acht",
            "9": "neun",
        }
        for digit, word in digit_map.items():
            text = re.sub(rf"\b{digit}\b", word, text)
        return text

    def remove_filler(self, text: str) -> str:
        """Remove filler words."""
        words = text.split()
        filtered = [w for w in words if w.lower() not in self.filler_words]
        return " ".join(filtered)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """Apply all normalization steps."""
        if not text:
            return ""

        if self.remove_annotations:
            text = self.remove_bracket_annotations(text)

        if self.normalize_numbers:
            text = self.normalize_numbers_german(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            allowed = "äöüßÄÖÜ "
            text = "".join(c for c in text if c not in string.punctuation or c in allowed)

        if self.remove_filler_words:
            text = self.remove_filler(text)

        text = self.normalize_whitespace(text)
        return text


class AudioPreprocessor:
    """Preprocess audio files without denoising/cleaning.

    This class keeps audio close to the original signal. It can resample,
    convert to mono, optionally trim edge silence, optionally peak-normalize,
    and optionally truncate long files.

    DeepFilterNet / FFmpeg denoise is intentionally not used here because it
    worsened WER in your setup.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        trim_silence: bool = False,
        trim_top_db: int = 20,
        normalize_volume: bool = False,
        max_duration_sec: Optional[float] = None,
        preserve_original_audio: bool = False,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db
        self.normalize_volume = normalize_volume
        self.max_duration_sec = max_duration_sec
        self.preserve_original_audio = preserve_original_audio

    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            waveform, sr = librosa.load(
                audio_path,
                sr=self.target_sample_rate,
                mono=(self.target_channels == 1),
            )
            return waveform, sr
        except Exception as e:
            logger.error(f"Failed to load {audio_path}: {e}")
            raise

    def trim_silence_from_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Trim silence from start/end."""
        if not self.trim_silence:
            return waveform

        trimmed, _ = librosa.effects.trim(
            waveform,
            top_db=self.trim_top_db,
        )
        return trimmed

    def normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Peak-normalize audio volume."""
        if not self.normalize_volume:
            return waveform

        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak * 0.95

        return waveform

    def truncate_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Truncate audio if too long."""
        if self.max_duration_sec is None:
            return waveform

        max_samples = int(self.max_duration_sec * sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        return waveform

    def process_audio(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, int, Optional[Path]]:
        """Process a single audio file.

        If preserve_original_audio=True, the file is not rewritten and the
        original path is kept in the manifest.
        """
        waveform, sr = self.load_audio(audio_path)

        waveform = self.trim_silence_from_audio(waveform, sr)
        waveform = self.normalize_audio(waveform)
        waveform = self.truncate_audio(waveform, sr)

        saved_path = None
        if output_path and not self.preserve_original_audio:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, waveform, sr)
            saved_path = output_path

        return waveform, sr, saved_path

    def process_manifest(
        self,
        manifest_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_manifest_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Process all audio files in a manifest.

        No denoise/audio cleaning is performed.
        """
        df = load_manifest(manifest_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_records = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing audio"):
            if not row.get("is_valid", True):
                processed_records.append(row.to_dict())
                continue

            audio_path = Path(row["audio_path"])
            output_path = output_dir / f"{row['sample_id']}.wav"

            try:
                waveform, sr, saved_path = self.process_audio(audio_path, output_path)

                record = row.to_dict()
                record["audio_path_original"] = str(audio_path)
                record["audio_path"] = str(saved_path) if saved_path is not None else str(audio_path)
                record["duration_sec"] = len(waveform) / sr
                record["sample_rate"] = sr
                record["channels"] = 1 if self.target_channels == 1 else self.target_channels
                record["is_preprocessed"] = True
                record["audio_cleaned"] = False
                record["audio_cleaning_applied"] = "none"
                record["audio_preprocessing_mode"] = (
                    "preserve_original" if self.preserve_original_audio else "resample_only"
                )

            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                record = row.to_dict()
                record["is_valid"] = False
                record["validation_issues"] = f"preprocessing_failed: {e}"

            processed_records.append(record)

        processed_df = pd.DataFrame(processed_records)
        save_manifest(processed_df, output_manifest_path)
        return processed_df


class AudioChunker:
    """Chunk long audio files into smaller segments."""

    def __init__(
        self,
        chunk_duration_sec: float = 30.0,
        overlap_sec: float = 1.0,
        min_chunk_duration_sec: float = 5.0,
    ):
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        self.min_chunk_duration_sec = min_chunk_duration_sec

    def chunk_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Chunk audio into segments."""
        chunk_samples = int(self.chunk_duration_sec * sample_rate)
        overlap_samples = int(self.overlap_sec * sample_rate)
        min_samples = int(self.min_chunk_duration_sec * sample_rate)

        chunks = []
        start = 0

        while start < len(waveform):
            end = min(start + chunk_samples, len(waveform))
            chunk = waveform[start:end]

            if len(chunk) >= min_samples:
                start_sec = start / sample_rate
                end_sec = end / sample_rate
                chunks.append((chunk, start_sec, end_sec))

            start += chunk_samples - overlap_samples

            if end >= len(waveform):
                break

        return chunks

    def chunk_manifest(
        self,
        manifest_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_manifest_path: Union[str, Path],
        max_duration_threshold: float = 30.0,
    ) -> pd.DataFrame:
        """Chunk long audio files in manifest."""
        df = load_manifest(manifest_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunked_records = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking audio"):
            duration = row.get("duration_sec", 0)

            if duration <= max_duration_threshold:
                chunked_records.append(row.to_dict())
                continue

            audio_path = Path(row["audio_path"])

            try:
                waveform, sr = librosa.load(audio_path, sr=row.get("sample_rate", 16000))
                chunks = self.chunk_audio(waveform, sr)

                for chunk_idx, (chunk, start_sec, end_sec) in enumerate(chunks):
                    chunk_path = output_dir / f"{row['sample_id']}_chunk{chunk_idx:03d}.wav"
                    sf.write(chunk_path, chunk, sr)

                    record = row.to_dict()
                    record["sample_id"] = f"{row['sample_id']}_chunk{chunk_idx:03d}"
                    record["parent_audio_id"] = row["sample_id"]
                    record["chunk_id"] = chunk_idx
                    record["audio_path"] = str(chunk_path)
                    record["duration_sec"] = len(chunk) / sr
                    record["chunk_start_sec"] = start_sec
                    record["chunk_end_sec"] = end_sec

                    chunked_records.append(record)

            except Exception as e:
                logger.warning(f"Failed to chunk {audio_path}: {e}")
                record = row.to_dict()
                record["is_valid"] = False
                chunked_records.append(record)

        chunked_df = pd.DataFrame(chunked_records)
        save_manifest(chunked_df, output_manifest_path)

        logger.info(f"Created {len(chunked_df)} chunks from {len(df)} originals")
        return chunked_df


# ---------------------------------------------------------------------------
# Audio Cleaning (disabled for ASR / kept only for optional future use)
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and raise on failure."""
    logger.debug(" ".join(shlex.quote(str(c)) for c in cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0 and check:
        logger.error("STDOUT: %s", p.stdout)
        logger.error("STDERR: %s", p.stderr)
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd[0]}")
    return p


def _probe_duration(path: Path) -> float:
    """Return duration in seconds of an audio/video file via ffprobe."""
    p = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
        capture_output=True,
    )
    try:
        return float(p.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


class AudioCleaner:
    """Optional DeepFilterNet cleaning pipeline.

    Disabled by default for ASR usage because denoising/cleaning worsened WER.
    To guard against accidental use, allow_enabled must be set to True.
    """

    def __init__(
        self,
        deep_filter_bin: Union[str, Path] = "deep-filter",
        extract_sample_rate: int = 48_000,
        gain_db: float = 8.0,
        highpass_freq: int = 120,
        dehum_freqs: Optional[List[Tuple[int, int]]] = None,
        target_lufs: float = -16.0,
        true_peak: float = -1.5,
        lra: float = 9.0,
        limiter: float = 0.98,
        deep_filter_pf: bool = False,
        allow_enabled: bool = False,
    ) -> None:
        self.deep_filter_bin = str(deep_filter_bin)
        self.extract_sr = extract_sample_rate
        self.gain_db = gain_db
        self.highpass_freq = highpass_freq
        self.dehum_freqs = dehum_freqs or [(50, -8), (100, -4)]
        self.target_lufs = target_lufs
        self.true_peak = true_peak
        self.lra = lra
        self.limiter = limiter
        self.deep_filter_pf = deep_filter_pf
        self.allow_enabled = allow_enabled

    def _ensure_enabled(self) -> None:
        if not self.allow_enabled:
            raise RuntimeError(
                "AudioCleaner is disabled because audio cleaning worsened WER. "
                "Set allow_enabled=True only if you explicitly want to use it."
            )

    def extract_audio(self, input_path: Path, wav_out: Path) -> Path:
        self._ensure_enabled()
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        wav_out.unlink(missing_ok=True)
        _run_cmd([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-map", "0:a:0?", "-vn",
            "-ac", "1", "-ar", str(self.extract_sr),
            "-c:a", "pcm_s16le",
            str(wav_out),
        ])
        if not wav_out.exists() or wav_out.stat().st_size == 0:
            raise RuntimeError(f"Audio extraction failed for {input_path}")
        return wav_out

    def _deepfilter(self, wav_in: Path, work_dir: Path, tag: str) -> Path:
        self._ensure_enabled()
        df_out = work_dir / f"df_{tag}"
        if df_out.exists():
            shutil.rmtree(df_out)
        df_out.mkdir(parents=True, exist_ok=True)

        cmd = [self.deep_filter_bin, "--output-dir", str(df_out)]
        if self.deep_filter_pf:
            cmd.append("--pf")
        cmd.append(str(wav_in))
        _run_cmd(cmd)

        outs = list(df_out.glob("*.wav"))
        if not outs:
            raise RuntimeError(f"deep-filter produced no output in {df_out}")
        result = work_dir / f"audio_{tag}.wav"
        shutil.copy2(str(outs[0]), str(result))
        return result

    def apply_gain(self, wav_in: Path, wav_out: Path) -> Path:
        self._ensure_enabled()
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        _run_cmd([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(wav_in),
            "-af", f"highpass=f={self.highpass_freq},volume={self.gain_db}dB",
            "-ar", str(self.extract_sr), "-ac", "1",
            "-c:a", "pcm_s16le",
            str(wav_out),
        ])
        return wav_out

    def finalise(self, wav_in: Path, wav_out: Path) -> Path:
        self._ensure_enabled()
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        dehum_entries = ";".join(
            f"entry({freq},{gain})" for freq, gain in self.dehum_freqs
        )
        af = ",".join([
            f"highpass=f={self.highpass_freq}",
            f"firequalizer=gain_entry='{dehum_entries}'",
            f"loudnorm=I={self.target_lufs}:TP={self.true_peak}:LRA={self.lra}:linear=true",
            f"alimiter=limit={self.limiter}",
        ])
        _run_cmd([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(wav_in),
            "-af", af,
            "-ar", str(self.extract_sr), "-ac", "1",
            "-c:a", "pcm_s16le",
            str(wav_out),
        ])
        return wav_out

    def mux_back(self, original_video: Path, clean_wav: Path, output_video: Path) -> Path:
        self._ensure_enabled()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        _run_cmd([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(original_video),
            "-i", str(clean_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_video),
        ])
        return output_video

    def clean(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        work_dir: Optional[Union[str, Path]] = None,
        mux_video: bool = False,
        keep_work: bool = False,
    ) -> Path:
        self._ensure_enabled()

        input_path = Path(input_path)
        output_path = Path(output_path)

        use_temp = work_dir is None
        work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="asr_clean_"))
        work.mkdir(parents=True, exist_ok=True)

        try:
            src_dur = _probe_duration(input_path)
            logger.info("[clean] Source: %.1fs (%.1f min) — %s", src_dur, src_dur / 60, input_path.name)

            wav_raw = self.extract_audio(input_path, work / "audio_raw.wav")
            dur = _probe_duration(wav_raw)
            logger.info("[clean] Step 1 extract_audio: %.1fs (%.1f min) | delta %.1fs", dur, dur / 60, dur - src_dur)

            wav_df1 = self._deepfilter(wav_raw, work, "pass1")
            prev, dur = dur, _probe_duration(wav_df1)
            logger.info("[clean] Step 2 deepfilter pass1: %.1fs (%.1f min) | delta %.1fs", dur, dur / 60, dur - prev)

            wav_gain = self.apply_gain(wav_df1, work / "audio_gain.wav")
            prev, dur = dur, _probe_duration(wav_gain)
            logger.info("[clean] Step 3 apply_gain: %.1fs (%.1f min) | delta %.1fs", dur, dur / 60, dur - prev)

            wav_df2 = self._deepfilter(wav_gain, work, "pass2")
            prev, dur = dur, _probe_duration(wav_df2)
            logger.info("[clean] Step 4 deepfilter pass2: %.1fs (%.1f min) | delta %.1fs", dur, dur / 60, dur - prev)

            wav_clean = self.finalise(wav_df2, work / "audio_clean.wav")
            prev, dur = dur, _probe_duration(wav_clean)
            logger.info("[clean] Step 5 finalise: %.1fs (%.1f min) | delta %.1fs", dur, dur / 60, dur - prev)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if mux_video:
                self.mux_back(input_path, wav_clean, output_path)
            else:
                shutil.copy2(str(wav_clean), str(output_path))

            logger.info("Cleaned %s -> %s", input_path.name, output_path)
            return output_path

        finally:
            if use_temp and not keep_work:
                shutil.rmtree(work, ignore_errors=True)

    def clean_batch(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        mux_video: bool = False,
        keep_work: bool = False,
    ) -> List[Path]:
        self._ensure_enabled()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".mp4" if mux_video else ".wav"

        results: List[Path] = []
        for p in tqdm(input_paths, desc="Cleaning audio"):
            p = Path(p)
            out = output_dir / f"{p.stem}_clean{ext}"
            try:
                self.clean(p, out, mux_video=mux_video, keep_work=keep_work)
                results.append(out)
            except Exception as e:
                logger.warning("Failed to clean %s: %s", p, e)
        logger.info("Cleaned %d / %d files", len(results), len(input_paths))
        return results

    def clean_directory(
        self,
        src_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".flac"),
        recursive: bool = True,
        mux_video: bool = False,
        keep_work: bool = False,
    ) -> List[Path]:
        self._ensure_enabled()

        src_dir = Path(src_dir)
        files: List[Path] = []
        for ext in extensions:
            pat = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(src_dir.glob(pat))
        files.sort()
        logger.info("Found %d files in %s", len(files), src_dir)
        return self.clean_batch(files, output_dir, mux_video=mux_video, keep_work=keep_work)


def convert_srt_transcripts(
    srt_dir: Union[str, Path],
    output_dir: Union[str, Path],
    recursive: bool = True,
) -> List[Tuple[Path, Path]]:
    """Convenience function: batch-convert SRT transcripts to clean text."""
    parser = SrtTranscriptParser()
    return parser.convert_directory(srt_dir, output_dir, recursive=recursive)


def convert_all_transcripts(
    src_dirs: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    extensions: Tuple[str, ...] = (".srt", ".docx", ".rtf"),
    recursive: bool = True,
) -> List[Tuple[Path, Path]]:
    """Batch-convert SRT, DOCX, and RTF transcripts from one or more directories."""
    converter = TranscriptConverter()
    if isinstance(src_dirs, (str, Path)):
        src_dirs = [src_dirs]

    all_converted: List[Tuple[Path, Path]] = []
    for d in src_dirs:
        all_converted.extend(
            converter.convert_directory(d, output_dir, extensions=extensions, recursive=recursive)
        )
    return all_converted


def run_preprocessing(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_manifest_path: Union[str, Path],
    target_sample_rate: int = 16000,
    normalize_transcripts: bool = True,
    trim_silence: bool = False,
    srt_dir: Optional[Union[str, Path]] = None,
    srt_output_dir: Optional[Union[str, Path]] = None,
    transcript_src_dirs: Optional[List[Union[str, Path]]] = None,
    preprocess_audio: bool = True,
    preserve_original_audio: bool = False,
    normalize_volume: bool = False,
) -> pd.DataFrame:
    """Run preprocessing from CLI.

    Audio cleaning is intentionally disabled. This function only does:
      - transcript conversion
      - optional transcript normalization
      - optional audio resampling/mono conversion
      - optional keep-original mode

    Args:
        manifest_path: Input manifest
        output_dir: Output directory for processed audio
        output_manifest_path: Output manifest path
        target_sample_rate: Target sample rate
        normalize_transcripts: Whether to normalize transcripts
        trim_silence: Whether to trim silence from start/end
        srt_dir: Optional SRT caption directory to convert before processing
        srt_output_dir: Where to write cleaned .txt ground truths
        transcript_src_dirs: Additional directories with DOCX/RTF transcripts
        preprocess_audio: Whether to run light audio preprocessing
        preserve_original_audio: Keep original audio_path instead of rewriting WAV files
        normalize_volume: Whether to peak-normalize audio

    Returns:
        Processed manifest DataFrame
    """
    _gt_out = srt_output_dir or (Path(srt_dir).parent / "ground_truth_text" if srt_dir else None)
    all_src_dirs: List[Union[str, Path]] = []

    if srt_dir is not None:
        all_src_dirs.append(srt_dir)
    if transcript_src_dirs:
        all_src_dirs.extend(transcript_src_dirs)

    if all_src_dirs and _gt_out:
        logger.info(f"Converting transcripts from {all_src_dirs} -> {_gt_out}")
        convert_all_transcripts(all_src_dirs, _gt_out)

    if preprocess_audio:
        logger.info(
            "Running light audio preprocessing only: resample/mono=%s, trim_silence=%s, "
            "normalize_volume=%s, preserve_original_audio=%s. Audio cleaning is disabled.",
            target_sample_rate,
            trim_silence,
            normalize_volume,
            preserve_original_audio,
        )

        preprocessor = AudioPreprocessor(
            target_sample_rate=target_sample_rate,
            trim_silence=trim_silence,
            normalize_volume=normalize_volume,
            preserve_original_audio=preserve_original_audio,
        )

        df = preprocessor.process_manifest(
            manifest_path=manifest_path,
            output_dir=output_dir,
            output_manifest_path=output_manifest_path,
        )
    else:
        logger.info("Skipping audio preprocessing entirely. Original audio paths are preserved.")
        df = load_manifest(manifest_path).copy()
        df["audio_cleaned"] = False
        df["audio_cleaning_applied"] = "none"
        df["audio_preprocessing_mode"] = "skipped"
        save_manifest(df, output_manifest_path)

    if normalize_transcripts and "transcript_raw" in df.columns:
        normalizer = TranscriptNormalizer()
        df["transcript_normalized"] = df["transcript_raw"].apply(
            lambda x: normalizer.normalize(x) if pd.notna(x) else ""
        )
        save_manifest(df, output_manifest_path)

    return df