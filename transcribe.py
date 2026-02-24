#!/usr/bin/env python3
"""
Speaker Diarization + Transcription System
Phase 1 + 2: Transcription with speaker identification and profile matching

Usage:
    python transcribe.py meeting.wav
    python transcribe.py meeting.wav --model small --max-speakers 5
    python transcribe.py meeting.wav --speaker-names "SPEAKER_00=田中,SPEAKER_01=鈴木"
"""

import os
import sys
import json
import time
import warnings

# Suppress noisy-but-harmless warnings from pyannote / huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
import argparse
from math import gcd
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from dotenv import load_dotenv

from profiles import SpeakerProfiles, parse_speaker_embeddings

WHISPER_SR = 16000          # Whisper expects 16 kHz
MIN_SEGMENT_DURATION = 0.5  # seconds — drop shorter segments
GAP_MERGE_THRESHOLD = 0.5   # seconds — merge same-speaker gaps shorter than this


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono float32. Returns (samples, sample_rate)."""
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    return audio, sr


def extract_segment_16k(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Extract a time slice and resample to 16 kHz for Whisper."""
    s = int(start * sr)
    e = min(int(end * sr), len(audio))
    chunk = audio[s:e]
    if len(chunk) == 0 or sr == WHISPER_SR:
        return chunk
    g = gcd(sr, WHISPER_SR)
    return resample_poly(chunk, WHISPER_SR // g, sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def run_diarization(
    audio: np.ndarray,
    sr: int,
    hf_token: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> tuple[list[tuple[float, float, str]], dict[str, np.ndarray]]:
    """
    Run pyannote/speaker-diarization-3.1.
    Accepts preloaded audio (numpy array) to avoid torchcodec/FFmpeg dependency.
    Returns (segments, speaker_embeddings) where:
      segments          : [(start_sec, end_sec, speaker_id), ...]
      speaker_embeddings: {speaker_id: flat float32 ndarray}  (empty if unavailable)
    """
    import torch
    from pyannote.audio import Pipeline

    print("[2/4] Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(torch.device("cpu"))

    params: dict = {}
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    # Pass preloaded waveform to avoid torchcodec/FFmpeg requirement
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    audio_input = {"waveform": waveform, "sample_rate": sr}

    print("      Running diarization (slowest step on CPU)...")
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            output = pipeline(audio_input, hook=hook, **params)
    except ImportError:
        output = pipeline(audio_input, **params)

    # pyannote >= 3.3 returns DiarizeOutput; extract the Annotation object
    annotation = None
    for candidate in [
        output,
        getattr(output, "speaker_diarization", None),
        getattr(output, "exclusive_speaker_diarization", None),
        getattr(output, "annotation", None),
    ]:
        if candidate is not None and hasattr(candidate, "itertracks"):
            annotation = candidate
            break
    else:
        attrs = [a for a in dir(output) if not a.startswith("_")]
        raise RuntimeError(
            f"Cannot extract Annotation from {type(output).__name__}. "
            f"Available attributes: {attrs}"
        )

    segments = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    n_speakers = len({s[2] for s in segments})
    print(f"      → {n_speakers} speaker(s), {len(segments)} raw segments detected")

    # Extract speaker embeddings for Phase 2 profile matching / enrollment
    raw_emb = getattr(output, "speaker_embeddings", None)
    speaker_embeddings = parse_speaker_embeddings(raw_emb, labels=annotation.labels())

    return segments, speaker_embeddings


# ---------------------------------------------------------------------------
# Segment merging
# ---------------------------------------------------------------------------

def merge_segments(
    segments: list[tuple[float, float, str]],
    gap_threshold: float = GAP_MERGE_THRESHOLD,
    min_duration: float = MIN_SEGMENT_DURATION,
) -> list[tuple[float, float, str]]:
    """Merge adjacent same-speaker segments; drop very short ones."""
    if not segments:
        return []

    merged: list[tuple[float, float, str]] = []
    cur_start, cur_end, cur_spk = segments[0]

    for start, end, spk in segments[1:]:
        if spk == cur_spk and start - cur_end <= gap_threshold:
            cur_end = end  # extend current segment
        else:
            if cur_end - cur_start >= min_duration:
                merged.append((cur_start, cur_end, cur_spk))
            cur_start, cur_end, cur_spk = start, end, spk

    if cur_end - cur_start >= min_duration:
        merged.append((cur_start, cur_end, cur_spk))

    return merged


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_segments(
    model,
    audio: np.ndarray,
    sr: int,
    segments: list[tuple[float, float, str]],
    language: str = "ja",
) -> list[str]:
    """
    Transcribe each diarization segment with faster-whisper.
    Returns list of transcription strings (same order as segments).
    """
    texts: list[str] = []
    n = len(segments)

    for i, (start, end, speaker) in enumerate(segments, 1):
        chunk = extract_segment_16k(audio, sr, start, end)

        # Skip if too short to contain speech
        if len(chunk) < WHISPER_SR * 0.3:
            texts.append("")
            continue

        # VAD filter helps on long segments but can drop short valid speech
        use_vad = (end - start) > 3.0

        seg_gen, _ = model.transcribe(
            chunk,
            language=language,
            beam_size=5,
            vad_filter=use_vad,
        )
        text = "".join(s.text for s in seg_gen).strip()
        texts.append(text)

        if i % 10 == 0 or i == n:
            preview = (text[:55] + "…") if len(text) > 55 else text
            print(f"      [{i:3d}/{n}] {speaker}: {preview!r}")

    return texts


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def build_json_output(
    segments: list[tuple[float, float, str]],
    texts: list[str],
    duration: float,
    speaker_names: dict[str, str],
) -> dict:
    """Build the JSON output matching the specified format."""
    speaker_map: dict[str, list[dict]] = defaultdict(list)

    for (start, end, spk_id), text in zip(segments, texts):
        if text:
            speaker_map[spk_id].append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            })

    speakers = [
        {
            "speaker_id": spk_id,
            "speaker_name": speaker_names.get(spk_id, spk_id),
            "segments": segs,
        }
        for spk_id, segs in sorted(speaker_map.items())
    ]

    return {"duration": round(duration, 3), "speakers": speakers}


def print_readable_transcript(result: dict) -> None:
    """Print a chronological, human-readable transcript."""
    all_segs: list[tuple[float, float, str, str]] = []
    for spk in result["speakers"]:
        name = spk["speaker_name"]
        for seg in spk["segments"]:
            all_segs.append((seg["start"], seg["end"], name, seg["text"]))
    all_segs.sort(key=lambda x: x[0])

    print("\n" + "=" * 64)
    print("TRANSCRIPT")
    print("=" * 64)
    for start, end, name, text in all_segs:
        ts = f"[{_fmt_time(start)} - {_fmt_time(end)}]"
        print(f"{ts} {name}: {text}")
    print("=" * 64)


def _fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Speaker diarization + transcription (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py meeting.wav
  python transcribe.py meeting.wav --model small --max-speakers 5
  python transcribe.py meeting.wav --speaker-names "SPEAKER_00=田中,SPEAKER_01=鈴木"
  python transcribe.py meeting.wav --output transcript.json --no-print
""",
    )
    parser.add_argument("audio", help="Input audio file (WAV, FLAC, OGG)")
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (or set HF_TOKEN in .env)",
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language code, e.g. ja, en (default: ja)",
    )
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Minimum number of speakers (optional hint)")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maximum number of speakers (optional hint)")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: <audio_stem>_transcript.json)",
    )
    parser.add_argument(
        "--speaker-names",
        default="",
        help="Name mappings: SPEAKER_00=田中,SPEAKER_01=鈴木",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress readable transcript output to stdout",
    )
    parser.add_argument(
        "--profile-dir",
        default="profiles",
        help="Speaker profile directory for auto-identification (default: profiles). "
             "Skipped if directory is empty or does not exist.",
    )
    parser.add_argument(
        "--enroll",
        default="",
        help="Enroll speakers after transcription: SPEAKER_00=田中,SPEAKER_01=鈴木. "
             "Updates profiles with embeddings from this session.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for profile matching (default: 0.65).",
    )
    args = parser.parse_args()

    # --- Validate ---
    if not args.hf_token:
        print(
            "ERROR: Hugging Face token required.\n"
            "  Option A: Add HF_TOKEN=<token> to .env\n"
            "  Option B: Pass --hf-token <token>"
        )
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    suffix = audio_path.suffix.lower()
    if suffix == ".mp3":
        print(
            "ERROR: MP3 is not directly supported.\n"
            "  Convert first:  ffmpeg -i input.mp3 output.wav"
        )
        sys.exit(1)

    # Parse speaker name mappings
    speaker_names: dict[str, str] = {}
    if args.speaker_names:
        for pair in args.speaker_names.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                speaker_names[k.strip()] = v.strip()

    output_path = (
        Path(args.output) if args.output
        else audio_path.with_name(f"{audio_path.stem}_transcript.json")
    )

    total_start = time.time()

    # --- 1. Load audio ---
    print(f"\n[1/4] Loading audio: {audio_path.name}")
    t0 = time.time()
    audio, sr = load_audio(str(audio_path))
    duration = len(audio) / sr
    load_time = time.time() - t0
    print(f"      Duration: {_fmt_time(duration)} ({duration / 60:.1f} min)  "
          f"SR: {sr} Hz  [{load_time:.1f}s]")

    # --- 2. Diarization ---
    t0 = time.time()
    raw_segments, speaker_embeddings = run_diarization(
        audio, sr, args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    diarize_time = time.time() - t0
    print(f"      Diarization time: {diarize_time:.1f}s  "
          f"(RTF: {diarize_time / duration:.2f}x)")

    segments = merge_segments(raw_segments)
    print(f"      After merging: {len(raw_segments)} → {len(segments)} segments")

    if not segments:
        print("ERROR: No speech segments found. Check your audio file.")
        sys.exit(1)

    # --- Profile-based auto-identification (Phase 2) ---
    profile_dir = Path(args.profile_dir)
    if profile_dir.exists() and speaker_embeddings:
        profiles = SpeakerProfiles(str(profile_dir))
        profiles.load()
        if len(profiles) > 0:
            auto_names = profiles.match_speakers(speaker_embeddings, threshold=args.threshold)
            identified = {k: v for k, v in auto_names.items() if v != k}
            if identified:
                print(f"      [Profile] Auto-identified: {identified}")
            # --speaker-names takes precedence; profiles fill in the rest
            for spk_id, name in auto_names.items():
                if spk_id not in speaker_names:
                    speaker_names[spk_id] = name

    # --- 3. Load Whisper model ---
    print(f"\n[3/4] Loading Whisper model ({args.model})...")
    t0 = time.time()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print(f"      Model loaded [{time.time() - t0:.1f}s]")

    # --- 4. Transcribe ---
    print(f"[4/4] Transcribing {len(segments)} segments  (language: {args.language})...")
    t0 = time.time()
    texts = transcribe_segments(model, audio, sr, segments, language=args.language)
    transcribe_time = time.time() - t0
    print(f"      Transcription time: {transcribe_time:.1f}s  "
          f"(RTF: {transcribe_time / duration:.2f}x)")

    # --- Build and save output ---
    result = build_json_output(segments, texts, duration, speaker_names)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # --- Enroll speakers from this session (Phase 2) ---
    if args.enroll and speaker_embeddings:
        enroll_map: dict[str, str] = {}
        for pair in args.enroll.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                enroll_map[k.strip()] = v.strip()
        if enroll_map:
            profiles = SpeakerProfiles(str(profile_dir))
            profiles.load()
            enrolled = 0
            for spk_id, name in enroll_map.items():
                if spk_id in speaker_embeddings:
                    profiles.enroll(name, speaker_embeddings[spk_id])
                    n = profiles._meta[name]["n_samples"]
                    print(f"      [Enroll] {spk_id} → {name}  (total samples: {n})")
                    enrolled += 1
                else:
                    print(f"      [Enroll] WARNING: {spk_id} not found in embeddings")
            if enrolled > 0:
                profiles.save()

    # --- Summary ---
    total_time = time.time() - total_start
    rtf = total_time / duration
    faster_or_slower = "faster" if rtf < 1 else "slower"

    print(f"\n{'=' * 64}")
    print("DONE")
    print(f"{'=' * 64}")
    print(f"  Audio duration  : {_fmt_time(duration)} ({duration / 60:.1f} min)")
    print(f"  Total time      : {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Real-time factor: {rtf:.2f}x  ({faster_or_slower} than real-time)")
    print(f"  Breakdown       : load {load_time:.0f}s  |  "
          f"diarize {diarize_time:.0f}s  |  transcribe {transcribe_time:.0f}s")
    print(f"  Output          : {output_path}")
    print()
    print("  Speakers:")
    for spk in result["speakers"]:
        n_segs = len(spk["segments"])
        total_speech = sum(s["end"] - s["start"] for s in spk["segments"])
        name = spk["speaker_name"]
        print(f"    {name:20s}: {n_segs:3d} segment(s), "
              f"{total_speech:5.0f}s total speech")

    if not args.no_print:
        print_readable_transcript(result)


if __name__ == "__main__":
    main()
