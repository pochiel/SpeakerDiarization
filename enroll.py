#!/usr/bin/env python3
"""
Speaker Enrollment Script — Phase 2

Runs speaker diarization on an audio file, then maps detected SPEAKER_xx IDs
to real names and saves the embeddings as speaker profiles.

Usage:
    # Enroll from audio
    python enroll.py meeting.wav --map "SPEAKER_00=田中,SPEAKER_01=鈴木"

    # Update profiles with more samples (cumulative)
    python enroll.py new_meeting.wav --map "SPEAKER_00=田中"

    # List enrolled speakers
    python enroll.py --list

    # Delete a speaker
    python enroll.py --delete 田中
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from profiles import SpeakerProfiles, parse_speaker_embeddings


# ---------------------------------------------------------------------------
# Diarization (shared logic, mirrors transcribe.py)
# ---------------------------------------------------------------------------

def _get_annotation(output):
    """Extract Annotation object from pyannote pipeline output."""
    for candidate in [
        output,
        getattr(output, "speaker_diarization", None),
        getattr(output, "exclusive_speaker_diarization", None),
        getattr(output, "annotation", None),
    ]:
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate
    attrs = [a for a in dir(output) if not a.startswith("_")]
    raise RuntimeError(
        f"Cannot extract Annotation from {type(output).__name__}. "
        f"Available attributes: {attrs}"
    )


def run_diarization_with_embeddings(
    audio: np.ndarray,
    sr: int,
    hf_token: str,
    min_speakers=None,
    max_speakers=None,
) -> tuple[list, dict[str, np.ndarray]]:
    """
    Run pyannote diarization and return (segments, speaker_embeddings).
    segments: [(start, end, speaker_id), ...]
    speaker_embeddings: {speaker_id: flat float32 ndarray}
    """
    import torch
    from pyannote.audio import Pipeline

    print("      Loading pyannote pipeline...")
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

    waveform = torch.from_numpy(audio).unsqueeze(0)
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            output = pipeline({"waveform": waveform, "sample_rate": sr}, hook=hook, **params)
    except ImportError:
        output = pipeline({"waveform": waveform, "sample_rate": sr}, **params)

    annotation = _get_annotation(output)
    segments = [
        (turn.start, turn.end, spk)
        for turn, _, spk in annotation.itertracks(yield_label=True)
    ]

    raw_emb = getattr(output, "speaker_embeddings", None)
    speaker_embeddings = parse_speaker_embeddings(raw_emb, labels=annotation.labels())

    return segments, speaker_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_speakers(segments: list) -> None:
    from collections import defaultdict
    stats: dict = defaultdict(lambda: {"n": 0, "duration": 0.0})
    for start, end, spk in segments:
        stats[spk]["n"] += 1
        stats[spk]["duration"] += end - start

    print("\n  Detected speakers:")
    for spk in sorted(stats):
        d = stats[spk]["duration"]
        n = stats[spk]["n"]
        print(f"    {spk}: {n} segments, {d:.0f}s total speech")
    print()


def _parse_map(s: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Speaker enrollment for profile-based identification (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enroll.py meeting.wav --map "SPEAKER_00=田中,SPEAKER_01=鈴木"
  python enroll.py meeting.wav --map "SPEAKER_00=田中" --min-speakers 3
  python enroll.py --list
  python enroll.py --delete 田中
""",
    )
    parser.add_argument("audio", nargs="?", help="Input audio file")
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (or set HF_TOKEN in .env)",
    )
    parser.add_argument(
        "--map",
        default="",
        help="Speaker mappings: SPEAKER_00=田中,SPEAKER_01=鈴木",
    )
    parser.add_argument(
        "--profile-dir",
        default="profiles",
        help="Profile directory (default: profiles)",
    )
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List enrolled speakers and exit",
    )
    parser.add_argument(
        "--delete",
        default="",
        metavar="NAME",
        help="Delete a speaker profile by name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for matching (default: 0.65)",
    )
    args = parser.parse_args()

    profiles = SpeakerProfiles(args.profile_dir)
    profiles.load()

    # --- --list ---
    if args.list:
        speakers = profiles.list_speakers()
        if not speakers:
            print(f"No speakers enrolled in '{args.profile_dir}'.")
        else:
            print(f"Enrolled speakers in '{args.profile_dir}':")
            for name in speakers:
                meta = profiles._meta.get(name, {})
                n = meta.get("n_samples", "?")
                updated = meta.get("updated_at", "?")
                print(f"  {name:20s}: {n} sample(s), last updated {updated}")
        return

    # --- --delete ---
    if args.delete:
        name = args.delete
        if profiles.delete(name):
            profiles.save()
            print(f"Deleted profile: {name}")
        else:
            print(f"Speaker not found: {name}")
        return

    # --- Enroll from audio ---
    if not args.audio:
        parser.error("audio file required (or use --list / --delete)")

    if not args.hf_token:
        print(
            "ERROR: Hugging Face token required.\n"
            "  Set HF_TOKEN in .env or pass --hf-token"
        )
        sys.exit(1)

    if not args.map:
        print(
            "ERROR: --map is required.\n"
            "  Example: --map 'SPEAKER_00=田中,SPEAKER_01=鈴木'\n\n"
            "  Tip: run python transcribe.py <audio> first to see which SPEAKER_xx\n"
            "  corresponds to which person, then re-run this script with --map."
        )
        sys.exit(1)

    speaker_map = _parse_map(args.map)
    if not speaker_map:
        print("ERROR: --map could not be parsed. Use format: SPEAKER_00=田中,SPEAKER_01=鈴木")
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    # 1. Load audio
    print(f"\n[1/3] Loading audio: {audio_path.name}")
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    print(f"      Duration: {duration / 60:.1f} min")

    # 2. Diarize
    print("[2/3] Running diarization...")
    t0 = time.time()
    segments, speaker_embeddings = run_diarization_with_embeddings(
        audio, sr, args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    elapsed = time.time() - t0
    n_speakers = len({s[2] for s in segments})
    print(f"      → {n_speakers} speaker(s), {len(segments)} segments  [{elapsed:.0f}s]")
    _summarize_speakers(segments)

    if not speaker_embeddings:
        print(
            "ERROR: Could not extract speaker embeddings from diarization output.\n"
            "  This may happen with some pyannote versions. Please report the issue."
        )
        sys.exit(1)

    print(f"  Available speaker IDs: {sorted(speaker_embeddings.keys())}")

    # 3. Enroll
    print("[3/3] Enrolling speakers...")
    enrolled = 0
    for spk_id, name in speaker_map.items():
        if spk_id in speaker_embeddings:
            profiles.enroll(name, speaker_embeddings[spk_id])
            n = profiles._meta[name]["n_samples"]
            print(f"  {spk_id} → {name}  (total samples: {n})")
            enrolled += 1
        else:
            known = sorted(speaker_embeddings.keys())
            print(f"  WARNING: {spk_id} not found. Known IDs: {known}")

    if enrolled > 0:
        profiles.save()
        print(f"\nSaved {enrolled} profile(s) to '{args.profile_dir}/'")
        print("Run 'python enroll.py --list' to verify.")
    else:
        print("\nNo profiles were enrolled. Check the speaker ID mapping.")


if __name__ == "__main__":
    main()
