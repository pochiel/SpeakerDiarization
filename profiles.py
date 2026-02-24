"""
Speaker Profile Management — Phase 2

Stores per-speaker embedding vectors and provides cosine-similarity matching.

Storage layout:
    profiles/
        speakers.json     # metadata: {name: {n_samples, updated_at}}
        田中.npy          # L2-normalized mean embedding (float32)
        鈴木.npy
        ...
"""

import json
import numpy as np
from pathlib import Path
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).flatten()
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def parse_speaker_embeddings(raw, labels=None) -> dict[str, np.ndarray]:
    """
    Convert DiarizeOutput.speaker_embeddings to {speaker_id: flat float32 ndarray}.

    pyannote >= 3.3 may return various formats; this function handles them
    defensively so that failures simply return an empty dict rather than crashing.

    Args:
        raw:    The speaker_embeddings attribute from DiarizeOutput.
        labels: Optional list of speaker IDs (e.g. from annotation.labels()).
                Required when raw is a plain ndarray to map rows to speaker IDs.
    """
    if raw is None:
        return {}

    result: dict[str, np.ndarray] = {}

    try:
        # Case 1: plain dict {speaker_id -> array-like}
        if isinstance(raw, dict):
            for k, v in raw.items():
                arr = np.array(v.data if hasattr(v, "data") else v, dtype=np.float32)
                result[str(k)] = arr.flatten()
            return result

        # Case 2: object with .labels and .data (SlidingWindowFeature-like)
        if hasattr(raw, "labels") and hasattr(raw, "data"):
            data = np.array(raw.data, dtype=np.float32)
            emb_labels = raw.labels
            for i, label in enumerate(emb_labels):
                result[str(label)] = data[i].flatten()
            return result

        # Case 3: plain ndarray or torch.Tensor with shape (n_speakers, dim)
        # Row order matches annotation.labels() (sorted alphabetically by pyannote).
        is_tensor = False
        try:
            import torch
            is_tensor = isinstance(raw, torch.Tensor)
        except ImportError:
            pass

        if is_tensor or isinstance(raw, np.ndarray):
            arr = raw.detach().cpu().numpy() if is_tensor else np.array(raw, dtype=np.float32)
            if arr.ndim == 2:
                n = arr.shape[0]
                if labels is not None and len(labels) == n:
                    for i, label in enumerate(labels):
                        result[str(label)] = arr[i].astype(np.float32).flatten()
                else:
                    # Fallback: synthetic IDs when no label list is provided
                    for i in range(n):
                        result[f"SPEAKER_{i:02d}"] = arr[i].astype(np.float32).flatten()
                return result

        # Case 4: iterable of (label, embedding) pairs
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                label, emb = item
                result[str(label)] = np.array(emb, dtype=np.float32).flatten()
        if result:
            return result

    except Exception:
        pass

    return {}


# ---------------------------------------------------------------------------
# SpeakerProfiles
# ---------------------------------------------------------------------------

class SpeakerProfiles:
    """Manages speaker embedding profiles with cumulative averaging."""

    def __init__(self, profile_dir: str = "profiles"):
        self.profile_dir = Path(profile_dir)
        self._meta: dict[str, dict] = {}           # name -> {n_samples, updated_at}
        self._embeddings: dict[str, np.ndarray] = {}  # name -> mean embedding

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load profiles from disk. Safe to call even if directory is empty."""
        meta_path = self.profile_dir / "speakers.json"
        if not meta_path.exists():
            return
        with open(meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)
        for name in list(self._meta):
            emb_path = self.profile_dir / f"{name}.npy"
            if emb_path.exists():
                self._embeddings[name] = np.load(str(emb_path))
            else:
                # metadata exists but .npy is missing — remove the orphan
                del self._meta[name]

    def save(self) -> None:
        """Persist profiles to disk."""
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        with open(self.profile_dir / "speakers.json", "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        for name, emb in self._embeddings.items():
            np.save(str(self.profile_dir / f"{name}.npy"), emb)

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """
        Add one embedding sample to a speaker's profile.

        Uses Welford-style online mean update so the mean embedding stays
        accurate without storing individual samples:
            new_mean = (old_mean * n + new_emb) / (n + 1)
        Both the stored mean and the incoming embedding are L2-normalized.
        """
        embedding = _l2_normalize(embedding)

        if name in self._embeddings:
            n = self._meta[name]["n_samples"]
            new_mean = _l2_normalize(
                self._embeddings[name] * n / (n + 1) + embedding / (n + 1)
            )
            self._embeddings[name] = new_mean
            self._meta[name]["n_samples"] = n + 1
        else:
            self._embeddings[name] = embedding
            self._meta[name] = {"n_samples": 1, "updated_at": str(date.today())}

        self._meta[name]["updated_at"] = str(date.today())

    def enroll_bulk(self, name: str, embeddings: list[np.ndarray]) -> None:
        """Enroll multiple embeddings for the same speaker at once."""
        for emb in embeddings:
            self.enroll(name, emb)

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(
        self,
        embedding: np.ndarray,
        threshold: float = 0.65,
    ) -> tuple[Optional[str], float]:
        """
        Find the closest known speaker by cosine similarity.

        Returns (name, similarity) if above threshold, else (None, similarity).
        Both stored profiles and the query are L2-normalized, so dot-product
        equals cosine similarity.
        """
        if not self._embeddings:
            return None, 0.0

        query = _l2_normalize(embedding)
        best_name: Optional[str] = None
        best_sim = -1.0

        for name, ref in self._embeddings.items():
            sim = float(np.dot(query, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < threshold:
            return None, best_sim
        return best_name, best_sim

    def match_speakers(
        self,
        speaker_embeddings: dict[str, np.ndarray],
        threshold: float = 0.65,
    ) -> dict[str, str]:
        """
        Map {SPEAKER_xx: embedding} to {SPEAKER_xx: name_or_SPEAKER_xx}.

        Resolves conflicts greedily (highest similarity first) so that at most
        one SPEAKER_xx is assigned to any given name.
        Unknown speakers (below threshold) keep their SPEAKER_xx id.
        """
        if not speaker_embeddings:
            return {}

        # Score all pairs
        scored: list[tuple[str, Optional[str], float]] = []
        for spk_id, emb in speaker_embeddings.items():
            name, sim = self.identify(emb, threshold)
            scored.append((spk_id, name, sim))

        # Greedy assignment: highest similarity first
        scored.sort(key=lambda x: -x[2])
        used_names: set[str] = set()
        result: dict[str, str] = {}

        for spk_id, name, sim in scored:
            if name and name not in used_names:
                result[spk_id] = name
                used_names.add(name)
            else:
                result[spk_id] = spk_id  # fallback: keep anonymous ID

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_speakers(self) -> list[str]:
        return sorted(self._embeddings.keys())

    def __len__(self) -> int:
        return len(self._embeddings)

    def __contains__(self, name: str) -> bool:
        return name in self._embeddings

    def delete(self, name: str) -> bool:
        """Remove a speaker profile. Returns True if found and deleted."""
        if name not in self._embeddings:
            return False
        del self._embeddings[name]
        del self._meta[name]
        npy_path = self.profile_dir / f"{name}.npy"
        if npy_path.exists():
            npy_path.unlink()
        return True
