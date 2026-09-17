"""Turn a finished ACE-Step job into ASS artifacts. Pure stdlib, no web deps.

Kept apart from ``ass_contract_routes`` (which pulls in FastAPI/torch) so the
reshaping logic — the fiddly bit — can be unit-tested without a GPU or a web
stack. A job is not "a WAV": ACE-Step emits the audio *plus* the 5Hz blueprint,
the lyrics and the metas, and the artifact is the unit that carries all of it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


# Real MIME per extension — ASS records content_type verbatim, so a lie here
# becomes a lie in someone's browser.
_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
}


def content_type_for(name: str) -> str:
    _, ext = os.path.splitext(name.lower())
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


def text_artifact(name: str, kind: str, text: str) -> Dict[str, Any]:
    """An artifact whose bytes we synthesise in-memory (codes, lyrics, metadata)."""
    payload = text.encode("utf-8")
    return {
        "name": name,
        "kind": kind,
        "content_type": content_type_for(name),
        "bytes": len(payload),
        "_src": ("text", text),
    }


def public(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the internal ``_src`` down to the ASS artifact shape ASS ingests."""
    return {k: v for k, v in artifact.items() if k != "_src"}


def artifacts_for(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Enumerate a succeeded job's artifacts from its stored ``result`` dict.

    Each carries an internal ``_src``: ``("file", path)`` for audio on disk,
    ``("text", string)`` for metadata we build here. Order: audio first, then the
    blueprint, then lyrics, then the JSON bundle.
    """
    result = result or {}
    artifacts: List[Dict[str, Any]] = []

    # 1) The audio — one clip per batch element. raw_audio_paths are real on-disk
    #    paths (audio_paths is the same list dressed as the old /v1/audio URLs).
    for path in result.get("raw_audio_paths", []) or []:
        if not path:
            continue
        name = os.path.basename(path)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        artifacts.append(
            {
                "name": name,
                "kind": "audio",
                "content_type": content_type_for(name),
                "bytes": size,
                "_src": ("file", path),
            }
        )

    # 2) The 5Hz blueprint the LM planned. Empty on a thinking=false / turbo run;
    #    only surfaced when it exists. Feeding it back as audio_code_string is the
    #    cover/reuse hook, so it must not be discarded the way stock ACE-Step did.
    codes = result.get("audio_codes") or ""
    if codes:
        artifacts.append(text_artifact("audio_codes.txt", "metadata", codes))

    # 3) Lyrics — the LM's rewrite when it ran, else what the caller sent.
    lyrics = result.get("cot_lyrics") or result.get("lyrics") or ""
    if lyrics:
        artifacts.append(text_artifact("lyrics.txt", "lyrics", lyrics))

    # 4) The rest worth keeping, as one JSON blob: the metas the DiT used, the
    #    seed (for reproduction), and which models ran.
    meta_blob = {
        "metas": result.get("metas", {}),
        "seed_value": result.get("seed_value", ""),
        "dit_model": result.get("dit_model", ""),
        "lm_model": result.get("lm_model", ""),
        "prompt": result.get("prompt", ""),
        "generation_info": result.get("generation_info", ""),
    }
    artifacts.append(
        text_artifact(
            "metadata.json",
            "metadata",
            json.dumps(meta_blob, ensure_ascii=False, indent=2),
        )
    )
    return artifacts
