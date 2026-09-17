"""Tests for the ASS artifact reshaping — pure stdlib, no GPU / web stack."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from acestep.api.http.ass_artifacts import (
    artifacts_for,
    content_type_for,
    public,
    text_artifact,
)


class ContentTypeTests(unittest.TestCase):
    def test_known_extensions(self):
        self.assertEqual(content_type_for("x.wav"), "audio/wav")
        self.assertEqual(content_type_for("x.MP3"), "audio/mpeg")  # case-insensitive
        self.assertEqual(content_type_for("cover.png"), "image/png")
        self.assertEqual(content_type_for("meta.json"), "application/json")

    def test_unknown_extension_falls_back(self):
        self.assertEqual(content_type_for("mystery.xyz"), "application/octet-stream")


class TextArtifactTests(unittest.TestCase):
    def test_byte_count_is_utf8_length_not_char_length(self):
        # 'é' is two bytes in UTF-8 — the size must reflect bytes, not chars.
        art = text_artifact("lyrics.txt", "lyrics", "café")
        self.assertEqual(art["bytes"], 5)
        self.assertEqual(art["kind"], "lyrics")
        self.assertEqual(art["_src"], ("text", "café"))

    def test_public_strips_internal_src(self):
        art = text_artifact("meta.json", "metadata", "{}")
        pub = public(art)
        self.assertNotIn("_src", pub)
        self.assertEqual(set(pub), {"name", "kind", "content_type", "bytes"})


class ArtifactsForTests(unittest.TestCase):
    def test_empty_result_still_yields_metadata(self):
        # Even a bare success gets the metadata.json bundle — nothing else.
        arts = artifacts_for({})
        self.assertEqual([a["name"] for a in arts], ["metadata.json"])
        self.assertEqual(arts[0]["kind"], "metadata")

    def test_full_mixed_kind_set_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "output_0.wav")
            with open(wav, "wb") as f:
                f.write(b"RIFF____fake____")
            result = {
                "raw_audio_paths": [wav],
                "audio_codes": "<|audio_code_1|><|audio_code_2|>",
                "cot_lyrics": "la la la",
                "metas": {"bpm": 120, "keyscale": "C Major"},
                "seed_value": "424242",
                "dit_model": "acestep-v15-xl-turbo",
                "lm_model": "acestep-5Hz-lm-4B",
            }
            arts = artifacts_for(result)

        names = [a["name"] for a in arts]
        self.assertEqual(
            names, ["output_0.wav", "audio_codes.txt", "lyrics.txt", "metadata.json"]
        )
        kinds = {a["name"]: a["kind"] for a in arts}
        self.assertEqual(kinds["output_0.wav"], "audio")
        self.assertEqual(kinds["audio_codes.txt"], "metadata")
        self.assertEqual(kinds["lyrics.txt"], "lyrics")

        audio = arts[0]
        self.assertEqual(audio["content_type"], "audio/wav")
        self.assertEqual(audio["bytes"], len(b"RIFF____fake____"))
        self.assertEqual(audio["_src"], ("file", wav))

        meta = next(a for a in arts if a["name"] == "metadata.json")
        blob = json.loads(meta["_src"][1])
        self.assertEqual(blob["seed_value"], "424242")
        self.assertEqual(blob["dit_model"], "acestep-v15-xl-turbo")
        self.assertEqual(blob["metas"]["bpm"], 120)

    def test_missing_audio_file_reports_zero_bytes_not_crash(self):
        arts = artifacts_for({"raw_audio_paths": ["/nope/gone.wav"]})
        audio = arts[0]
        self.assertEqual(audio["name"], "gone.wav")
        self.assertEqual(audio["bytes"], 0)

    def test_lyrics_falls_back_to_plain_when_no_cot(self):
        arts = artifacts_for({"lyrics": "plain words"})
        lyric = next(a for a in arts if a["name"] == "lyrics.txt")
        self.assertEqual(lyric["_src"], ("text", "plain words"))

    def test_no_codes_no_lyrics_means_no_such_artifacts(self):
        arts = artifacts_for({"raw_audio_paths": [], "audio_codes": "", "lyrics": ""})
        self.assertEqual([a["name"] for a in arts], ["metadata.json"])


if __name__ == "__main__":
    unittest.main()
