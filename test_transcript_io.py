import json
import tempfile
import unittest
from pathlib import Path

from transcript_io import Cue, convert_file, parse_timestamped_text, resolve_cue_overlaps


class TranscriptIOTests(unittest.TestCase):
    def test_srt_timestamps_and_speaker_label(self):
        content = """1
00:00:00,000 --> 00:00:01,000
Speaker 1
Hallo Welt.

2
00:00:01,000 --> 00:00:02.500
Das ist ein Test.
"""
        entries, report = parse_timestamped_text(content)
        self.assertEqual(report.format_name, "srt/vtt")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].speaker, "Speaker 1")
        self.assertEqual(entries[0].text, "Hallo Welt.")
        self.assertAlmostEqual(entries[1].end, 2.5)

    def test_frame_based_timeline_auto_detects_60_fps(self):
        content = "00:00:00:00 - 00:00:00:59\n\nHallo.\n"
        entries, report = parse_timestamped_text(content)
        self.assertEqual(report.format_name, "timeline")
        self.assertEqual(report.fps, 60.0)
        self.assertAlmostEqual(entries[0].end, 59 / 60)

    def test_frame_based_timeline_supports_explicit_25_fps(self):
        content = "00:00:00:00 - 00:00:01:00\n\nHallo Europa.\n"
        entries, report = parse_timestamped_text(content, fps=25)
        self.assertEqual(report.format_name, "timeline")
        self.assertEqual(report.fps, 25)
        self.assertAlmostEqual(entries[0].end, 1.0)

    def test_sequential_overlap_policy_prefers_later_text(self):
        cues = [
            Cue(0.0, 2.0, "first", source_index=0),
            Cue(1.0, 3.0, "second", source_index=1),
            Cue(1.0, 4.0, "replacement", source_index=2),
        ]
        resolved, clipped, dropped = resolve_cue_overlaps(cues, "sequential")
        self.assertEqual([(cue.start, cue.end, cue.text) for cue in resolved], [
            (0.0, 1.0, "first"),
            (1.0, 4.0, "replacement"),
        ])
        self.assertEqual(clipped, 1)
        self.assertEqual(dropped, 1)

    def test_premiere_json_round_trip_preserves_schema_and_speaker(self):
        data = {
            "language": "de-de",
            "speakers": [{"id": "speaker-a", "name": "Speaker 1"}],
            "segments": [{
                "start": 0.0,
                "duration": 1.0,
                "language": "de-de",
                "speaker": "speaker-a",
                "words": [{
                    "confidence": 1,
                    "duration": 1.0,
                    "eos": True,
                    "start": 0.0,
                    "tags": [],
                    "text": "Hallo.",
                    "type": "word",
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            converted, report = convert_file(path)

        self.assertEqual(report.format_name, "premiere-json")
        self.assertEqual(converted["language"], "de-de")
        self.assertEqual(converted["speakers"], data["speakers"])
        self.assertEqual(converted["segments"][0]["words"][0]["text"], "Hallo.")
        self.assertFalse(any(key.startswith("_") for segment in converted["segments"] for word in segment["words"] for key in word))


if __name__ == "__main__":
    unittest.main()
