import json
import tempfile
import unittest
from pathlib import Path

from transcript_io import Cue, convert_cues, convert_file, parse_timestamped_text, resolve_cue_overlaps


class TranscriptIOTests(unittest.TestCase):
    @staticmethod
    def words(data):
        return [word for segment in data["segments"] for word in segment["words"]]

    def test_numeric_speech_is_not_confused_with_srt_cue_numbers(self):
        content = """1
00:00:01,000 --> 00:00:01,540
100

2
00:00:01,540 --> 00:00:02,000
percent
2026

3
00:00:02,000 --> 00:00:02,060
42
"""
        cues, report = parse_timestamped_text(content)
        self.assertEqual([cue.text for cue in cues], ["100", "percent 2026", "42"])
        self.assertEqual(report.entries_dropped, 0)

    def test_numeric_speech_in_unnumbered_timestamped_text(self):
        content = "00:00:01,000 --> 00:00:02,000\n100\n00:00:02,000 --> 00:00:03,000\n200\n"
        cues, _ = parse_timestamped_text(content)
        self.assertEqual([cue.text for cue in cues], ["100", "200"])

    def test_numbered_srt_without_blank_lines(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\n100\n2\n00:00:02,000 --> 00:00:03,000\n200\n"
        cues, _ = parse_timestamped_text(content)
        self.assertEqual([cue.text for cue in cues], ["100", "200"])

    def test_word_level_sample_preserves_short_and_zero_duration_words(self):
        # Representative cases from the TEMPORARY SAMPLE: numbers, a 60 ms
        # word, repeated zero-length cues followed by speech at the same time,
        # a pause, and zero-length words at the end of the file.
        expected = [
            (53.6, 54.14, "100"),
            (54.14, 54.2, "percent."),
            (649.4, 649.4, "that"),
            (649.4, 649.4, "they"),
            (649.4, 649.48, "of"),
            (649.48, 649.48, "what"),
            (650.0, 650.2, "happened?"),
            (806.06, 806.06, "on"),
            (806.06, 806.06, "the"),
            (806.06, 806.06, "board?"),
        ]
        def timestamp(value):
            milliseconds = round(value * 1000)
            return f"00:{milliseconds // 60000:02d}:{milliseconds // 1000 % 60:02d},{milliseconds % 1000:03d}"
        content = "\n\n".join(
            f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}"
            for index, (start, end, text) in enumerate(expected, 1)
        ) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "word-timed.srt"
            # Also exercise the BOM and Windows line endings common in exports.
            path.write_bytes(("\ufeff" + content.replace("\n", "\r\n")).encode("utf-8"))
            for policy in ("sequential", "preserve"):
                with self.subTest(policy=policy):
                    data, report = convert_file(path, overlap_policy=policy)
                    words = self.words(data)
                    self.assertEqual(len(words), len(expected))
                    for word, (start, end, text) in zip(words, expected):
                        self.assertEqual(word["text"], text)
                        self.assertAlmostEqual(word["start"], start)
                        self.assertAlmostEqual(word["duration"], end - start)
                    self.assertEqual(report.entries_dropped, 0)
                    self.assertEqual(report.overlaps_clipped, 0)
                    self.assertEqual(report.zero_duration_words, 6)

    def test_zero_duration_cues_do_not_hide_positive_duration_overlaps(self):
        cues = [Cue(0, 2, "first"), Cue(1, 1, "point", source_index=1),
                Cue(1, 3, "second", source_index=2)]
        resolved, clipped, dropped = resolve_cue_overlaps(cues)
        self.assertEqual([(cue.text, cue.start, cue.end) for cue in resolved],
                         [("first", 0, 1), ("point", 1, 1), ("second", 1, 3)])
        self.assertEqual((clipped, dropped), (1, 0))
        self.assertEqual(cues[0].end, 2)  # Normalization must not change callers' input.

    def test_negative_duration_is_dropped_but_zero_duration_is_retained(self):
        cues, report = parse_timestamped_text(
            "1\n00:00:02,000 --> 00:00:01,000\ninvalid\n\n"
            "2\n00:00:02,000 --> 00:00:02,000\nvalid\n"
        )
        self.assertEqual([cue.text for cue in cues], ["valid"])
        self.assertEqual(report.entries_dropped, 1)

    def test_preserved_overlap_segment_encloses_every_word(self):
        data = convert_cues([Cue(0, 10, "long"), Cue(1, 2, "short")], overlap_policy="preserve")
        self.assertEqual(data["segments"][0]["duration"], 10)

    def test_cue_level_text_still_uses_estimated_word_timing(self):
        words = self.words(convert_cues([Cue(1, 3, "two words")]))
        self.assertEqual([(word["start"], word["duration"]) for word in words], [(1, 1), (2, 1)])

    def test_premiere_json_keeps_repeated_zero_duration_words(self):
        original = convert_cues([Cue(0, 0, "I"), Cue(0, 0, "spoke"), Cue(0, 0.06, "to")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            converted, report = convert_file(path)
        self.assertEqual(self.words(converted), self.words(original))
        self.assertEqual(report.entries_dropped, 0)
        self.assertEqual(report.zero_duration_words, 2)

    def test_premiere_json_overlap_clipping_looks_past_zero_duration_words(self):
        original = convert_cues([Cue(0, 2, "first"), Cue(1, 1, "point", source_index=1),
                                 Cue(1, 3, "second", source_index=2)], overlap_policy="preserve")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            converted, report = convert_file(path)
        self.assertEqual([(word["text"], word["start"], word["duration"]) for word in self.words(converted)],
                         [("first", 0, 1), ("point", 1, 0), ("second", 1, 2)])
        self.assertEqual((report.overlaps_clipped, report.entries_dropped), (1, 0))

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
