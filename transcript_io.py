"""Input normalization and Premiere Pro transcript JSON generation.

The input side intentionally accepts the formats commonly produced by
captioning and transcription tools:

* SRT/WebVTT-style millisecond timestamps
* frame-based timeline timestamps (``HH:MM:SS:FF``)
* Premiere transcript JSON and simple JSON cue/segment arrays

Inputs that only have cue-level timing are converted to word-level timing by
evenly distributing each cue's duration across its words.  That is an
approximation; true word timing can only come from a word-timed source.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import uuid
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LANGUAGE = "en-us"
DEFAULT_FPS = 30.0
MIN_SYNTHETIC_WORD_DURATION = 0.001
MAX_WORDS_PER_SEGMENT = 15

TIMECODE = r"\d{1,3}:\d{2}:\d{2}(?::\d{1,3}|[,.]\d{1,3})?"
TIMESTAMP_LINE_RE = re.compile(
    rf"^(?P<start>{TIMECODE})[ \t]*(?:-->|-)[ \t]*(?P<end>{TIMECODE})(?:[ \t]+[^\r\n]*)?$",
    re.MULTILINE,
)
SPEAKER_LINE_RE = re.compile(r"^\s*(?:speaker|spk)\s*[:#-]?\s*(.+?)\s*$", re.IGNORECASE)
MARKUP_RE = re.compile(r"</?(?:i|b|u|font)(?:\s[^>]*)?>", re.IGNORECASE)


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None = None
    source_index: int = 0


@dataclass
class ParseReport:
    format_name: str
    entries_read: int
    entries_dropped: int = 0
    overlaps_clipped: int = 0
    fps: float | None = None


def _timecode_has_frames(value: str) -> bool:
    return len(value.split(":")) == 4


def detect_frame_rate(timecodes: Iterable[str], requested_fps: float | None = None) -> float:
    """Choose a frame rate for ``HH:MM:SS:FF`` timecodes.

    30 fps is the least surprising default when all frame values are below 30.
    A value of 30 or more requires at least 60 fps, which lets the common 60
    fps timeline export work without an extra flag.  Ambiguous 24/25/30 fps
    files can always use ``--fps`` to override this choice.
    """
    if requested_fps is not None:
        if requested_fps <= 0:
            raise ValueError("--fps must be greater than zero")
        return requested_fps

    frame_values = [int(value.split(":")[-1]) for value in timecodes if _timecode_has_frames(value)]
    if not frame_values:
        return DEFAULT_FPS
    if max(frame_values) >= 30:
        return 60.0
    return DEFAULT_FPS


def parse_timecode(value: str, fps: float = DEFAULT_FPS) -> float:
    """Convert SRT/VTT milliseconds or timeline frames to seconds."""
    parts = value.split(":")
    if len(parts) == 4:
        hours, minutes, seconds, frames = map(int, parts)
        if frames >= fps:
            raise ValueError(f"Frame value {frames} is invalid for {fps:g} fps in {value}")
        return (hours * 3600) + (minutes * 60) + seconds + (frames / fps)

    if len(parts) != 3:
        raise ValueError(f"Unsupported timestamp: {value}")
    hours, minutes = map(int, parts[:2])
    seconds_part = parts[2]
    fraction = 0
    if "," in seconds_part or "." in seconds_part:
        seconds_part, fraction_part = re.split(r"[,.]", seconds_part, maxsplit=1)
        fraction = int(fraction_part.ljust(3, "0")[:3])
    seconds = int(seconds_part)
    return (hours * 3600) + (minutes * 60) + seconds + (fraction / 1000)


def _clean_cue_text(raw_body: str) -> tuple[str, str | None]:
    lines = [line.strip() for line in raw_body.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    speaker = None
    text_lines = []
    for line in lines:
        if not line:
            continue
        if line.isdigit():
            # SRT cue numbers sit between one timestamp block and the next.
            continue
        speaker_match = SPEAKER_LINE_RE.fullmatch(line)
        if speaker_match and speaker is None:
            speaker_value = speaker_match.group(1).strip()
            speaker = f"Speaker {speaker_value}" if speaker_value.isdigit() else speaker_value
            continue
        text_lines.append(MARKUP_RE.sub("", line))
    return " ".join(text_lines).strip(), speaker


def parse_timestamped_text(content: str, fps: float | None = None) -> tuple[list[Cue], ParseReport]:
    """Parse SRT/VTT or frame-based timeline text."""
    matches = list(TIMESTAMP_LINE_RE.finditer(content))
    if not matches:
        raise ValueError("No supported timestamp ranges found")

    timestamp_values = [value for match in matches for value in match.group("start", "end")]
    effective_fps = detect_frame_rate(timestamp_values, fps)
    frame_based = any(_timecode_has_frames(value) for value in timestamp_values)
    entries: list[Cue] = []
    dropped = 0

    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        text, speaker = _clean_cue_text(content[match.end() : body_end])
        if not text:
            dropped += 1
            continue
        start = parse_timecode(match.group("start"), effective_fps)
        end = parse_timecode(match.group("end"), effective_fps)
        if end <= start:
            dropped += 1
            continue
        entries.append(Cue(start, end, text, speaker=speaker, source_index=index))

    return entries, ParseReport(
        format_name="timeline" if frame_based else "srt/vtt",
        entries_read=len(matches),
        entries_dropped=dropped,
        fps=effective_fps if frame_based else None,
    )


def parse_srt(srt_file_path: str | Path, fps: float | None = None) -> list[dict[str, Any]]:
    """Backward-compatible wrapper returning the old cue-dictionary shape."""
    content = Path(srt_file_path).read_text(encoding="utf-8-sig")
    entries, _ = parse_timestamped_text(content, fps=fps)
    return [{"start": cue.start, "end": cue.end, "text": cue.text, "speaker": cue.speaker} for cue in entries]


def resolve_cue_overlaps(cues: list[Cue], policy: str = "sequential") -> tuple[list[Cue], int, int]:
    """Resolve overlapping cues while preserving their source order.

    ``sequential`` clips the earlier cue at the next cue's start. If two cues
    start together, the later cue wins and the earlier cue is dropped. This is
    the safest behavior for transcript import because Premiere gets one
    unambiguous text sequence instead of simultaneous words.
    """
    if policy not in {"sequential", "preserve"}:
        raise ValueError("overlap policy must be 'sequential' or 'preserve'")
    ordered = sorted(cues, key=lambda cue: (cue.start, cue.source_index))
    if policy == "preserve":
        return ordered, 0, 0

    resolved: list[Cue] = []
    clipped = 0
    dropped = 0
    for cue in ordered:
        if cue.end <= cue.start:
            dropped += 1
            continue
        if resolved and cue.start < resolved[-1].end:
            previous = resolved[-1]
            if cue.start <= previous.start:
                resolved.pop()
                dropped += 1
            else:
                previous.end = cue.start
                clipped += 1
        if cue.end > cue.start:
            resolved.append(cue)
        else:
            dropped += 1
    return resolved, clipped, dropped


def make_word_obj(text: str, start: float, duration: float, confidence: float = 1.0,
                  minimum_duration: float = 0.08) -> dict[str, Any]:
    return {
        "confidence": round(confidence, 3),
        "duration": round(max(duration, minimum_duration), 3),
        "eos": text.rstrip().endswith((".", "?", "!", ",")),
        "start": round(start, 3),
        "tags": [],
        "text": text,
        "type": "word",
    }


def _speaker_catalog(cues: list[Cue], speaker_name: str | None = None) -> tuple[list[dict[str, str]], dict[str, str]]:
    names = []
    for cue in cues:
        if cue.speaker and cue.speaker not in names:
            names.append(cue.speaker)
    if not names:
        names = [speaker_name or "Speaker 1"]
    ids = {name: str(uuid.uuid4()) for name in names}
    return ([{"id": ids[name], "name": name} for name in names], ids)


def cues_to_words(cues: list[Cue], speaker_ids: dict[str, str], default_speaker: str) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for cue in cues:
        cue_words = cue.text.split()
        if not cue_words:
            continue
        duration = (cue.end - cue.start) / len(cue_words)
        speaker_id = speaker_ids.get(cue.speaker or default_speaker, speaker_ids[default_speaker])
        for index, text in enumerate(cue_words):
            word = make_word_obj(
                text,
                cue.start + (index * duration),
                duration,
                minimum_duration=MIN_SYNTHETIC_WORD_DURATION,
            )
            word["_speaker"] = speaker_id
            words.append(word)
    return words


def _flush_segment(segments: list[dict[str, Any]], current: list[dict[str, Any]], language: str) -> None:
    if not current:
        return
    start = current[0]["start"]
    end = current[-1]["start"] + current[-1]["duration"]
    speaker = current[0].get("_speaker")
    clean_words = []
    for word in current:
        clean_word = dict(word)
        clean_word.pop("_speaker", None)
        clean_words.append(clean_word)
    segments.append({
        "duration": round(end - start, 3),
        "language": language,
        "speaker": speaker,
        "start": round(start, 3),
        "words": clean_words,
    })


def words_to_segments(words: list[dict[str, Any]], language: str = DEFAULT_LANGUAGE) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_speaker = None
    for word in words:
        speaker = word.get("_speaker")
        if current and speaker != current_speaker:
            _flush_segment(segments, current, language)
            current = []
        if not current:
            current_speaker = speaker
        current.append(word)
        if word["eos"] or len(current) >= MAX_WORDS_PER_SEGMENT:
            _flush_segment(segments, current, language)
            current = []
    _flush_segment(segments, current, language)
    return segments


def _normalize_words(words: list[dict[str, Any]], overlap_policy: str) -> tuple[list[dict[str, Any]], int, int]:
    ordered = sorted(enumerate(words), key=lambda item: (item[1]["start"], item[0]))
    if overlap_policy == "preserve":
        return [word for _, word in ordered], 0, 0
    if overlap_policy != "sequential":
        raise ValueError("overlap policy must be 'sequential' or 'preserve'")

    result: list[dict[str, Any]] = []
    clipped = 0
    dropped = 0
    for _, source_word in ordered:
        word = dict(source_word)
        word_end = word["start"] + word["duration"]
        if word_end <= word["start"]:
            dropped += 1
            continue
        if result:
            previous = result[-1]
            previous_end = previous["start"] + previous["duration"]
            if word["start"] <= previous["start"] + 1e-6:
                result.pop()
                dropped += 1
            elif word["start"] < previous_end - 0.0011:
                previous["duration"] = round(word["start"] - previous["start"], 3)
                clipped += 1
        if word["duration"] > 0:
            result.append(word)
        else:
            dropped += 1
    return result, clipped, dropped


def _premiere_data_from_json(data: Any, language_override: str | None, overlap_policy: str) -> tuple[dict[str, Any], ParseReport]:
    if isinstance(data, dict) and "segments" in data and any("words" in segment for segment in data["segments"]):
        language = language_override or data.get("language", DEFAULT_LANGUAGE)
        speakers = data.get("speakers") or [{"id": str(uuid.uuid4()), "name": "Speaker 1"}]
        words = []
        for segment in data["segments"]:
            speaker = segment.get("speaker")
            for word in segment.get("words", []):
                copy = dict(word)
                copy["_speaker"] = speaker
                words.append(copy)
        words, clipped, dropped = _normalize_words(words, overlap_policy)
        segments = words_to_segments(words, language)
        return {"language": language, "segments": segments, "speakers": speakers}, ParseReport(
            format_name="premiere-json",
            entries_read=len(data["segments"]),
            entries_dropped=dropped,
            overlaps_clipped=clipped,
        )

    if isinstance(data, dict):
        raw_entries = data.get("segments") or data.get("cues") or data.get("entries")
        language = language_override or data.get("language", DEFAULT_LANGUAGE)
    elif isinstance(data, list):
        raw_entries = data
        language = language_override or DEFAULT_LANGUAGE
    else:
        raise ValueError("JSON must be an object or array")

    if not isinstance(raw_entries, list):
        raise ValueError("JSON does not contain segments, cues, or entries")
    cues = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            continue
        start = item.get("start", item.get("startTime"))
        end = item.get("end", item.get("endTime"))
        text = item.get("text", item.get("content"))
        if start is None or end is None or text is None:
            continue
        cues.append(Cue(float(start), float(end), str(text), item.get("speaker"), index))
    return _convert_cues(cues, language, overlap_policy, format_name="json-cues")


def _convert_cues(cues: list[Cue], language: str, overlap_policy: str, format_name: str,
                  speaker_name: str | None = None) -> tuple[dict[str, Any], ParseReport]:
    resolved, clipped, dropped = resolve_cue_overlaps(cues, overlap_policy)
    speakers, speaker_ids = _speaker_catalog(resolved, speaker_name=speaker_name)
    default_speaker = speakers[0]["name"]
    words = cues_to_words(resolved, speaker_ids, default_speaker)
    segments = words_to_segments(words, language)
    return {"language": language, "segments": segments, "speakers": speakers}, ParseReport(
        format_name=format_name,
        entries_read=len(cues),
        entries_dropped=dropped,
        overlaps_clipped=clipped,
    )


def convert_cues(cues: list[Cue], language: str = DEFAULT_LANGUAGE,
                 overlap_policy: str = "sequential", speaker_name: str | None = None) -> dict[str, Any]:
    """Convert already-parsed cue objects to Premiere transcript JSON."""
    data, _ = _convert_cues(cues, language, overlap_policy, format_name="cues", speaker_name=speaker_name)
    return data


def convert_file(input_path: str | Path, language: str | None = None, fps: float | None = None,
                 overlap_policy: str = "sequential", speaker_name: str | None = None) -> tuple[dict[str, Any], ParseReport]:
    path = Path(input_path)
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json" or content.lstrip().startswith(("{", "[")):
        return _premiere_data_from_json(json.loads(content), language, overlap_policy)

    cues, report = parse_timestamped_text(content, fps=fps)
    data, conversion_report = _convert_cues(
        cues,
        language or DEFAULT_LANGUAGE,
        overlap_policy,
        format_name=report.format_name,
        speaker_name=speaker_name,
    )
    conversion_report.entries_read = report.entries_read
    conversion_report.entries_dropped += report.entries_dropped
    conversion_report.fps = report.fps
    return data, conversion_report
