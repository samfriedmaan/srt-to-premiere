"""
captions.py — unified caption tool

Commands:
  transcribe  <audio.mp3> [--language de] [--model large-v3-turbo]
  apply-edits <audio.json> <edited_client.txt>
  from-srt    <captions.srt>
"""

import difflib
import json
import os
import re
import sys
import uuid


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def fix_icloud_path(path):
    if "comappleCloudDocs" in path:
        return path.replace("comappleCloudDocs", "com~apple~CloudDocs")
    return path


def make_word_obj(text, start, duration, confidence=1.0):
    is_eos = text.rstrip().endswith(('.', '?', '!', ','))
    return {
        "confidence": round(confidence, 3),
        "duration": round(max(duration, 0.001), 3),
        "eos": is_eos,
        "start": round(start, 3),
        "tags": [],
        "text": text,
        "type": "word",
    }


def words_to_segments(words, speaker_id, premiere_language="en-us"):
    """Group a flat word list into Premiere segments (max 15 words or EOS)."""
    segments = []
    current = []
    segment_start = 0.0

    for word in words:
        if not current:
            segment_start = word["start"]
        current.append(word)
        if word["eos"] or len(current) >= 15:
            seg_end = current[-1]["start"] + current[-1]["duration"]
            segments.append({
                "duration": round(seg_end - segment_start, 3),
                "language": premiere_language,
                "speaker": speaker_id,
                "start": round(segment_start, 3),
                "words": current,
            })
            current = []

    if current:
        seg_end = current[-1]["start"] + current[-1]["duration"]
        segments.append({
            "duration": round(seg_end - segment_start, 3),
            "language": premiere_language,
            "speaker": speaker_id,
            "start": round(segment_start, 3),
            "words": current,
        })

    return segments


def write_premiere_json(segments, speaker_id, premiere_language, output_path):
    data = {
        "language": premiere_language,
        "segments": segments,
        "speakers": [{"id": speaker_id, "name": "Speaker 1"}],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

def cmd_transcribe(args):
    import argparse
    parser = argparse.ArgumentParser(prog="captions.py transcribe")
    parser.add_argument("audio", help="Path to MP3 (or any audio) file")
    parser.add_argument("--language", default="de", help="Whisper language code (default: de)")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name (default: large-v3-turbo)")
    parser.add_argument("--premiere-language", default="en-us", dest="premiere_language",
                        help="Language tag written into Premiere JSON (default: en-us)")
    opts = parser.parse_args(args)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed. Run:  pip install faster-whisper")
        sys.exit(1)

    audio_path = fix_icloud_path(opts.audio)
    base = os.path.splitext(audio_path)[0]
    json_path = base + ".json"
    txt_path = base + "_client.txt"

    print(f"Loading model '{opts.model}' …")
    model = WhisperModel(opts.model, device="auto", compute_type="auto")

    print(f"Transcribing {audio_path}  (language={opts.language}) …")
    segments_iter, _info = model.transcribe(
        audio_path,
        language=opts.language,
        word_timestamps=True,
    )

    speaker_id = str(uuid.uuid4())
    all_words = []
    txt_blocks = []  # list of (timestamp_str, words_in_segment)

    for seg in segments_iter:
        raw_words = list(seg.words)
        if not raw_words:
            continue

        seg_words = []
        for w in raw_words:
            text = w.word.strip()
            if not text:
                continue
            duration = w.end - w.start
            word_obj = make_word_obj(text, w.start, duration, confidence=w.probability)
            all_words.append(word_obj)
            seg_words.append(text)

        txt_blocks.append((seconds_to_hms(seg.start), seg_words))

    if not all_words:
        print("No words found in transcription.")
        sys.exit(1)

    # Write Premiere JSON
    segments = words_to_segments(all_words, speaker_id, opts.premiere_language)
    write_premiere_json(segments, speaker_id, opts.premiere_language, json_path)
    print(f"Wrote {json_path}  ({len(all_words)} words, {len(segments)} segments)")

    # Write client TXT
    lines = []
    for ts, words in txt_blocks:
        lines.append(f"[{ts}]")
        lines.append(" ".join(words))
        lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {txt_path}")


# ---------------------------------------------------------------------------
# apply-edits
# ---------------------------------------------------------------------------

def normalize(word):
    """Strip punctuation and lowercase for fuzzy alignment."""
    return re.sub(r"[^\w]", "", word.lower())


def cmd_apply_edits(args):
    import argparse
    parser = argparse.ArgumentParser(prog="captions.py apply-edits")
    parser.add_argument("json_file", help="Original Premiere JSON (with word timings)")
    parser.add_argument("txt_file", help="Client-edited TXT file")
    opts = parser.parse_args(args)

    json_path = fix_icloud_path(opts.json_file)
    txt_path = fix_icloud_path(opts.txt_file)

    base = os.path.splitext(json_path)[0]
    out_path = (json_path if base.endswith("_updated")
                else base + "_updated.json")

    # Load original words
    with open(json_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    speaker_id = original_data["speakers"][0]["id"]
    premiere_language = original_data.get("language", "en-us")

    orig_words = [w for seg in original_data["segments"] for w in seg["words"]]

    # Parse TXT — strip [H:MM:SS] timestamp lines
    timestamp_re = re.compile(r"^\[\d+:\d{2}:\d{2}\]$")
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()

    new_words_raw = []
    for line in txt_content.splitlines():
        line = line.strip()
        if not line or timestamp_re.match(line):
            continue
        new_words_raw.extend(line.split())

    # Sequence alignment on normalized keys
    orig_keys = [normalize(w["text"]) for w in orig_words]
    new_keys = [normalize(w) for w in new_words_raw]

    sm = difflib.SequenceMatcher(None, orig_keys, new_keys, autojunk=False)
    result_words = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        orig_block = orig_words[i1:i2]
        new_block = new_words_raw[j1:j2]

        if tag == "equal":
            # Use new text (preserves client capitalization fixes)
            for orig, new_text in zip(orig_block, new_block):
                w = dict(orig)
                w["text"] = new_text
                result_words.append(w)

        elif tag == "replace":
            # Distribute the original time span evenly across new words
            total_start = orig_block[0]["start"]
            total_end = orig_block[-1]["start"] + orig_block[-1]["duration"]
            word_dur = (total_end - total_start) / len(new_block)
            for k, new_text in enumerate(new_block):
                result_words.append(make_word_obj(new_text, total_start + k * word_dur, word_dur))

        elif tag == "delete":
            pass  # client removed these words

        elif tag == "insert":
            # Synthesize timing from the gap before the next original word
            prev_end = (result_words[-1]["start"] + result_words[-1]["duration"]
                        if result_words else 0.0)
            next_start = (orig_words[i2]["start"]
                          if i2 < len(orig_words) else prev_end + 0.1 * len(new_block))
            gap = max(next_start - prev_end, 0.0)
            word_dur = max(gap / len(new_block), 0.05) if new_block else 0.05
            for k, new_text in enumerate(new_block):
                result_words.append(make_word_obj(new_text, prev_end + k * word_dur, word_dur))

    segments = words_to_segments(result_words, speaker_id, premiere_language)
    write_premiere_json(segments, speaker_id, premiere_language, out_path)
    print(f"Wrote {out_path}  ({len(result_words)} words, {len(segments)} segments)")


# ---------------------------------------------------------------------------
# from-srt
# ---------------------------------------------------------------------------

def cmd_from_srt(args):
    import argparse
    parser = argparse.ArgumentParser(prog="captions.py from-srt")
    parser.add_argument("srt_file", help="Path to SRT file")
    opts = parser.parse_args(args)

    srt_path = fix_icloud_path(opts.srt_file)
    output_path = os.path.splitext(srt_path)[0] + ".json"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from srt_to_json import parse_srt, convert_to_premiere_json

    print(f"Converting {srt_path} → {output_path} …")
    srt_entries = parse_srt(srt_path)
    premiere_json = convert_to_premiere_json(srt_entries)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(premiere_json, f, separators=(",", ":"))
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "transcribe": cmd_transcribe,
    "apply-edits": cmd_apply_edits,
    "from-srt": cmd_from_srt,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands:")
        print("  transcribe  <audio.mp3> [--language de] [--model large-v3-turbo]")
        print("  apply-edits <audio.json> <edited_client.txt>")
        print("  from-srt    <captions.srt>")
        sys.exit(1)

    COMMANDS[sys.argv[1]](sys.argv[2:])
