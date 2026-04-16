"""
captions.py — unified caption tool

Commands:
  transcribe  <audio.mp3> [--language de] [--model large-v3-turbo]
  to-premiere <edited_transcript.txt>
  apply-edits <audio.json> <edited_plain.txt>
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


def seconds_to_srt_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fix_icloud_path(path):
    if "comappleCloudDocs" in path:
        return path.replace("comappleCloudDocs", "com~apple~CloudDocs")
    return path


def make_word_obj(text, start, duration, confidence=1.0):
    is_eos = text.rstrip().endswith(('.', '?', '!', ','))
    return {
        "confidence": round(confidence, 3),
        "duration": round(max(duration, 0.08), 3),
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
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        return output_path
    except PermissionError:
        fallback_path = os.path.basename(output_path)
        print(f"Warning: Permission denied for {output_path}. Saving to current directory as {fallback_path}")
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        return fallback_path


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

def cmd_transcribe(args):
    import argparse
    parser = argparse.ArgumentParser(prog="captions.py transcribe")
    parser.add_argument("audio", help="Path to MP3 (or any audio) file")
    parser.add_argument("--language", default="de", help="Whisper language code (default: de)")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name (default: large-v3-turbo)")
    parser.add_argument("--compute-type", default="float32", dest="compute_type",
                        help="Whisper compute type (default: float32 for stability on Mac)")
    parser.add_argument("--premiere-language", default="en-us", dest="premiere_language",
                        help="Language tag written into Premiere JSON (default: en-us)")
    parser.add_argument("-o", "--output", help="Optional output path for the JSON/TXT (defaults to same folder as audio)")
    opts = parser.parse_args(args)

    try:
        from faster_whisper import WhisperModel
        from tqdm import tqdm
    except ImportError:
        print("Required libraries missing. Run:  pip install faster-whisper tqdm")
        sys.exit(1)

    audio_path = fix_icloud_path(opts.audio)
    base = os.path.splitext(audio_path)[0]
    
    # Initial target paths
    json_path = opts.output if opts.output and opts.output.lower().endswith(".json") else base + ".json"
    txt_path = json_path.replace(".json", "_client.txt") if json_path.endswith(".json") else base + "_client.txt"
    if opts.output and not opts.output.lower().endswith(".json"):
        json_path = os.path.join(opts.output, os.path.basename(base) + ".json")
        txt_path = os.path.join(opts.output, os.path.basename(base) + "_client.txt")

    print(f"Loading model '{opts.model}' (compute_type={opts.compute_type}) …")
    model = WhisperModel(opts.model, device="auto", compute_type=opts.compute_type)

    print(f"Transcribing {audio_path}  (language={opts.language}) …")
    segments_iter, _info = model.transcribe(
        audio_path,
        language=opts.language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    speaker_id = str(uuid.uuid4())
    all_words = []
    srt_blocks = []  # list of (index, start_ts, end_ts, text)

    with tqdm(total=round(_info.duration, 2), unit="sec", desc="Transcribing") as pbar:
        for idx, seg in enumerate(segments_iter, 1):
            raw_words = list(seg.words)
            if not raw_words:
                continue

            seg_text_parts = []
            for w in raw_words:
                text = w.word.strip()
                if not text:
                    continue
                word_obj = make_word_obj(text, w.start, w.end - w.start, confidence=w.probability)
                all_words.append(word_obj)
                seg_text_parts.append(text)

            if seg_text_parts:
                start_ts = seconds_to_srt_timestamp(seg.start)
                end_ts = seconds_to_srt_timestamp(seg.end)
                srt_blocks.append((idx, start_ts, end_ts, " ".join(seg_text_parts)))
            
            pbar.update(round(seg.end - pbar.n, 2))

    if not all_words:
        print("No words found in transcription.")
        sys.exit(1)

    # Write Premiere JSON
    segments = words_to_segments(all_words, speaker_id, opts.premiere_language)
    
    data = {
        "language": opts.premiere_language,
        "segments": segments,
        "speakers": [{"id": speaker_id, "name": "Speaker 1"}],
    }
    
    # Try writing JSON first to determine the target directory
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        final_json_path = json_path
        final_txt_path = txt_path
    except PermissionError:
        final_json_path = os.path.basename(json_path)
        final_txt_path = os.path.basename(txt_path)
        print(f"Warning: Permission denied for {json_path}. Saving BOTH files to current directory.")
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

    print(f"Wrote {final_json_path}  ({len(all_words)} words, {len(segments)} segments)")

    # Prepare client TXT lines in SRT format
    txt_lines = []
    for idx, start_ts, end_ts, text in srt_blocks:
        txt_lines.append(str(idx))
        txt_lines.append(f"{start_ts} --> {end_ts}")
        txt_lines.append(text)
        txt_lines.append("")
        
    # Write client TXT to the SAME final location
    with open(final_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines).rstrip() + "\n\n")
    print(f"Wrote {final_txt_path}")


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

def srt_time_to_seconds(srt_time):
    """Converts SRT/Transcript timestamp (HH:MM:SS[:/.,]mmm) to float seconds."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[:,\.](\d{2,3})", srt_time)
    if not match:
        return 0.0
    h, m, s, ms_str = match.groups()
    h, m, s = map(int, [h, m, s])
    ms_val = int(ms_str)
    if len(ms_str) == 2:
        ms_val *= 10
    return h * 3600 + m * 60 + s + ms_val / 1000.0


def parse_srt(srt_file_path):
    """Parses SRT or timestamped TXT file and returns a list of dictionaries with start, end, and text."""
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}[:,\.]\d{2,3})\s*(?:-->|-)\s*(\d{2}:\d{2}:\d{2}[:,\.]\d{2,3})")
    
    entries = []
    parts = ts_pattern.split(content)
    
    for i in range(1, len(parts), 3):
        start = srt_time_to_seconds(parts[i])
        end = srt_time_to_seconds(parts[i+1])
        text_raw = parts[i+2]
        
        lines = [line.strip() for line in text_raw.split('\n') if line.strip()]
        if lines and lines[-1].isdigit():
            lines.pop()
            
        text = " ".join(lines).strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})
    return entries


def convert_to_premiere_json(srt_entries):
    """Converts parsed SRT entries to Premiere Pro JSON structure."""
    speaker_id = str(uuid.uuid4())
    all_words = []
    MIN_WORD_DURATION = 0.08
    
    for entry in srt_entries:
        words_in_block = entry['text'].split()
        if not words_in_block: continue
        block_duration = entry['end'] - entry['start']
        word_duration = max(block_duration / len(words_in_block), MIN_WORD_DURATION)
        
        for i, word_text in enumerate(words_in_block):
            word_start = entry['start'] + (i * word_duration)
            all_words.append(make_word_obj(word_text, word_start, word_duration))

    return {
        "language": "en-us",
        "segments": words_to_segments(all_words, speaker_id, "en-us"),
        "speakers": [{"id": speaker_id, "name": "Speaker 1"}]
    }


def cmd_to_premiere(args):
    import argparse
    parser = argparse.ArgumentParser(prog="captions.py to-premiere")
    parser.add_argument("transcript_file", help="Path to edited TXT or SRT file")
    parser.add_argument("-o", "--output", help="Optional output path for the JSON (defaults to same folder as input)")
    opts = parser.parse_args(args)

    input_path = fix_icloud_path(opts.transcript_file)
    output_path = opts.output if opts.output else os.path.splitext(input_path)[0] + ".json"

    print(f"Converting {input_path} → {output_path} …")
    srt_entries = parse_srt(input_path)
    if not srt_entries:
        print(f"Error: No timestamps found in {input_path}. Check the format.")
        sys.exit(1)
        
    premiere_json = convert_to_premiere_json(srt_entries)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(premiere_json, f, separators=(",", ":"))
        final_path = output_path
    except PermissionError:
        final_path = os.path.basename(output_path)
        print(f"Warning: Permission denied for {output_path}. Saving to current directory.")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(premiere_json, f, separators=(",", ":"))
            
    print(f"Done. Saved to {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "transcribe": cmd_transcribe,
    "apply-edits": cmd_apply_edits,
    "to-premiere": cmd_to_premiere,
    "from-srt": cmd_to_premiere,  # Legacy alias
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands:")
        print("  transcribe  <audio.mp3> [--language de] [--model large-v3-turbo]")
        print("  to-premiere <edited_transcript.txt>")
        print("  apply-edits <audio.json> <edited_plain.txt>")
        sys.exit(1)

    COMMANDS[sys.argv[1]](sys.argv[2:])
