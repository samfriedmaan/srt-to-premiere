"""
captions.py — unified caption tool

Commands:
  transcribe  <audio.mp3> [--language de] [--model large-v3-turbo]
  to-premiere <transcript.srt|timeline.txt|transcript.json>
  apply-edits <audio.json> <edited_plain.txt>
"""

import difflib
import json
import os
import re
import sys
import uuid
from pathlib import Path

from transcript_io import Cue, convert_cues, convert_file, parse_srt

# Dynamically locate and load CUDA/cuDNN DLLs on Windows (from nvidia-* pip packages)
if os.name == "nt":
    import site
    try:
        site_dirs = sys.path + site.getsitepackages() + [site.getusersitepackages()]
    except Exception:
        site_dirs = sys.path
    seen_dirs = set()
    for s_dir in site_dirs:
        if not s_dir or s_dir in seen_dirs:
            continue
        seen_dirs.add(s_dir)
        for pkg in ["cublas", "cudnn"]:
            cuda_path = os.path.join(s_dir, "nvidia", pkg, "bin")
            if os.path.exists(cuda_path):
                try:
                    os.add_dll_directory(cuda_path)
                except Exception:
                    pass


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
    parser.add_argument("--model", default="nebi/whisper-large-v3-turbo-swiss-german-ct2-int8",
                        help="Whisper model name or HuggingFace repo ID (default: Swiss German fine-tuned ct2 model)")
    parser.add_argument("--compute-type", default="int8", dest="compute_type",
                        help="Whisper compute type (default: int8, matches Swiss German ct2 model quantization)")
    parser.add_argument("--device", default="auto", help="Device to use: cpu, cuda, or auto (default: auto)")
    parser.add_argument("--prompt", default="Schweizerdeutsch. Transkription auf Hochdeutsch.",
                        help="Initial prompt to prime the model (default: Swiss German context)")
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

    import platform
    system_os = platform.system()
    machine_arch = platform.machine()
    selected_device = opts.device

    if opts.device == "auto":
        if system_os == "Darwin":
            if machine_arch == "arm64":
                print("[System Info] Apple Silicon M-series detected. Running optimized CPU mode (using Apple Accelerate/ARM64).")
            else:
                print("[System Info] macOS detected. Running on CPU.")
            selected_device = "cpu"
        else:
            try:
                import ctranslate2
                cuda_devices = ctranslate2.get_cuda_device_count()
            except Exception:
                cuda_devices = 0

            if cuda_devices > 0:
                print(f"[System Info] NVIDIA GPU detected ({cuda_devices} device(s)). Attempting CUDA execution...")
                selected_device = "cuda"
            else:
                print("[System Info] No NVIDIA GPU/CUDA support detected. Running on CPU.")
                selected_device = "cpu"
    else:
        print(f"[System Info] Explicitly using device: {opts.device}")

    print(f"Loading model '{opts.model}' (compute_type={opts.compute_type}, device={selected_device}) …")
    model = WhisperModel(opts.model, device=selected_device, compute_type=opts.compute_type)

    print(f"Transcribing {audio_path}  (language={opts.language}, prompt={opts.prompt!r}) …")

    def run_transcription(whisper_model):
        segments_iter, _info = whisper_model.transcribe(
            audio_path,
            language=opts.language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=opts.prompt,
            condition_on_previous_text=False,
        )

        speaker_id = str(uuid.uuid4())
        all_words = []

        with tqdm(total=round(_info.duration, 2), unit="sec", desc="Transcribing") as pbar:
            for seg in segments_iter:
                raw_words = list(seg.words)
                if not raw_words:
                    continue

                for w in raw_words:
                    text = w.word.strip()
                    if not text:
                        continue
                    word_obj = make_word_obj(text, w.start, w.end - w.start, confidence=w.probability)
                    all_words.append(word_obj)

                pbar.update(round(seg.end - pbar.n, 2))
        return all_words, speaker_id

    try:
        all_words, speaker_id = run_transcription(model)
    except RuntimeError as e:
        err_msg = str(e).lower()
        if selected_device != "cpu" and any(x in err_msg for x in ["cublas", "cudnn", "cuda", "library not found"]):
            print(f"\nCUDA/GPU transcription failed: {e}")
            print("Attempting automatic fallback to CPU mode...")
            cpu_compute_type = opts.compute_type
            if cpu_compute_type == "float16":
                cpu_compute_type = "int8"
            
            print(f"Reloading model '{opts.model}' on CPU (compute_type={cpu_compute_type}) …")
            model = WhisperModel(opts.model, device="cpu", compute_type=cpu_compute_type)
            all_words, speaker_id = run_transcription(model)
        else:
            raise

    # Build sentence-level SRT blocks from word timestamps.
    # Split at sentence-ending punctuation; cap at MAX_WORDS_PER_BLOCK as a safety valve.
    MAX_WORDS_PER_BLOCK = 30
    srt_blocks = []
    bucket = []

    def flush_bucket(bucket, idx):
        if not bucket:
            return idx
        start_ts = seconds_to_srt_timestamp(bucket[0]["start"])
        end_ts = seconds_to_srt_timestamp(bucket[-1]["start"] + bucket[-1]["duration"])
        srt_blocks.append((idx, start_ts, end_ts, " ".join(w["text"] for w in bucket)))
        return idx + 1

    srt_idx = 1
    for word_obj in all_words:
        bucket.append(word_obj)
        is_sentence_end = word_obj["text"].rstrip().endswith(('.', '?', '!'))
        if is_sentence_end or len(bucket) >= MAX_WORDS_PER_BLOCK:
            srt_idx = flush_bucket(bucket, srt_idx)
            bucket = []

    srt_idx = flush_bucket(bucket, srt_idx)  # flush any trailing words

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
# to-premiere / from-srt
# ---------------------------------------------------------------------------

def convert_to_premiere_json(srt_entries, language="en-us", overlap_policy="sequential"):
    """Backward-compatible conversion for callers that already parsed cues."""
    cues = [
        Cue(item["start"], item["end"], item["text"], item.get("speaker"), index)
        for index, item in enumerate(srt_entries)
    ]
    return convert_cues(cues, language=language, overlap_policy=overlap_policy)


def cmd_to_premiere(args):
    import argparse

    parser = argparse.ArgumentParser(
        prog="captions.py to-premiere",
        description="Convert SRT, VTT, frame-based timeline text, or JSON to Premiere transcript JSON.",
    )
    parser.add_argument("transcript_file", help="Input SRT/VTT/TXT/JSON file")
    parser.add_argument("-o", "--output", help="Output JSON path or directory")
    parser.add_argument(
        "--premiere-language",
        default=None,
        help="Language metadata, e.g. de-de. JSON inputs keep their existing language by default.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate for HH:MM:SS:FF inputs (including European 25 fps); auto-detects 60 when needed, otherwise defaults to 30.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("sequential", "preserve"),
        default="sequential",
        help="Resolve overlaps by clipping earlier text (default), or preserve them.",
    )
    parser.add_argument("--speaker-name", help="Speaker name for timestamped text without speaker labels")
    opts = parser.parse_args(args)

    input_path = Path(fix_icloud_path(opts.transcript_file))
    if opts.output:
        requested_output = Path(fix_icloud_path(opts.output))
        is_directory = requested_output.is_dir() or opts.output.endswith(os.sep)
        output_path = requested_output / f"{input_path.stem}.json" if is_directory else requested_output
    else:
        output_path = input_path.with_suffix(".json")

    print(f"Converting {input_path} → {output_path} …")
    try:
        premiere_json, report = convert_file(
            input_path,
            language=opts.premiere_language,
            fps=opts.fps,
            overlap_policy=opts.overlap_policy,
            speaker_name=opts.speaker_name,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(
            json.dumps(premiere_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        final_path = output_path
    except PermissionError:
        final_path = Path(output_path.name)
        print(f"Warning: Permission denied for {output_path}. Saving to current directory as {final_path}")
        final_path.write_text(
            json.dumps(premiere_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    word_count = sum(len(segment["words"]) for segment in premiere_json["segments"])
    details = [
        f"format={report.format_name}",
        f"{len(premiere_json['segments'])} segments",
        f"{word_count} words",
    ]
    if report.fps:
        details.append(f"fps={report.fps:g}")
    if report.overlaps_clipped:
        details.append(f"{report.overlaps_clipped} overlaps clipped")
    if report.entries_dropped:
        details.append(f"{report.entries_dropped} entries dropped")
    print(f"Done. Saved to {final_path} ({', '.join(details)})")


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
        print("  to-premiere <transcript.srt|timeline.txt|transcript.json>")
        print("  apply-edits <audio.json> <edited_plain.txt>")
        sys.exit(1)

    COMMANDS[sys.argv[1]](sys.argv[2:])
