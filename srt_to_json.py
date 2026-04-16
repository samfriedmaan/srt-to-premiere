import json
import re
import uuid

def srt_time_to_seconds(srt_time):
    """Converts SRT/Transcript timestamp (HH:MM:SS[:/.,]mmm) to float seconds."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[:,\.](\d{2,3})", srt_time)
    if not match:
        return 0.0
    h, m, s, ms_str = match.groups()
    h, m, s = map(int, [h, m, s])
    ms_val = int(ms_str)
    # If 2 digits, treat as centiseconds (1/100th), if 3 treat as milliseconds (1/1000th)
    if len(ms_str) == 2:
        ms_val *= 10
    return h * 3600 + m * 60 + s + ms_val / 1000.0

def parse_srt(srt_file_path):
    """Parses SRT or timestamped TXT file and returns a list of dictionaries with start, end, and text."""
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex for timestamps: Handles both "00:00:00,000 --> 00:00:00,000" and "00:00:00:00 - 00:00:00:00"
    # Supports separators: , . : and delimiters: --> -
    ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}[:,\.]\d{2,3})\s*(?:-->|-)\s*(\d{2}:\d{2}:\d{2}[:,\.]\d{2,3})")
    
    entries = []
    # Split by timestamp to get blocks. parts will be [before, start1, end1, text1, start2, end2, text2, ...]
    parts = ts_pattern.split(content)
    
    for i in range(1, len(parts), 3):
        start = srt_time_to_seconds(parts[i])
        end = srt_time_to_seconds(parts[i+1])
        # Next part is the text, up to the next timestamp (or end of file)
        text_raw = parts[i+2]
        
        # Clean up text lines, removing blank lines and trailing index numbers
        lines = [line.strip() for line in text_raw.split('\n') if line.strip()]
        if lines and lines[-1].isdigit():
            lines.pop()
            
        text = " ".join(lines).strip()
        if text:
            entries.append({
                "start": start,
                "end": end,
                "text": text
            })
    return entries

def convert_to_premiere_json(srt_entries):
    """Converts parsed SRT entries to Premiere Pro JSON structure."""
    speaker_id = str(uuid.uuid4())
    segments = []
    all_words = []
    
    # Minimum duration for a single word to be readable in Premiere (approx 2 frames at 24fps)
    MIN_WORD_DURATION = 0.08
    
    for entry in srt_entries:
        words_in_block = entry['text'].split()
        if not words_in_block:
            continue
            
        block_duration = entry['end'] - entry['start']
        # Distribute time, but ensure no word is shorter than MIN_WORD_DURATION
        word_duration = max(block_duration / len(words_in_block), MIN_WORD_DURATION)
        
        for i, word_text in enumerate(words_in_block):
            word_start = entry['start'] + (i * word_duration)
            is_eos = word_text.endswith(('.', '?', '!', ','))
            
            all_words.append({
                "confidence": 1.0,
                "duration": round(word_duration, 3),
                "eos": is_eos,
                "start": round(word_start, 3),
                "tags": [],
                "text": word_text,
                "type": "word"
            })

    # Group words into segments (max 15 words or EOS)
    current_segment_words = []
    segment_start = 0
    
    for word in all_words:
        if not current_segment_words:
            segment_start = word['start']
        current_segment_words.append(word)
        if word['eos'] or len(current_segment_words) >= 15:
            segment_end = word['start'] + word['duration']
            segments.append({
                "duration": round(segment_end - segment_start, 3),
                "language": "en-us",
                "speaker": speaker_id,
                "start": round(segment_start, 3),
                "words": current_segment_words
            })
            current_segment_words = []

    if current_segment_words:
        segment_end = current_segment_words[-1]['start'] + current_segment_words[-1]['duration']
        segments.append({
            "duration": round(segment_end - segment_start, 3),
            "language": "en-us",
            "speaker": speaker_id,
            "start": round(segment_start, 3),
            "words": current_segment_words
        })

    return {
        "language": "en-us",
        "segments": segments,
        "speakers": [{"id": speaker_id, "name": "Speaker 1"}]
    }

if __name__ == "__main__":
    import sys
    import os

    input_srt = "video.srt"
    if len(sys.argv) > 1:
        input_srt = sys.argv[1]
        if "comappleCloudDocs" in input_srt:
            input_srt = input_srt.strip("'").strip('"').replace("comappleCloudDocs", "com~apple~CloudDocs")

    output_json = os.path.splitext(input_srt)[0] + ".json"
    srt_entries = parse_srt(input_srt)
    if not srt_entries:
        print(f"Error: No timestamps found in {input_srt}. Check the format.")
        sys.exit(1)
        
    premiere_json = convert_to_premiere_json(srt_entries)
        
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(premiere_json, f, indent=None, separators=(',', ':'))
        print(f"Done. Saved to {output_json}")
    except PermissionError:
        fallback_json = os.path.basename(output_json)
        print(f"Warning: Permission denied for {output_json}. Saving to current directory.")
        with open(fallback_json, 'w', encoding='utf-8') as f:
            json.dump(premiere_json, f, indent=None, separators=(',', ':'))
        print(f"Done. Saved to {fallback_json}")
