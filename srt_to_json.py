import json
import re
import uuid

def srt_time_to_seconds(srt_time):
    """Converts SRT timestamp (HH:MM:SS,mmm) to float seconds."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", srt_time)
    if not match:
        return 0.0
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0

def parse_srt(srt_file_path):
    """Parses SRT file and returns a list of dictionaries with start, end, and text."""
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline or single newline if it's a sequence number
    blocks = re.split(r'\n\s*\n', content.strip())
    
    entries = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            # Line 1: index (ignore)
            # Line 2: timestamps
            times = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            if len(times) == 2:
                start = srt_time_to_seconds(times[0])
                end = srt_time_to_seconds(times[1])
                # Line 3+: text
                text = " ".join(lines[2:]).strip()
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
    
    # In the example, it seems segments are grouped. 
    # For a simple SRT, each caption block can be treated as a word or group of words.
    # The example has VERY granular word-level timing. 
    # Since SRT only provides block-level timing, we will treat each block as a word 
    # if it's short, or split it if it's long.
    
    all_words = []
    for entry in srt_entries:
        # Split text into words and distribute duration equally
        words_in_block = entry['text'].split()
        if not words_in_block:
            continue
            
        block_duration = entry['end'] - entry['start']
        word_duration = block_duration / len(words_in_block)
        
        for i, word_text in enumerate(words_in_block):
            word_start = entry['start'] + (i * word_duration)
            # Ensure text matches style (e.g., adding punctuation back)
            # In SRT, the block usually has punctuation. 
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

    # Group words into segments (e.g., max 15 words per segment or 5 seconds)
    current_segment_words = []
    segment_start = 0
    
    for i, word in enumerate(all_words):
        if not current_segment_words:
            segment_start = word['start']
        
        current_segment_words.append(word)
        
        # Condition to close segment: EOS or max words
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

    # Final segment if remaining
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
        "speakers": [
            {
                "id": speaker_id,
                "name": "Speaker 1"
            }
        ]
    }

if __name__ == "__main__":
    import sys
    import os

    input_srt = "video.srt"
    
    if len(sys.argv) > 1:
        input_srt = sys.argv[1]
        
        # macOS drag-and-drop bug workaround for iCloud paths
        if "comappleCloudDocs" in input_srt:
            input_srt = input_srt.replace("comappleCloudDocs", "com~apple~CloudDocs")

    output_json = os.path.splitext(input_srt)[0] + ".json"
        
    print(f"Converting {input_srt} to {output_json}...")
    srt_entries = parse_srt(input_srt)
    premiere_json = convert_to_premiere_json(srt_entries)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(premiere_json, f, indent=None, separators=(',', ':'))
    
    print("Done.")
