# SRT to Premiere Pro Unified Tooling

A powerful Python toolset for transcribing audio and generating Premiere Pro-ready JSON transcripts. It features a granular workflow designed for client review and high-precision timing.

## Workflow

The tool supports a seamless 3-step process for perfect captions:

1. **Transcribe**: Convert audio to a Premiere JSON and a human-readable transcript.
   ```bash
   python3 captions.py transcribe interview.mp3
   ```
   *Generates `interview.json` and `interview_client.txt`.*

2. **Edit**: Give the `interview_client.txt` to your client. They can edit text and fix typos using a simple SRT-style format.

3. **Convert**: Turn the edited transcript back into a perfect Premiere JSON.
   ```bash
   python3 captions.py to-premiere interview_client_edited.txt
   ```
   *Generates `interview_client_edited.json` ready for import.*

---

## Features

- **Granular Transcription**: Uses `faster-whisper` for high-quality, segment-based transcription with precise timestamps.
- **Flexible Parser**: The `to-premiere` command is robust and handles various transcript formats:
    - Standard SRT: `00:00:00,000 --> 00:00:01,000`
    - Alternative: `00:00:00:00 - 00:00:01:00`
    - Sparse formatting: Handles extra newlines and numeric indices automatically.
- **Timing Improvements**: Automatically enforces a **0.08s minimum word duration** to prevent "flickering" or too-tight timings in Premiere.
- **macOS Friendly**: Built-in support for terminal drag-and-drop fixes for iCloud paths.

## Commands

### `transcribe`
Transcribe audio files using OpenAI's Whisper models.
- **Usage**: `python3 captions.py transcribe <audio> [--language de] [--model large-v3-turbo]`
- **Requirements**: `pip install faster-whisper tqdm`

### `to-premiere`
Convert an edited `.txt` or `.srt` file into a Premiere Pro JSON transcript.
- **Usage**: `python3 captions.py to-premiere <transcript.txt>`
- **Note**: No dependencies required for this command (standard library only).

### `apply-edits`
*Legacy/Alternative:* Fuzzy-alignes an edited plain-text file with an original JSON.
- **Usage**: `python3 captions.py apply-edits <original.json> <edited.txt>`

## Requirements
For transcription:
- Python 3.8+
- `faster-whisper`
- `tqdm`

For conversion (`to-premiere`):
- Python 3 Standard Library only.
