# SRT to Premiere Pro Unified Tooling

A Python toolset for transcribing Swiss German audio and generating Premiere Pro-ready JSON transcripts. It also converts existing SRT files, frame-based timeline caption exports, WebVTT-style files, and compatible JSON transcript/cue files.

## Workflow

1. **Transcribe**: Convert audio to a Premiere JSON and a human-readable SRT-style transcript.
   ```bash
   python3 captions.py transcribe interview.mp3
   ```
   *Generates `interview.json` and `interview_client.txt`.*

2. **Edit**: Send `interview_client.txt` to the client. They correct errors in the SRT-style text.

3. **Convert**: Turn the corrected transcript into a Premiere-ready JSON. The converter accepts SRT, VTT, frame-based timeline text, and JSON.
   ```bash
   python3 captions.py to-premiere interview_client_edited.txt
   ```
   *Generates `interview_client_edited.json` ready for import.*

---

## Setup (new machine)

### 1. Install Python dependencies

```bash
pip install faster-whisper huggingface_hub tqdm
```

### 2. Download the Swiss German model (first transcription only)

The fine-tuned Swiss German model downloads automatically on first use — no manual step required. It's ~800 MB and is cached locally so subsequent runs are instant.

**Cache location:** `~/.cache/huggingface/hub/models--nebi--whisper-large-v3-turbo-swiss-german-ct2-int8/`

**To pre-download explicitly** (e.g. before going offline):
```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nebi/whisper-large-v3-turbo-swiss-german-ct2-int8')"
```

**To copy the model from an existing machine** (avoids re-downloading):
```bash
# On the source machine, find the cache:
ls ~/.cache/huggingface/hub/models--nebi--whisper-large-v3-turbo-swiss-german-ct2-int8/

# Copy the entire folder to the same path on the new machine.
rsync -av ~/.cache/huggingface/hub/models--nebi--whisper-large-v3-turbo-swiss-german-ct2-int8/ \
  user@other-machine:~/.cache/huggingface/hub/models--nebi--whisper-large-v3-turbo-swiss-german-ct2-int8/
```

### 3. Verify

```bash
python3 captions.py transcribe some_audio.mp3
```

---

## Commands

### `transcribe`

Transcribe audio using the Swiss German fine-tuned Whisper model.

```bash
python3 captions.py transcribe <audio.mp3> [options]
```

| Option | Default | Description |
|---|---|---|
| `--model` | `nebi/whisper-large-v3-turbo-swiss-german-ct2-int8` | Model name or HuggingFace repo ID |
| `--language` | `de` | Whisper language code |
| `--compute-type` | `int8` | Quantization type (`int8`, `float16`, `float32`) |
| `--prompt` | `Schweizerdeutsch. Transkription auf Hochdeutsch.` | Initial prompt to prime the model |
| `-o` / `--output` | Same folder as audio | Output path for JSON (TXT is placed alongside it) |

**About the model:** `nebi/whisper-large-v3-turbo-swiss-german-ct2-int8` is a CTranslate2-format version of `Flurin17/whisper-large-v3-turbo-swiss-german`, fine-tuned on the SwissDial-ZH and STT4SG-350 datasets (343+ hours of Swiss German speech from ZHAW/ETH Zurich). It outputs Standard German text from Swiss German speech. The `--language de` flag is correct — passing `gsw` is not supported by Whisper.

**Performance on Apple Silicon (M1/M4):** faster-whisper uses CPU only (no MPS/Metal support). Expect roughly real-time speed (~1–1.5× audio duration) on M-series chips.

### `to-premiere`

Convert a timestamped input into Premiere Pro's word-timed transcript JSON. No extra dependencies are needed; this command uses the Python standard library only.

```bash
python3 captions.py to-premiere INPUT [-o OUTPUT.json]
```

Supported inputs:

- Standard SRT timestamps such as `00:00:01,200 --> 00:00:03,400`
- WebVTT-style millisecond timestamps
- Frame-based timeline timestamps such as `00:00:01:12 - 00:00:03:05`
- Premiere transcript JSON (`language`, `segments`, `speakers`, and word arrays)
- Simple JSON arrays/objects containing `start`, `end`, and `text`

For cue-level inputs, the converter evenly distributes each cue's duration across its words. This creates the word-level timing Premiere requires, but it is necessarily an estimate unless the source already has word timestamps.

Options:

| Option | Default | Description |
|---|---|---|
| `--premiere-language` | Input JSON language or `en-us` | Premiere language metadata, e.g. `de-de` |
| `--fps` | Auto / 30 | Frame rate for `HH:MM:SS:FF` inputs; supports 24, 25, 30, 50, 60, and other numeric rates |
| `--overlap-policy` | `sequential` | Clip earlier cues when timings overlap; use `preserve` to keep overlaps |
| `--speaker-name` | `Speaker 1` | Speaker name for timestamped text without speaker labels |

Examples:

```bash
# Standard SRT
python3 captions.py to-premiere captions.srt -o transcript.json --premiere-language de-de

# Premiere/timeline text with HH:MM:SS:FF ranges
python3 captions.py to-premiere timeline.txt -o transcript.json --fps 30 --premiere-language de-de

# European 25 fps timeline
python3 captions.py to-premiere timeline.txt -o transcript.json --fps 25 --premiere-language de-de

# Existing Premiere JSON; preserve its language and speaker metadata
python3 captions.py to-premiere existing-transcript.json -o normalized.json
```

Timeline exports may include `Speaker 1` lines. Those lines become speaker metadata rather than transcript text. Empty cues are ignored. With the default `sequential` policy, an overlap is resolved by ending the earlier cue at the later cue's start; if two cues start together, the later cue wins. The command reports clips and dropped cues so the normalization is visible.

### `apply-edits`

Fuzzy-aligns an edited plain-text file with the original JSON to preserve precise word timings.

```bash
python3 captions.py apply-edits <original.json> <edited.txt>
```

---

## Requirements

```
faster-whisper
huggingface_hub
tqdm
```

Install: `pip install faster-whisper huggingface_hub tqdm`

Python 3.8+ required. `to-premiere` and `apply-edits` need no extra packages.

## Tests

Run the standard-library regression tests with:

```bash
python3 -m unittest -v test_transcript_io.py
```
