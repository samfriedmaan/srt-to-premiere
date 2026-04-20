# SRT to Premiere Pro Unified Tooling

A Python toolset for transcribing Swiss German audio and generating Premiere Pro-ready JSON transcripts. Features a client-review workflow for correcting errors before final import.

## Workflow

1. **Transcribe**: Convert audio to a Premiere JSON and a human-readable SRT-style transcript.
   ```bash
   python3 captions.py transcribe interview.mp3
   ```
   *Generates `interview.json` and `interview_client.txt`.*

2. **Edit**: Send `interview_client.txt` to the client. They correct errors in the SRT-style text.

3. **Convert**: Turn the corrected transcript into a Premiere-ready JSON.
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

Convert an edited `.txt` or `.srt` file into a Premiere Pro JSON.

```bash
python3 captions.py to-premiere <transcript.txt> [-o output.json]
```

No extra dependencies needed — standard library only.

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
