# Anything to Premiere Transcript

A Python converter for turning existing timestamped transcripts into Premiere Pro's word-timed JSON format. The main command is `to-premiere`; supported sources include SRT, frame-based timeline caption exports, WebVTT-style files, and compatible JSON transcript/cue files.

## Quick start

```bash
python3 captions.py to-premiere transcript.srt -o transcript.json
```

Import the resulting JSON as a transcript in Premiere. Conversion uses only the Python standard library; no model download or transcription dependencies are needed.

### Word-level timing

- **One word per SRT cue:** each word keeps its supplied start and end time, including short words and gaps between words. Numeric speech such as `100` is retained.
- **Zero-duration words:** words with identical start and end timestamps are retained in source order, including several words at the same timestamp. The command reports their count. These words have no measured duration in the source; the converter does not invent one. Adobe's [transcript schema](https://github.com/AdobeDocs/uxp-premiere-pro-samples/blob/main/sample-panels/premiere-api/assets/transcript_format_spec.json) permits zero-duration words and segments.
- **Several words per cue:** the cue's duration is evenly distributed across its words. These word times are estimates.
- **Premiere transcript JSON:** existing word timings, confidence, tags, and speaker metadata are retained, subject to the selected overlap policy.

The converter does not recover precise word timings from plain text or sentence-level timestamps. Accurate timing requires a source that already contains word timestamps.

## `to-premiere`

```bash
python3 captions.py to-premiere INPUT [-o OUTPUT.json]
```

Supported inputs:

- Standard SRT timestamps such as `00:00:01,200 --> 00:00:03,400`
- WebVTT-style millisecond timestamps
- Frame-based timeline timestamps such as `00:00:01:12 - 00:00:03:05`
- Premiere transcript JSON (`language`, `segments`, `speakers`, and word arrays)
- Simple JSON arrays/objects containing `start`, `end`, and `text`

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

Timeline exports may include `Speaker 1` lines. Those lines become speaker metadata rather than transcript text. Empty cues and reversed time ranges are ignored. With the default `sequential` policy, an overlap is resolved by ending the earlier cue at the later cue's start; if two positive-duration cues start together, the later cue wins. Zero-duration cues are retained without clipping other cues. Use `--overlap-policy preserve` to keep overlapping positive-duration timings unchanged. The command reports clips, dropped cues, and retained zero-duration words so the normalization is visible.

## Legacy audio transcription and editing

The original `transcribe`, `apply-edits`, and `from-srt` commands remain available. `from-srt` is an alias for `to-premiere`.

### `transcribe`

Transcribe audio using the Swiss German fine-tuned Whisper model. Install its optional dependencies first:

```bash
pip install faster-whisper huggingface_hub tqdm
python3 captions.py transcribe interview.mp3
```

Generates `interview.json` with word timing and `interview_client.txt` with SRT-style text for editing. Convert an edited timestamped TXT file using `to-premiere`; this estimates timing within each edited cue.

| Option | Default | Description |
|---|---|---|
| `--model` | `nebi/whisper-large-v3-turbo-swiss-german-ct2-int8` | Model name or HuggingFace repo ID |
| `--language` | `de` | Whisper language code |
| `--compute-type` | `int8` | Quantization type (`int8`, `float16`, `float32`) |
| `--prompt` | `Schweizerdeutsch. Transkription auf Hochdeutsch.` | Initial prompt to prime the model |
| `-o` / `--output` | Same folder as audio | Output path for JSON (TXT is placed alongside it) |

The model downloads automatically on first use and is cached in `~/.cache/huggingface/hub/models--nebi--whisper-large-v3-turbo-swiss-german-ct2-int8/`. To pre-download:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nebi/whisper-large-v3-turbo-swiss-german-ct2-int8')"
```

### `apply-edits`

Fuzzy-aligns an edited plain-text file with the original JSON to preserve precise word timings.

```bash
python3 captions.py apply-edits <original.json> <edited.txt>
```

---

`to-premiere` and `apply-edits` need no extra packages. The full `uv` project specifies Python 3.14+ for the transcription dependency environment.

## Tests

Run the standard-library regression tests with:

```bash
python3 -m unittest -v test_transcript_io.py
```
