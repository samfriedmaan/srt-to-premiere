# Anything to Premiere Transcript

A CLI for creating Premiere Pro's word-timed transcript JSON from audio, video, or existing timestamped transcripts. `transcribe` runs local speech recognition; `to-premiere` converts SRT, frame-based timeline caption exports, WebVTT-style files, and compatible JSON transcript/cue files.

## Quick start

```bash
# Install the transcription dependencies once (Python 3.14+).
uv sync

# Audio or video → Premiere transcript JSON, next to the source file.
uv run captions.py transcribe "/path/to/interview.mp4"

# Existing timestamped transcript → Premiere transcript JSON.
python3 captions.py to-premiere transcript.srt -o transcript.json
```

Import the resulting JSON as a transcript in Premiere. `to-premiere` uses only the Python standard library; no model download or transcription dependencies are needed for conversion.

## `transcribe`

```bash
uv run captions.py transcribe INPUT_AUDIO_OR_VIDEO [-o OUTPUT.json]
```

The default model is **Whisper large-v3-turbo**. Language detection is automatic, and the detected language sets Premiere's metadata. The command reads the audio stream from video containers and writes Premiere JSON directly from the engine's word timestamps. It does not pass the transcript through sentence-level SRT or pad short words. All processing uses local engines.

### Reusing models

Before downloading, the command searches known Buzz, MacWhisper, Whisper, Hugging Face, and project cache locations for the **requested model**. A cached smaller model is never silently substituted for large-v3-turbo. Other applications' caches are read only; Buzz and MacWhisper do not need to be running.

On **Apple Silicon**, cached Buzz/original Whisper `.pt` weights run on the **Apple GPU through MLX** automatically. The first run converts the local weights to FP16 MLX format and saves a separate copy in the project cache (about 1.5 GiB for turbo). This conversion needs no model download and happens once. Later runs reuse it directly. Uninstalling MacWhisper does not affect this workflow. MLX uses greedy decoding and the official Whisper alignment heads for word timestamps; text can differ from beam-search decoding.

| Cached format | Engine | Requirements |
|---|---|---|
| Buzz/original Whisper `.pt` on Apple Silicon | MLX / Metal GPU | Included in `uv sync`; one-time local conversion |
| MLX model directory | MLX / Metal GPU | Included in `uv sync` on Apple Silicon |
| Buzz/original Whisper `.pt` on other platforms, or `--backend whisper` | Original Whisper / PyTorch | Included in `uv sync` |
| Buzz/Hugging Face CTranslate2 directory | Faster Whisper | Included in `uv sync` |
| Buzz/MacWhisper whisper.cpp `.bin`, including quantized models | whisper.cpp | `whisper-cli` and FFmpeg on PATH; on macOS: `brew install whisper-cpp ffmpeg` |
| MacWhisper WhisperKit or whisper.cpp model | MacWhisper's `mw` CLI | Install from MacWhisper Settings → Advanced → Command-Line Tool; word-timed JSON export requires MacWhisper Pro |

MacWhisper's [official CLI](https://docs.macwhisper.com/article/57-macwhisper-command-line-tool) runs its existing engines and models, including WhisperKit/Core ML. Compiled WhisperKit models are not interchangeable with PyTorch or CTranslate2 models. If `mw` is available, installed local Whisper models are queried through it; cloud model IDs are excluded. If an engine returns only segment timestamps, transcription fails with an explanation instead of presenting estimated timing as measured word timing.

On Apple Silicon, automatic selection prefers MLX, then whisper.cpp, MacWhisper's optional `mw`, Faster Whisper, and original Whisper. On other platforms it prefers Faster Whisper, whisper.cpp, and original Whisper. CUDA is used automatically when available; PyTorch/CTranslate2 engines fall back to CPU on CUDA failures. `--backend whisper` keeps the original PyTorch engine, which uses CPU on Macs for word alignment. `models` lists the cache owner, weight format, path, and default engine/device without loading model weights.

If no compatible cached model is found, Apple Silicon downloads original Whisper weights once and converts them locally for MLX; other platforms download Faster Whisper's format. Later runs reuse the cache. If cached `.bin` weights are found but `whisper-cli` is missing, the command explains how to install the engine instead of downloading duplicate weights. An explicit `--backend` override restricts cache matching to that engine and can require downloading a different format. `--offline` permits local conversion but prevents model downloads.

Inspect discovery or use a cache outside the standard locations:

```bash
uv run captions.py models
uv run captions.py models --model large-v3-turbo
uv run captions.py transcribe video.mov --model-dir "/external/Buzz/models"
uv run captions.py transcribe video.mov --model "/external/custom-model.pt"
```

Discovery honors `BUZZ_MODEL_ROOT`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, and `XDG_CACHE_HOME`. Buzz's macOS cache and both sandboxed and nonsandboxed MacWhisper model locations are checked; Windows and Linux Buzz caches are also supported. The download cache defaults to `~/.cache/srt-to-premiere/models` (or the corresponding `XDG_CACHE_HOME` path).

### Options

| Option | Default | Description |
|---|---|---|
| `--model` | `large-v3-turbo` | Model name, CTranslate2 Hugging Face repository, local `.pt`/`.bin` file, or MLX/CTranslate2 directory |
| `--backend` | `auto` | `mlx`, `faster-whisper`, `whisper`, `whisper-cpp`, or `macwhisper` |
| `--language` | `auto` | Source language code, e.g. `en` or `de` |
| `--premiere-language` | Detected language | Override metadata, e.g. `en-gb` or `pt-br`; unsupported source languages use `??-??` |
| `--device` | `auto` | `cpu`, `cuda` for PyTorch/CTranslate2, or `metal` for MLX/whisper.cpp |
| `--compute-type` | `auto` | MLX uses FP16; Faster Whisper uses INT8 on CPU and FP16 on CUDA; original Whisper uses FP32 on CPU and FP16 on CUDA |
| `--threads` | Engine default | CPU thread count |
| `--beam-size` | Engine default | MLX uses `1` (greedy); other engines use `5`. MLX does not support beam search |
| `--prompt` | None | Optional vocabulary/context prompt |
| `--no-vad` | VAD enabled for Faster Whisper | Disable Faster Whisper's silence filtering |
| `--offline` | Off | Require a matching installed model and prohibit model downloads |
| `--model-dir` | Known caches | Add a model search root; repeat for several roots |
| `--cache-dir` | Project cache | Set the download/reuse cache |
| `--whisper-cpp-bin` / `--mw-bin` | PATH | Override the relevant CLI executable |
| `--word-srt` | Off | Also write a word-level `.srt` with the same timings |
| `--client-txt` | Off | Also write the legacy SRT-style `_client.txt` |
| `--speaker-name` | `Speaker 1` | Label for speech without speaker attribution |
| `-o` / `--output` | Next to input | JSON filename or output directory |
| `--overwrite` | Off | Allow replacing existing outputs |

Inference overrides such as device, compute type, prompt, threads, beam size, and VAD settings apply to the relevant standalone engine. MacWhisper manages inference settings inside its app; it rejects these overrides here. These commands do not run speaker diarization; MacWhisper is called with `--no-speakers` for efficiency.

```bash
# Reuse a cached model and forbid network model downloads.
uv run captions.py transcribe video.mp4 --offline --word-srt

# Choose Faster Whisper explicitly, with fast greedy decoding.
uv run captions.py transcribe video.mp4 --backend faster-whisper --beam-size 1

# Explicit Swiss German model and prompt, preserving the earlier workflow.
uv run captions.py transcribe interview.mp3 \
  --model nebi/whisper-large-v3-turbo-swiss-german-ct2-int8 \
  --language de --prompt "Schweizerdeutsch. Transkription auf Hochdeutsch." \
  --client-txt
```

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

## Legacy transcript editing

`from-srt` remains an alias for `to-premiere`. `--client-txt` retains the earlier client-editing output; converting an edited timestamped TXT with `to-premiere` estimates timing within its cues.

### `apply-edits`

Fuzzy-aligns an edited plain-text file with the original JSON to preserve precise word timings.

```bash
python3 captions.py apply-edits <original.json> <edited.txt>
```

---

`to-premiere` and `apply-edits` need no extra packages. The full `uv` project specifies Python 3.14+ for the transcription dependency environment. Alternatively, install transcription dependencies with `pip install -r requirements.txt` in a virtual environment.

## Tests

Run the standard-library regression tests with:

```bash
python3 -m unittest discover -v
```
