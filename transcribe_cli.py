"""CLI for cached-model discovery and audio/video-to-Premiere transcription."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from model_cache import (BACKENDS, DEFAULT_MODEL, MODEL_NAMES, cache_roots, discover_models,
                         macwhisper_models, model_cache_dir, resolve_model)
from transcription import PREMIERE_LANGUAGES, premiere_transcript, render_srt, transcribe, validate_media


def _model_options(parser):
    parser.add_argument("--model-dir", action="append", default=[], help="Additional model cache root; can be repeated")
    parser.add_argument("--cache-dir", type=Path, default=model_cache_dir(), help="Download cache directory")
    parser.add_argument("--mw-bin", help="Path to MacWhisper's mw CLI")


def cmd_models(args):
    parser = argparse.ArgumentParser(prog="captions.py models", description="List reusable installed Whisper models")
    parser.add_argument("--model", help="Filter by model name or Hugging Face repository")
    _model_options(parser)
    opts = parser.parse_args(args)
    roots = cache_roots([str(opts.cache_dir)] + opts.model_dir)
    app_models = macwhisper_models(opts.mw_bin)
    found = []
    for name in [opts.model] if opts.model else MODEL_NAMES:
        found.extend(discover_models(name, roots, app_models))
    if not found:
        print("No compatible cached models found.")
    else:
        print(f"Cached model files ({len(found)}):\n")
    formats = {"whisper": "Original Whisper / PyTorch (.pt)", "mlx": "Apple MLX",
               "faster-whisper": "CTranslate2", "whisper-cpp": "whisper.cpp (GGML)",
               "macwhisper": "MacWhisper app engine"}
    for source in found:
        location = Path(source.location)
        weight_files = [location] if location.is_file() else [location / filename for filename in
                        ("weights.safetensors", "weights.npz", "model.bin") if (location / filename).is_file()]
        size = f" ({sum(path.stat().st_size for path in weight_files) / 1024 ** 3:.2f} GiB)" if weight_files else ""
        print(f"  Model:  {source.name}\n  From:   {source.origin}\n  Format: {formats[source.backend]}{size}\n  Path:   {source.location}\n")
    try:
        selected = resolve_model(opts.model or DEFAULT_MODEL, roots=roots, app_models=app_models)
        print(f"Default selection: {selected.name} → {selected.backend}"
              f" ({'reuse cached weights' if selected.cached else 'download on first use'})")
        if selected.backend == "mlx":
            print("Device: Apple GPU (Metal). Cached .pt weights are converted locally once; no model redownload.")
        elif selected.backend == "whisper":
            print("Device: CPU on macOS; CPU/CUDA elsewhere. Use --backend whisper to select this engine explicitly.")
        elif selected.backend == "faster-whisper":
            print("Device: CPU (INT8) or NVIDIA GPU (CUDA), selected automatically.")
        elif selected.backend == "whisper-cpp":
            print("Device: Metal on supported Macs; CPU elsewhere.")
    except ValueError as error:
        print(f"Default selection unavailable: {error}")


def cmd_transcribe(args):
    parser = argparse.ArgumentParser(prog="captions.py transcribe", description="Transcribe audio or video directly to a word-timed Premiere transcript")
    parser.add_argument("media", help="Audio or video file (MP4, MOV, MKV, WAV, MP3, etc.)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name, Hugging Face repository, or local model path (default: large-v3-turbo)")
    parser.add_argument("--backend", choices=BACKENDS, default="auto", help="Inference engine (default: auto, reuse compatible caches first)")
    parser.add_argument("--language", default="auto", help="Whisper language code, or auto (default)")
    parser.add_argument("--premiere-language", help="Override Premiere language metadata; default is the detected language")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "metal"), default="auto")
    parser.add_argument("--compute-type", default="auto", help="auto, int8 (faster-whisper), float16, float32, etc.")
    parser.add_argument("--prompt", help="Optional transcription vocabulary/context prompt")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0: engine default)")
    parser.add_argument("--beam-size", type=int, help="Decoding beam size (default: 1 for MLX, 5 for other engines)")
    parser.add_argument("--no-vad", action="store_true", help="Disable faster-whisper voice activity filtering")
    parser.add_argument("--offline", action="store_true", help="Use cached models only; never download")
    parser.add_argument("--whisper-cpp-bin", help="Path to whisper-cli")
    parser.add_argument("--speaker-name", default="Speaker 1")
    parser.add_argument("--word-srt", action="store_true", help="Also write an SRT with one cue per word")
    parser.add_argument("--client-txt", action="store_true", help="Also write the legacy SRT-style client TXT")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    parser.add_argument("-o", "--output", help="Output JSON path or directory (default: next to input)")
    _model_options(parser)
    opts = parser.parse_args(args)
    try:
        started = time.monotonic()
        if opts.threads < 0 or (opts.beam_size is not None and opts.beam_size < 1):
            raise ValueError("--threads must be nonnegative and --beam-size must be positive")
        if opts.premiere_language and opts.premiere_language not in PREMIERE_LANGUAGES:
            raise ValueError(f"Unsupported Premiere language code: {opts.premiere_language}")
        media = Path(opts.media.replace("comappleCloudDocs", "com~apple~CloudDocs")).expanduser().resolve()
        if not media.is_file():
            raise ValueError(f"Media file not found: {media}")
        if opts.output:
            target = Path(opts.output.replace("comappleCloudDocs", "com~apple~CloudDocs")).expanduser()
            output = target if target.suffix.lower() == ".json" and not target.is_dir() else target / f"{media.stem}.json"
        else:
            output = media.with_suffix(".json")
        outputs = [output]
        if opts.word_srt:
            outputs.append(output.with_suffix(".srt"))
        if opts.client_txt:
            outputs.append(output.with_name(output.stem + "_client.txt"))
        for target in outputs:
            if target.resolve() == media:
                raise ValueError("Output cannot overwrite the input media file")
            if target.exists() and not opts.overwrite:
                raise ValueError(f"Output already exists: {target}. Use --overwrite to replace it.")
        validate_media(media)
        roots = cache_roots([str(opts.cache_dir.expanduser())] + opts.model_dir)
        app_models = macwhisper_models(opts.mw_bin) if opts.backend in ("auto", "macwhisper") else []
        source = resolve_model(opts.model, opts.backend, offline=opts.offline, roots=roots,
                               cpp_executable=opts.whisper_cpp_bin, app_models=app_models, device=opts.device)
        print(f"{'Reusing' if source.cached else 'Downloading'} {source.name} ({source.backend}, {source.origin}): {source.location}", flush=True)
        result = transcribe(source, media, language=None if opts.language == "auto" else opts.language,
                            prompt=opts.prompt, device=opts.device, compute_type=opts.compute_type,
                            threads=opts.threads, beam_size=opts.beam_size, vad=not opts.no_vad,
                            cache_dir=opts.cache_dir.expanduser(), offline=opts.offline,
                            cpp_executable=opts.whisper_cpp_bin, mw_executable=opts.mw_bin)
        data = premiere_transcript(result, opts.premiere_language, opts.speaker_name)
        for target in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if opts.word_srt:
            output.with_suffix(".srt").write_text(render_srt(data), encoding="utf-8")
        if opts.client_txt:
            output.with_name(output.stem + "_client.txt").write_text(render_srt(data, word_level=False), encoding="utf-8")
        count = sum(len(segment["words"]) for segment in data["segments"])
        print(f"Wrote {output} ({count} words, language={data['language']}, {time.monotonic() - started:.1f}s total)")
        for target in outputs[1:]:
            print(f"Wrote {target}")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
