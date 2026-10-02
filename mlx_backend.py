"""Apple Silicon inference with a one-time local conversion of Whisper weights."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile

from model_cache import ModelSource, converted_model_path, is_mlx_model, mlx_available


def prepare_model(source: ModelSource, cache: Path, offline: bool) -> Path:
    import mlx.core as mx

    checkpoint = Path(source.location)
    if is_mlx_model(checkpoint):
        return checkpoint
    if not source.cached:
        if offline:
            raise ValueError("--offline prevents model downloads")
        import whisper
        checkpoint = Path(whisper._download(whisper._MODELS[source.name], str(cache / "whisper"), False))
    if not checkpoint.is_file() or checkpoint.suffix != ".pt":
        raise ValueError("MLX requires a local Whisper .pt or MLX model directory")
    target = converted_model_path(checkpoint, source.name, cache)
    if is_mlx_model(target):
        print(f"Reusing Apple GPU cache: {target}", flush=True)
        return target

    import torch
    import whisper
    print("Preparing Apple GPU model from local weights (once; no model download).", flush=True)
    # Same tensor mapping as Apple's mlx-examples/whisper/convert.py.
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    weights = {}
    for key, tensor in checkpoint_data["model_state_dict"].items():
        if key == "encoder.positional_embedding":
            continue  # MLX regenerates these sinusoidal embeddings.
        key = key.replace("mlp.0", "mlp1").replace("mlp.2", "mlp2")
        if "conv" in key and tensor.ndim == 3:
            tensor = tensor.swapaxes(1, 2)
        weights[key] = mx.array(tensor.detach().numpy()).astype(mx.float16)
    config = dict(checkpoint_data["dims"], model_type="whisper")
    alignment = whisper._ALIGNMENT_HEADS.get(source.name)
    metadata = {"model": source.name, "source": str(checkpoint.resolve()),
                "alignment_heads": alignment.decode("ascii") if alignment else None}
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete directory only; interrupted conversions are never reused.
    with tempfile.TemporaryDirectory(prefix=".converting-", dir=target.parent) as directory:
        temporary = Path(directory)
        mx.save_safetensors(str(temporary / "weights.safetensors"), weights)
        (temporary / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (temporary / "source.json").write_text(json.dumps(metadata), encoding="utf-8")
        try:
            temporary.rename(target)
        except FileExistsError:
            if not is_mlx_model(target):
                raise
    print(f"Apple GPU model ready: {target}", flush=True)
    return target


def transcribe_mlx(source: ModelSource, media: Path, *, cache_dir: Path, offline: bool,
                   language: str | None, prompt: str | None, device: str,
                   compute_type: str, beam_size: int):
    from transcription import TimedWord, Transcription

    if not mlx_available() or device not in ("auto", "metal"):
        raise ValueError("MLX requires Apple Silicon and --device auto/metal")
    if compute_type not in ("auto", "float16", "float32"):
        raise ValueError("MLX supports --compute-type auto, float16, or float32")
    if beam_size != 1:
        raise ValueError("MLX uses greedy decoding (--beam-size 1); beam search requires --backend whisper or faster-whisper")
    import mlx.core as mx
    from faster_whisper.audio import decode_audio

    model_path = prepare_model(source, cache_dir, offline)
    module = importlib.import_module("mlx_whisper.transcribe")
    fp16 = compute_type != "float32"
    print(f"Using MLX on Apple GPU ({'float16' if fp16 else 'float32'}, greedy decoding)", flush=True)
    model = module.ModelHolder.get_model(str(model_path), mx.float16 if fp16 else mx.float32)
    metadata_path = model_path / "source.json"
    if metadata_path.is_file():
        alignment = json.loads(metadata_path.read_text(encoding="utf-8")).get("alignment_heads")
        if alignment:
            model.set_alignment_heads(alignment.encode("ascii"))
    result = module.transcribe(decode_audio(str(media)), path_or_hf_repo=str(model_path),
                               language=language, initial_prompt=prompt, word_timestamps=True,
                               condition_on_previous_text=False, fp16=fp16, verbose=False)
    words = [TimedWord(word["word"], word["start"], word["end"], word.get("probability", 1.0))
             for segment in result["segments"] for word in segment.get("words", [])]
    return Transcription(words, result.get("language"))
