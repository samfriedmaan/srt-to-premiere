"""Audio/video transcription backends and direct Premiere transcript export."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import traceback
import uuid

from model_cache import ModelSource, model_cache_dir
from transcript_io import make_word_obj, words_to_segments


LANGUAGES = {
    "en": "en-us", "de": "de-de", "es": "es-es", "fr": "fr-fr", "ja": "ja-jp",
    "pt": "pt-pt", "ko": "ko-kr", "it": "it-it", "ru": "ru-ru", "hi": "hi-in",
    "no": "nb-no", "nn": "nb-no", "sv": "sv-se", "nl": "nl-nl", "da": "da-dk",
    "id": "id-id", "th": "th-th", "vi": "vi-vn", "ms": "ms-my", "tr": "tr-tr",
    "pl": "pl-pl", "tl": "fil-ph", "te": "te-in", "ml": "ml-in", "pa": "pa-in",
    "zh": "cmn-hans", "yue": "zh-hk",
}
PREMIERE_LANGUAGES = set(LANGUAGES.values()) | {"en-gb", "cmn-hant", "pt-br", "??-??"}


@dataclass
class TimedWord:
    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: str | None = None


@dataclass
class Transcription:
    words: list[TimedWord]
    language: str | None


def validate_media(media: Path) -> None:
    try:
        import av
    except ImportError as error:
        raise ValueError("Install transcription dependencies with uv sync or pip install -r requirements.txt") from error
    try:
        with av.open(str(media)) as container:
            if not container.streams.audio:
                raise ValueError(f"Media has no audio track: {media}")
    except av.error.FFmpegError as error:
        raise ValueError(f"Unable to read media: {media}: {error}") from error


def premiere_transcript(result: Transcription, language: str | None = None,
                        speaker_name: str = "Speaker 1") -> dict:
    language = language or LANGUAGES.get(result.language, "??-??")
    if language not in PREMIERE_LANGUAGES:
        raise ValueError(f"Unsupported Premiere language code: {language}")
    speakers = {}
    words = []
    for source in result.words:
        if not source.text.strip():
            continue
        values = (source.start, source.end, source.confidence)
        valid_numbers = all(isinstance(value, Real) and not isinstance(value, bool)
                            and math.isfinite(value) for value in values)
        if not valid_numbers or source.start < 0 or source.end < source.start or not 0 <= source.confidence <= 1:
            raise ValueError(f"Invalid word timing or confidence for {source.text!r}: "
                             f"start={source.start!r}, end={source.end!r}, confidence={source.confidence!r}")
        name = source.speaker or speaker_name
        if name not in speakers:
            speakers[name] = str(uuid.uuid4())
        speaker_id = speakers[name]
        word = make_word_obj(source.text.strip(), float(source.start), float(source.end) - float(source.start),
                             float(source.confidence), minimum_duration=0.0)
        word["_speaker"] = speaker_id
        words.append(word)
    if not words:
        raise ValueError("No speech with word timestamps was found")
    return {"language": language, "segments": words_to_segments(words, language),
            "speakers": [{"id": identifier, "name": name} for name, identifier in speakers.items()]}


def srt_timestamp(seconds: float) -> str:
    # Round once before splitting, so 59.9996 becomes 00:01:00,000.
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def render_srt(data: dict, word_level: bool = True) -> str:
    blocks = []
    for segment in data["segments"]:
        groups = [[word] for word in segment["words"]] if word_level else [segment["words"]]
        for words in groups:
            start = min(word["start"] for word in words)
            end = max(word["start"] + word["duration"] for word in words)
            blocks.append(f"{len(blocks) + 1}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n" + " ".join(word["text"] for word in words))
    return "\n\n".join(blocks) + "\n"


def _gpu_error(error: RuntimeError) -> bool:
    return any(marker in str(error).lower() for marker in ("cuda", "cudnn", "cublas", "out of memory", "driver", "library not found"))


def _faster_transcribe(source: ModelSource, media: Path, *, device: str, compute_type: str,
                       language: str | None, prompt: str | None, threads: int,
                       beam_size: int, vad: bool, cache_dir: Path, offline: bool) -> Transcription:
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
        from tqdm import tqdm
    except ImportError as error:
        raise ValueError("Install transcription dependencies with uv sync or pip install -r requirements.txt") from error
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("faster-whisper supports --device auto, cpu, or cuda")
    selected = device
    if selected == "auto":
        try:
            selected = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        except RuntimeError:
            selected = "cpu"

    def run(selected_device: str) -> Transcription:
        precision = ("float16" if selected_device == "cuda" else "int8") if compute_type == "auto" else compute_type
        print(f"Using faster-whisper on {selected_device} ({precision})")
        model = WhisperModel(source.location, device=selected_device, compute_type=precision,
                             cpu_threads=threads, download_root=str(cache_dir), local_files_only=offline or source.cached)
        segments, info = model.transcribe(str(media), language=language, initial_prompt=prompt,
                                          word_timestamps=True, beam_size=beam_size, vad_filter=vad,
                                          vad_parameters={"min_silence_duration_ms": 500},
                                          condition_on_previous_text=False)
        words = []
        with tqdm(total=info.duration, unit="sec", desc="Transcribing") as progress:
            for segment in segments:
                words.extend(TimedWord(word.word, word.start, word.end, word.probability) for word in segment.words or [])
                progress.update(max(0, min(info.duration, segment.end) - progress.n))
        return Transcription(words, info.language)
    try:
        return run(selected)
    except RuntimeError as error:
        if device != "auto" or selected != "cuda" or not _gpu_error(error):
            raise
        print(f"CUDA unavailable ({error}); retrying on CPU")
        traceback.clear_frames(error.__traceback__)
        # Automatic precision is recomputed for CPU. Explicit FP16 cannot run there.
        if compute_type in ("float16", "int8_float16"):
            compute_type = "int8"
        return run("cpu")


def _whisper_transcribe(source: ModelSource, media: Path, *, device: str, compute_type: str,
                        language: str | None, prompt: str | None, threads: int,
                        beam_size: int, cache_dir: Path) -> Transcription:
    try:
        import torch
        import whisper
        from faster_whisper.audio import decode_audio
    except ImportError as error:
        raise ValueError("Reusing .pt models requires openai-whisper: run uv sync or pip install -r requirements.txt") from error
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("The whisper backend supports --device auto, cpu, or cuda; use whisper-cpp or MacWhisper for Metal/Core ML")
    if compute_type not in ("auto", "float16", "float32"):
        raise ValueError("Original Whisper supports --compute-type auto, float16, or float32; INT8 requires faster-whisper")
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cpu" and compute_type == "float16":
        raise ValueError("Original Whisper cannot use float16 on CPU; select --compute-type auto or float32")
    if threads:
        torch.set_num_threads(threads)
    # PyAV reads the audio stream of video containers without a temporary WAV.
    audio = decode_audio(str(media))

    def run(selected_device: str) -> Transcription:
        fp16 = selected_device != "cpu" and compute_type != "float32"
        print(f"Using original Whisper on {selected_device} ({'float16' if fp16 else 'float32'})")
        model = whisper.load_model(source.location, device=selected_device, download_root=str(cache_dir / "whisper"))
        # Loading a checkpoint by path avoids ALL downloads, including on a
        # corrupt cache. Restore the official alignment preset for word timing.
        alignment = getattr(whisper, "_ALIGNMENT_HEADS", {}).get(source.name)
        if source.cached and alignment is not None:
            model.set_alignment_heads(alignment)
        result = model.transcribe(audio, language=language, initial_prompt=prompt,
                                  word_timestamps=True, fp16=fp16, beam_size=beam_size,
                                  condition_on_previous_text=False, verbose=False)
        words = [TimedWord(word["word"], word["start"], word["end"], word.get("probability", 1.0))
                 for segment in result["segments"] for word in segment.get("words", [])]
        return Transcription(words, result.get("language"))
    try:
        return run(selected)
    except RuntimeError as error:
        if device != "auto" or selected != "cuda" or not _gpu_error(error):
            raise
        print(f"CUDA unavailable ({error}); retrying on CPU")
        traceback.clear_frames(error.__traceback__)
        torch.cuda.empty_cache()
        return run("cpu")


def parse_macwhisper(data: dict, language: str | None = None) -> Transcription:
    """MacWhisper CLI JSON uses milliseconds, unlike Whisper/Premiere JSON."""
    words = []
    for segment in data.get("segments", []):
        if segment.get("text", "").strip() and not segment.get("words"):
            raise ValueError("MacWhisper's selected model did not return word timing; choose a model with word timestamps")
        for word in segment.get("words", []):
            words.append(TimedWord(word["text"], word["start"] / 1000, word["end"] / 1000,
                                   word.get("confidence", 1.0), segment.get("speaker")))
    return Transcription(words, language or data.get("language"))


def parse_cpp(data: dict) -> Transcription:
    """Merge timed BPE token fragments into words; attach punctuation to words."""
    words = []
    current: TimedWord | None = None

    def append_word(word: TimedWord) -> None:
        if words and all(not character.isalnum() for character in word.text):
            words[-1].text += word.text
            words[-1].end = max(words[-1].end, word.end)
        else:
            words.append(word)

    for segment in data.get("transcription", []):
        for token in segment.get("tokens", []):
            text = token.get("text", "")
            if not text or re.fullmatch(r"\[_.*\]|<\|.*\|>", text.strip()):
                continue
            offsets = token.get("offsets")
            if not offsets or offsets["from"] < 0 or offsets["to"] < offsets["from"]:
                raise ValueError("whisper-cli did not return valid token timestamps; update whisper.cpp")
            if text[0].isspace() and text.strip() and current:
                append_word(current)
                current = None
            if not text.strip():
                continue
            start, end = offsets["from"] / 1000, offsets["to"] / 1000
            if current is None:
                current = TimedWord(text.strip(), start, end, token.get("p", 1.0))
            else:
                current.text += text.strip()
                current.end = max(current.end, end)
                current.confidence = min(current.confidence, token.get("p", 1.0))
        if current:
            append_word(current)
            current = None
    return Transcription(words, data.get("result", {}).get("language"))


def _cpp_transcribe(source: ModelSource, media: Path, *, executable: str | None,
                    language: str | None, prompt: str | None, device: str,
                    threads: int, beam_size: int) -> Transcription:
    executable = executable or shutil.which("whisper-cli")
    ffmpeg = shutil.which("ffmpeg")
    if not executable or not ffmpeg:
        raise ValueError("whisper-cpp requires whisper-cli and ffmpeg on PATH (macOS: brew install whisper-cpp ffmpeg)")
    if device not in ("auto", "cpu", "metal"):
        raise ValueError("whisper-cpp supports --device auto, cpu, or metal")
    with tempfile.TemporaryDirectory(prefix="premiere-transcribe-") as directory:
        wav = Path(directory) / "audio.wav"
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(media), "-map", "0:a:0",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        output = Path(directory) / "transcript"
        command = [executable, "-m", source.location, "-f", str(wav), "-l", language or "auto",
                   "-ojf", "-of", str(output), "-ml", "1", "-sow", "-bs", str(beam_size)]
        if device == "cpu":
            command.append("-ng")
        if threads:
            command += ["-t", str(threads)]
        if prompt:
            command += ["--prompt", prompt]
        subprocess.run(command, check=True)
        return parse_cpp(json.loads(output.with_suffix(".json").read_text(encoding="utf-8")))


def transcribe(source: ModelSource, media: Path, *, language: str | None = None,
               prompt: str | None = None, device: str = "auto", compute_type: str = "auto",
               threads: int = 0, beam_size: int | None = None, vad: bool = True,
               cache_dir: Path | None = None, offline: bool = False,
               cpp_executable: str | None = None, mw_executable: str | None = None) -> Transcription:
    cache_dir = cache_dir or model_cache_dir()
    beam_size = beam_size if beam_size is not None else (1 if source.backend == "mlx" else 5)
    if source.backend == "mlx":
        from mlx_backend import transcribe_mlx
        return transcribe_mlx(source, media, cache_dir=cache_dir, offline=offline,
                              language=language, prompt=prompt, device=device,
                              compute_type=compute_type, beam_size=beam_size)
    if source.backend == "faster-whisper":
        return _faster_transcribe(source, media, device=device, compute_type=compute_type,
                                  language=language, prompt=prompt, threads=threads,
                                  beam_size=beam_size, vad=vad, cache_dir=cache_dir, offline=offline)
    if source.backend == "whisper":
        return _whisper_transcribe(source, media, device=device, compute_type=compute_type,
                                   language=language, prompt=prompt, threads=threads,
                                   beam_size=beam_size, cache_dir=cache_dir)
    if source.backend == "whisper-cpp":
        if compute_type != "auto":
            raise ValueError("whisper.cpp precision is determined by the model file; use a quantized .bin model instead of --compute-type")
        return _cpp_transcribe(source, media, executable=cpp_executable, language=language,
                               prompt=prompt, device=device, threads=threads, beam_size=beam_size)
    if source.backend == "macwhisper":
        executable = mw_executable or shutil.which("mw")
        if not executable:
            raise ValueError("Install MacWhisper's mw CLI from Settings > Advanced > Command-Line Tool")
        if device != "auto" or compute_type != "auto" or prompt or threads or beam_size != 5 or not vad:
            raise ValueError("MacWhisper controls inference settings in its app; device, compute type, prompt, threads, beam size, and VAD overrides require another backend")
        command = [executable, "transcribe", str(media), "--model", source.location,
                   "--format", "json", "--language", language or "auto", "--no-speakers"]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        return parse_macwhisper(json.loads(result.stdout), language)
    raise ValueError(f"Unsupported backend: {source.backend}")
