import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model_cache import (ModelSource, cache_roots, converted_model_path, discover_models,
                         macwhisper_models, resolve_model)
from transcribe_cli import cmd_models, cmd_transcribe
from transcription import (TimedWord, Transcription, parse_cpp, parse_macwhisper,
                           premiere_transcript, render_srt, srt_timestamp, transcribe)


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        engine = patch("model_cache.mlx_available", return_value=False)
        engine.start()
        self.addCleanup(engine.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def checkpoint(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path

    def ct2(self, name):
        path = self.root / name
        path.mkdir(parents=True)
        for filename in ("config.json", "tokenizer.json", "model.bin"):
            (path / filename).write_text("{}")
        return path

    def test_buzz_checkpoint_reused_offline(self):
        path = self.checkpoint("whisper/large-v3-turbo.pt")
        source = resolve_model("turbo", offline=True, roots=[("Buzz", self.root)], app_models=[])
        self.assertEqual((source.backend, source.location, source.cached), ("whisper", str(path), True))

    def test_apple_gpu_reuses_buzz_weights_and_cpu_override_still_works(self):
        path = self.checkpoint("whisper/large-v3-turbo.pt")
        with patch("model_cache.mlx_available", return_value=True):
            gpu = resolve_model(offline=True, roots=[("Buzz", self.root)], app_models=[])
            cpu = resolve_model(offline=True, roots=[("Buzz", self.root)], app_models=[], device="cpu")
            explicit = resolve_model(str(path), backend="whisper")
        self.assertEqual((gpu.backend, gpu.location, gpu.cached), ("mlx", str(path), True))
        self.assertEqual(cpu.backend, "whisper")
        self.assertEqual(explicit.backend, "whisper")

    def test_converted_turbo_is_not_mistaken_for_large_v3_and_stale_cache_is_ignored(self):
        checkpoint = self.checkpoint("whisper/large-v3-turbo.pt")
        converted = converted_model_path(checkpoint, "large-v3-turbo", self.root)
        converted.mkdir(parents=True)
        (converted / "config.json").write_text("{}")
        (converted / "weights.safetensors").write_bytes(b"fixture")
        (converted / "source.json").write_text(json.dumps({"model": "large-v3-turbo", "source": str(checkpoint)}))
        with patch("model_cache.mlx_available", return_value=True):
            source = resolve_model(offline=True, roots=[("cache", self.root)], app_models=[])
        self.assertEqual(source.location, str(converted))
        self.assertEqual(discover_models("large-v3", [("cache", self.root)]), [])
        checkpoint.write_bytes(b"different checkpoint")
        self.assertFalse(any(model.backend == "mlx" for model in discover_models("turbo", [("cache", self.root)])))

    def test_model_listing_explains_format_origin_and_default_engine(self):
        source = ModelSource("whisper", str(self.checkpoint("whisper/large-v3-turbo.pt")), "large-v3-turbo", "Buzz")
        output = io.StringIO()
        with patch("transcribe_cli.discover_models", return_value=[source]), patch("transcribe_cli.macwhisper_models", return_value=[]), patch("transcribe_cli.resolve_model", return_value=source), contextlib.redirect_stdout(output):
            cmd_models(["--model", "turbo"])
        self.assertIn("Model:  large-v3-turbo", output.getvalue())
        self.assertIn("From:   Buzz", output.getvalue())
        self.assertIn("Original Whisper / PyTorch (.pt)", output.getvalue())
        self.assertIn("Default selection:", output.getvalue())

    def test_damaged_conversion_metadata_does_not_hide_existing_checkpoint(self):
        checkpoint = self.checkpoint("whisper/large-v3-turbo.pt")
        converted = converted_model_path(checkpoint, "large-v3-turbo", self.root)
        converted.mkdir(parents=True)
        (converted / "config.json").write_text("{}")
        (converted / "weights.safetensors").write_bytes(b"fixture")
        (converted / "source.json").write_text("incomplete metadata")
        sources = discover_models("turbo", [("cache", self.root)])
        self.assertEqual([(source.backend, source.location) for source in sources], [("whisper", str(checkpoint))])

    def test_does_not_silently_substitute_a_smaller_model(self):
        self.checkpoint("whisper/small.pt")
        with self.assertRaisesRegex(ValueError, "offline"):
            resolve_model("large-v3-turbo", offline=True, roots=[("Buzz", self.root)], app_models=[])

    def test_cold_start_downloads_optimized_turbo(self):
        source = resolve_model(roots=[], app_models=[])
        self.assertEqual(source.backend, "faster-whisper")
        self.assertEqual(source.location, "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
        self.assertFalse(source.cached)

    def test_explicit_original_whisper_download(self):
        source = resolve_model("small", "whisper", roots=[], app_models=[])
        self.assertEqual((source.backend, source.location, source.cached), ("whisper", "small", False))

    def test_hugging_face_repository_and_cached_current_revision(self):
        repo = "models--vendor--custom"
        current = self.ct2(f"{repo}/snapshots/current")
        self.ct2(f"{repo}/snapshots/old")
        self.checkpoint(f"{repo}/refs/main").write_text("current")
        source = resolve_model("vendor/custom", offline=True, roots=[("Hugging Face", self.root)], app_models=[])
        self.assertEqual(source.location, str(current))

    def test_incomplete_ct2_cache_does_not_trigger_hidden_tokenizer_download_offline(self):
        self.checkpoint("small/model.bin")
        self.checkpoint("small/config.json")
        with self.assertRaisesRegex(ValueError, "offline"):
            resolve_model("small", offline=True, roots=[("custom", self.root)], app_models=[])

    def test_explicit_local_paths_and_wrong_backend(self):
        path = self.checkpoint("custom.pt")
        self.assertEqual(resolve_model(str(path)).backend, "whisper")
        with self.assertRaisesRegex(ValueError, "requires --backend whisper"):
            resolve_model(str(path), "faster-whisper")
        self.assertEqual(resolve_model(str(self.ct2("custom-ct2"))).backend, "faster-whisper")

    def test_nonexistent_local_path_is_not_a_download_request(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            resolve_model(str(self.root / "missing.pt"))

    def test_macwhisper_cpp_model_is_detected_with_quantized_variant(self):
        path = self.checkpoint("ggml-model-whisper-large-v3-turbo-q5_0.bin")
        source = resolve_model(roots=[("MacWhisper", self.root)], cpp_executable="whisper-cli", app_models=[])
        self.assertEqual((source.backend, source.location), ("whisper-cpp", str(path)))

    def test_missing_cpp_engine_does_not_redownload_cached_weights(self):
        self.checkpoint("ggml-small.bin")
        with patch("model_cache.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "whisper-cli is missing"):
                resolve_model("small", roots=[("MacWhisper", self.root)], app_models=[])

    def test_requested_device_filters_out_incompatible_cached_engines(self):
        self.checkpoint("ggml-small.bin")
        original = self.checkpoint("whisper/small.pt")
        source = resolve_model("small", device="cuda", roots=[("cache", self.root)],
                               cpp_executable="whisper-cli", app_models=[])
        self.assertEqual((source.backend, source.location), ("whisper", str(original)))
        with self.assertRaisesRegex(ValueError, "does not support --device metal"):
            resolve_model("small", backend="faster-whisper", device="metal", roots=[], app_models=[])

    def test_engine_override_can_download_different_format(self):
        self.checkpoint("whisper/large-v3-turbo.pt")
        source = resolve_model(backend="faster-whisper", roots=[("Buzz", self.root)], app_models=[])
        self.assertFalse(source.cached)

    def test_macwhisper_lists_only_installed_local_whisper_engines(self):
        result = subprocess.CompletedProcess([], 0, stdout="ID NAME SIZE\n  whisperkit:openai_whisper-small Small 483 MB\n  whisperkit:openai_whisper-large-v3_turbo Turbo 1.5GB\n  cloud:openai Turbo -\n  parakeet-pro:model Parakeet 494 MB\n")
        with patch("model_cache.subprocess.run", return_value=result):
            models = macwhisper_models("/path with spaces/mw")
        self.assertEqual([(source.name, source.location) for source in models], [("small", "whisperkit:openai_whisper-small"), ("large-v3-turbo", "whisperkit:openai_whisper-large-v3_turbo")])
        source = resolve_model("small", "macwhisper", roots=[], app_models=models)
        self.assertEqual(source, models[0])

    def test_platform_cache_locations_and_environment_overrides(self):
        with patch("model_cache.Path.home", return_value=self.root), patch("model_cache.platform.system", return_value="Darwin"), patch.dict("os.environ", {"BUZZ_MODEL_ROOT": str(self.root / "buzz-custom"), "HF_HUB_CACHE": str(self.root / "hf")}, clear=True):
            roots = dict((str(path), origin) for origin, path in cache_roots())
        self.assertIn(str(self.root / "Library/Caches/Buzz/models"), roots)
        self.assertIn(str(self.root / "Library/Containers/com.goodsnooze.MacWhisper/Data/Library/Application Support/MacWhisper/models"), roots)
        self.assertIn(str(self.root / "buzz-custom"), roots)
        self.assertIn(str(self.root / "hf"), roots)


class TranscriptionTests(unittest.TestCase):
    def test_mlx_restores_word_alignment_and_exports_engine_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "source.json").write_text(json.dumps({"alignment_heads": "alignment"}))
            model = Mock()
            holder = SimpleNamespace(get_model=Mock(return_value=model))
            engine = Mock(return_value={"language": "en", "segments": [{"words": [
                {"word": " New", "start": 0.0, "end": 0.12, "probability": 0.718657732},
            ]}]})
            core = SimpleNamespace(float16="float16", float32="float32")
            modules = {"mlx": SimpleNamespace(core=core), "mlx.core": core,
                       "mlx_whisper.transcribe": SimpleNamespace(ModelHolder=holder, transcribe=engine),
                       "faster_whisper.audio": SimpleNamespace(decode_audio=Mock(return_value="audio frames"))}
            source = ModelSource("mlx", str(path), "large-v3-turbo", "cache")
            with patch.dict("sys.modules", modules), patch("mlx_backend.mlx_available", return_value=True), patch("mlx_backend.prepare_model", return_value=path), contextlib.redirect_stdout(io.StringIO()):
                data = premiere_transcript(transcribe(source, Path("video.mp4"), offline=True))
            model.set_alignment_heads.assert_called_once_with(b"alignment")
            self.assertTrue(engine.call_args.kwargs["word_timestamps"])
            self.assertNotIn("beam_size", engine.call_args.kwargs)
            self.assertEqual(engine.call_args.args[0], "audio frames")
            word = data["segments"][0]["words"][0]
            self.assertEqual((word["text"], word["start"], word["duration"], word["confidence"]), ("New", 0.0, 0.12, 0.719))

    def test_faster_whisper_requests_word_timing_and_falls_back_when_gpu_initialization_fails(self):
        source = ModelSource("faster-whisper", "/cached/ct2", "tiny", "cache")
        model = Mock()
        segment = SimpleNamespace(end=0.1, words=[SimpleNamespace(word="Hi", start=0, end=0.06, probability=0.9)])
        model.transcribe.return_value = (iter([segment]), SimpleNamespace(duration=0.1, language="en"))
        constructor = Mock(side_effect=[RuntimeError("CUDA driver failed"), model])
        modules = {"ctranslate2": SimpleNamespace(get_cuda_device_count=lambda: 1),
                   "faster_whisper": SimpleNamespace(WhisperModel=constructor),
                   "tqdm": SimpleNamespace(tqdm=lambda **kwargs: contextlib.nullcontext(SimpleNamespace(n=0, update=Mock())))}
        with patch.dict("sys.modules", modules), contextlib.redirect_stdout(io.StringIO()):
            result = transcribe(source, Path("video.mp4"), offline=True)
        self.assertEqual(constructor.call_args_list[0].kwargs["device"], "cuda")
        self.assertEqual(constructor.call_args_list[1].kwargs["device"], "cpu")
        self.assertEqual(constructor.call_args_list[1].kwargs["compute_type"], "int8")
        self.assertTrue(constructor.call_args.kwargs["local_files_only"])
        self.assertTrue(model.transcribe.call_args.kwargs["word_timestamps"])
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertEqual(result.words[0].end, 0.06)

    def test_original_checkpoint_uses_video_audio_and_restores_alignment_heads(self):
        source = ModelSource("whisper", "/cached/large-v3-turbo.pt", "large-v3-turbo", "Buzz")
        model = Mock()
        model.transcribe.return_value = {"language": "de", "segments": [{"words": [{"word": "Hi", "start": 0.02, "end": 0.06, "probability": 0.9}]}]}
        loader = Mock(return_value=model)
        decoder = Mock(return_value="decoded frames")
        modules = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), set_num_threads=Mock()),
                   "whisper": SimpleNamespace(load_model=loader, _ALIGNMENT_HEADS={"large-v3-turbo": b"alignment"}),
                   "faster_whisper.audio": SimpleNamespace(decode_audio=decoder)}
        with patch.dict("sys.modules", modules), contextlib.redirect_stdout(io.StringIO()):
            result = transcribe(source, Path("video.mp4"), offline=True)
        self.assertEqual(loader.call_args.args[0], source.location)
        decoder.assert_called_once_with("video.mp4")
        model.set_alignment_heads.assert_called_once_with(b"alignment")
        self.assertTrue(model.transcribe.call_args.kwargs["word_timestamps"])
        self.assertEqual(model.transcribe.call_args.args[0], "decoded frames")
        self.assertEqual((result.language, result.words[0].end), ("de", 0.06))

    def test_precise_short_and_zero_duration_timing_is_preserved(self):
        result = Transcription([TimedWord("100", 0.0, 0.06, 0.9), TimedWord("words.", 0.06, 0.06, 0.8)], "en")
        data = premiere_transcript(result)
        words = data["segments"][0]["words"]
        self.assertEqual(data["language"], "en-us")
        self.assertEqual([(word["start"], word["duration"]) for word in words], [(0.0, 0.06), (0.06, 0.0)])
        self.assertEqual(words[0]["confidence"], 0.9)
        self.assertNotIn("_speaker", words[0])

    def test_numpy_word_numbers_export_as_native_json_numbers(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is installed with transcription dependencies")
        data = premiere_transcript(Transcription([
            TimedWord(" New", np.float64(0), np.float64(0.12), np.float64(0.718657732)),
            TimedWord("York", np.float32(0.12), np.float32(0.36), np.float32(0.99)),
        ], "en"))
        json.dumps(data, allow_nan=False)
        words = data["segments"][0]["words"]
        self.assertEqual((words[0]["start"], words[0]["duration"], words[0]["confidence"]), (0, 0.12, 0.719))
        self.assertTrue(all(type(word[key]) is float for word in words for key in ("start", "duration", "confidence")))

    def test_language_detection_overrides_and_unknown_languages(self):
        result = Transcription([TimedWord("Hallo.", 0, 1)], "de")
        self.assertEqual(premiere_transcript(result)["language"], "de-de")
        self.assertEqual(premiere_transcript(result, "en-gb")["language"], "en-gb")
        result.language = "gsw"
        self.assertEqual(premiere_transcript(result)["language"], "??-??")
        with self.assertRaisesRegex(ValueError, "Unsupported Premiere language"):
            premiere_transcript(result, "xx-xx")

    def test_speaker_references_and_turns(self):
        data = premiere_transcript(Transcription([TimedWord("one", 0, 1, speaker="Alice"), TimedWord("two", 1, 2, speaker="Bob")], "en"))
        self.assertEqual([speaker["name"] for speaker in data["speakers"]], ["Alice", "Bob"])
        self.assertEqual([segment["speaker"] for segment in data["segments"]], [speaker["id"] for speaker in data["speakers"]])

    def test_invalid_engine_word_timing_fails_cleanly(self):
        for word in (TimedWord("bad", 1, 0), TimedWord("bad", -1, 0), TimedWord("bad", 0, float("nan")), TimedWord("bad", 0, 1, float("nan"))):
            with self.subTest(word=word), self.assertRaisesRegex(ValueError, "Invalid word"):
                premiere_transcript(Transcription([word], "en"))
        with self.assertRaisesRegex(ValueError, "No speech"):
            premiere_transcript(Transcription([], "en"))

    def test_macwhisper_milliseconds_are_converted_without_estimation(self):
        result = parse_macwhisper({"segments": [{"text": "Hello world.", "speaker": "Alice", "words": [{"text": "Hello", "start": 0, "end": 640}, {"text": "world.", "start": 660, "end": 1600}]}]})
        self.assertEqual([(word.text, word.start, word.end, word.speaker) for word in result.words], [("Hello", 0, 0.64, "Alice"), ("world.", 0.66, 1.6, "Alice")])

    def test_macwhisper_missing_word_timestamps_is_not_silently_estimated(self):
        with self.assertRaisesRegex(ValueError, "did not return word timing"):
            parse_macwhisper({"segments": [{"text": "Hello world.", "start": 0, "end": 1000}]})

    def test_cpp_merges_timed_subword_tokens_and_punctuation(self):
        def token(text, start, end):
            return {"text": text, "offsets": {"from": start, "to": end}, "p": 0.9}
        result = parse_cpp({"result": {"language": "en"}, "transcription": [{"tokens": [token("[_BEG_]", 0, 0), token(" Hello", 100, 300), token(" un", 400, 500), token("usual", 500, 800), token(".", 800, 820)]}]})
        self.assertEqual([(word.text, word.start, word.end) for word in result.words], [("Hello", 0.1, 0.3), ("unusual.", 0.4, 0.82)])

    def test_cpp_timestamp_tokens_and_separate_punctuation_segments(self):
        result = parse_cpp({"transcription": [
            {"tokens": [{"text": "[_TT_1000]"}, {"text": " Hello", "offsets": {"from": 0, "to": 100}}]},
            {"tokens": [{"text": ".", "offsets": {"from": 100, "to": 120}}]},
        ]})
        self.assertEqual([(word.text, word.start, word.end) for word in result.words], [("Hello.", 0, 0.12)])

    def test_cpp_missing_word_timing_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "token timestamps"):
            parse_cpp({"transcription": [{"tokens": [{"text": " Hello"}]}]})

    def test_srt_rounding_carries_into_next_minute(self):
        self.assertEqual(srt_timestamp(59.9996), "00:01:00,000")
        data = premiere_transcript(Transcription([TimedWord("Hello", 0.06, 0.12), TimedWord("world.", 0.2, 0.2)], "en"))
        self.assertIn("00:00:00,200 --> 00:00:00,200\nworld.", render_srt(data))
        self.assertIn("Hello world.", render_srt(data, word_level=False))

    def test_macwhisper_command_uses_local_model_and_safe_argument_vector(self):
        source = ModelSource("macwhisper", "whisperkit:openai_whisper-small", "small", "MacWhisper")
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"segments": [{"words": [{"text": "Hi", "start": 0, "end": 100}]}]}))
        with patch("transcription.subprocess.run", return_value=result) as run:
            transcribe(source, Path("/tmp/video with $characters.mp4"), mw_executable="/tmp/mw")
        self.assertIn("/tmp/video with $characters.mp4", run.call_args.args[0])
        self.assertIn("--no-speakers", run.call_args.args[0])
        self.assertNotIn("shell", run.call_args.kwargs)


class TranscribeCLITests(unittest.TestCase):
    def test_end_to_end_exports_and_model_resolution_without_loading_real_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "video.mp4"
            media.write_bytes(b"fixture")
            result = Transcription([TimedWord("Hi.", 0.02, 0.08)], "de")
            with patch("transcribe_cli.validate_media"), patch("transcribe_cli.macwhisper_models", return_value=[]), patch("transcribe_cli.resolve_model") as resolve, patch("transcribe_cli.transcribe", return_value=result) as run, contextlib.redirect_stdout(io.StringIO()):
                cmd_transcribe([str(media), "--offline", "--word-srt", "--client-txt", "-o", str(root / "new" / "output.json")])
            self.assertTrue(resolve.call_args.kwargs["offline"])
            self.assertIsNone(run.call_args.kwargs["language"])
            self.assertEqual(json.loads((root / "new/output.json").read_text())["language"], "de-de")
            self.assertTrue((root / "new/output.srt").is_file())
            self.assertTrue((root / "new/output_client.txt").is_file())

    def test_existing_output_is_checked_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "video.mp4"
            media.write_bytes(b"fixture")
            media.with_suffix(".json").write_text("keep this")
            with patch("transcribe_cli.transcribe") as run, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cmd_transcribe([str(media)])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
            self.assertEqual(media.with_suffix(".json").read_text(), "keep this")

    def test_no_audio_is_reported_before_resolving_or_downloading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "silent.mp4"
            media.write_bytes(b"fixture")
            with patch("transcribe_cli.validate_media", side_effect=ValueError("Media has no audio track")), patch("transcribe_cli.resolve_model") as resolve, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cmd_transcribe([str(media)])
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
