import queue
import sys
import threading
import types
from unittest.mock import MagicMock

from tools import tts_streaming as ts


def test_openai_tts_advertises_audio_streaming_but_not_continuous_text_input():
    caps = ts.provider_realtime_tts_capabilities("openai")

    assert caps["streaming_audio_output"] is True
    assert caps["continuous_text_input"] is False


def test_elevenlabs_tts_advertises_true_continuous_text_input(monkeypatch):
    monkeypatch.setattr(ts, "_resolve_key", lambda env, provider_id: "key")

    caps = ts.provider_realtime_tts_capabilities("elevenlabs")

    assert caps["streaming_audio_output"] is True
    assert caps["continuous_text_input"] is True


def test_elevenlabs_continuous_tts_uses_one_realtime_session_for_turn(monkeypatch):
    calls = []

    class _RealtimeClient:
        def convert_realtime(self, **kwargs):
            calls.append(kwargs)
            for text in kwargs["text"]:
                yield f"audio:{text}".encode()

    class _ElevenLabs:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})
            self.text_to_speech = _RealtimeClient()

    monkeypatch.setattr(ts, "_resolve_key", lambda env, provider_id: "key")
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: _ElevenLabs)

    provider = ts.ElevenLabsContinuousTextStreamer(
        {"elevenlabs": {"voice_id": "voice", "model_id": "eleven_flash_v2_5"}},
        {"voice_id": "voice", "model_id": "eleven_flash_v2_5"},
    )
    text = queue.Queue()
    text.put("Hello")
    text.put(", world")
    text.put(None)

    assert list(provider.stream_turn(text, threading.Event())) == [b"audio:Hello", b"audio:, world"]
    realtime_calls = [call for call in calls if "text" in call]
    assert len(realtime_calls) == 1
    assert realtime_calls[0]["voice_id"] == "voice"
    assert realtime_calls[0]["model_id"] == "eleven_flash_v2_5"
    assert realtime_calls[0]["output_format"] == "pcm_24000"


def test_configured_elevenlabs_resolves_to_continuous_text_streamer(monkeypatch):
    monkeypatch.setattr(ts, "_resolve_key", lambda env, provider_id: "key")

    provider = ts.resolve_streaming_provider({
        "provider": "elevenlabs",
        "elevenlabs": {"voice_id": "voice"},
    })

    assert isinstance(provider, ts.ElevenLabsContinuousTextStreamer)


def test_elevenlabs_continuous_tts_cancellation_stops_delta_input(monkeypatch):
    consumed = []

    class _RealtimeClient:
        def convert_realtime(self, **kwargs):
            for text in kwargs["text"]:
                consumed.append(text)
                yield b"audio"

    class _ElevenLabs:
        def __init__(self, **_kwargs):
            self.text_to_speech = _RealtimeClient()

    monkeypatch.setattr(ts, "_resolve_key", lambda env, provider_id: "key")
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: _ElevenLabs)

    stop = threading.Event()
    text = queue.Queue()
    text.put("first")
    text.put("second")
    text.put(None)

    provider = ts.ElevenLabsContinuousTextStreamer({}, {})
    audio_iter = provider.stream_turn(text, stop)
    assert next(audio_iter) == b"audio"
    stop.set()
    assert list(audio_iter) == []
    assert consumed == ["first"]


def test_elevenlabs_continuous_tts_cancellation_unblocks_empty_delta_queue(monkeypatch):
    entered_realtime = threading.Event()
    finished = threading.Event()

    class _RealtimeClient:
        def convert_realtime(self, **kwargs):
            entered_realtime.set()
            for _text in kwargs["text"]:
                yield b"audio"

    class _ElevenLabs:
        def __init__(self, **_kwargs):
            self.text_to_speech = _RealtimeClient()

    monkeypatch.setattr(ts, "_resolve_key", lambda env, provider_id: "key")
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: _ElevenLabs)

    stop = threading.Event()
    text = queue.Queue()
    provider = ts.ElevenLabsContinuousTextStreamer({}, {})

    def consume():
        try:
            list(provider.stream_turn(text, stop))
        finally:
            finished.set()

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    assert entered_realtime.wait(1)

    stop.set()

    assert finished.wait(1), "cancellation must unblock an empty ElevenLabs delta queue"


def test_speaker_pipeline_uses_continuous_provider_once_for_whole_turn(monkeypatch):
    from tools import tts_tool

    consumed = []
    calls = []

    class _Continuous(ts.ContinuousTextTTSProvider):
        @staticmethod
        def available():
            return True

        def stream(self, text):
            raise AssertionError("sentence stream path should not be used")

        def stream_turn(self, text_queue, stop_event):
            calls.append("stream_turn")
            while not stop_event.is_set():
                delta = text_queue.get()
                if delta is None:
                    return
                consumed.append(delta)
                yield b"\x00\x00"

    out = MagicMock()
    sd = MagicMock()
    sd.OutputStream.return_value = out
    fake_np = types.SimpleNamespace(
        int16="int16",
        frombuffer=lambda chunk, dtype: types.SimpleNamespace(reshape=lambda *_args: chunk),
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setattr(tts_tool.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: sd)
    monkeypatch.setattr(ts, "resolve_streaming_provider", lambda *_args, **_kwargs: _Continuous({}, {}))

    text = queue.Queue()
    text.put("Hello")
    text.put(", world")
    text.put(None)
    done = threading.Event()

    tts_tool.stream_tts_to_speaker(text, threading.Event(), done, provider="elevenlabs")

    assert calls == ["stream_turn"]
    assert consumed == ["Hello", ", world"]
    assert out.write.call_count == 2
    assert done.is_set()


def test_speaker_pipeline_does_not_whole_turn_buffer_continuous_provider_without_pcm_output(monkeypatch):
    from tools import tts_tool

    stream_turn_called = False
    spoken = []
    played = []

    class _Continuous(ts.ContinuousTextTTSProvider):
        @staticmethod
        def available():
            return True

        def stream(self, text):
            raise AssertionError("continuous provider should not stream without live PCM output")

        def stream_turn(self, text_queue, stop_event):
            nonlocal stream_turn_called
            stream_turn_called = True
            while not stop_event.is_set():
                delta = text_queue.get()
                if delta is None:
                    return
                yield f"audio:{delta}".encode()

    monkeypatch.setattr(tts_tool.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ts, "resolve_streaming_provider", lambda *_args, **_kwargs: _Continuous({}, {}))
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", lambda text, output_path: spoken.append(text) or output_path)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", lambda path: played.append(path))
    monkeypatch.setattr(tts_tool.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(tts_tool.os.path, "getsize", lambda path: 1)
    monkeypatch.setattr(tts_tool.os, "unlink", lambda path: None)

    text = queue.Queue()
    text.put("Hello, world. ")
    text.put("Second sentence.")
    text.put(None)
    done = threading.Event()

    tts_tool.stream_tts_to_speaker(text, threading.Event(), done, provider="elevenlabs")

    assert stream_turn_called is False
    assert spoken == ["Hello, world. Second sentence."]
    assert len(played) == 1
    assert done.is_set()
