import pytest
from urllib.parse import parse_qs, urlparse

from agent.voice_realtime import (
    RealtimeCapability,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeVoiceOrchestrator,
    VoiceProviderCapabilities,
    broker_elevenlabs_realtime_stt_bridge_session,
    broker_openai_realtime_transcription_session,
    consume_elevenlabs_realtime_stt_bridge_token,
    normalize_elevenlabs_realtime_stt_event,
    normalize_openai_realtime_transcription_event,
    select_voice_capabilities,
    stt_provider_realtime_capabilities,
    tts_provider_realtime_capabilities,
)


def test_capability_selection_falls_back_independently():
    stt = VoiceProviderCapabilities(
        provider="openai",
        native=frozenset({
            RealtimeCapability.MIC_AUDIO_INPUT,
            RealtimeCapability.TRANSCRIPT_PARTIAL,
            RealtimeCapability.TRANSCRIPT_FINAL,
            RealtimeCapability.PROVIDER_VAD,
        }),
    )
    tts = VoiceProviderCapabilities(
        provider="edge",
        native=frozenset(),
    )

    selected = select_voice_capabilities(stt=stt, tts=tts)

    assert selected.uses_native(RealtimeCapability.TRANSCRIPT_PARTIAL)
    assert selected.uses_native(RealtimeCapability.PROVIDER_VAD)
    assert selected.fallback_for(RealtimeCapability.CONTINUOUS_TTS_INPUT) == "sentence_chunked_tts"
    assert selected.fallback_for(RealtimeCapability.STREAMING_TTS_AUDIO) == "sentence_chunked_tts"


def test_builtin_provider_capability_adapters_are_honest(monkeypatch):
    monkeypatch.setattr(
        "tools.tts_streaming.provider_realtime_tts_capabilities",
        lambda provider: {
            "streaming_audio_output": provider in {"openai", "elevenlabs"},
            "continuous_text_input": provider == "elevenlabs",
        },
    )

    openai_stt = stt_provider_realtime_capabilities("openai")
    eleven_stt = stt_provider_realtime_capabilities("elevenlabs")
    edge_tts = tts_provider_realtime_capabilities("edge")
    openai_tts = tts_provider_realtime_capabilities("openai")
    eleven_tts = tts_provider_realtime_capabilities("elevenlabs")

    assert RealtimeCapability.TRANSCRIPT_PARTIAL in openai_stt.native
    assert RealtimeCapability.PROVIDER_VAD in eleven_stt.native
    assert RealtimeCapability.STREAMING_TTS_AUDIO not in edge_tts.native
    assert RealtimeCapability.STREAMING_TTS_AUDIO in openai_tts.native
    assert RealtimeCapability.CONTINUOUS_TTS_INPUT not in openai_tts.native
    assert RealtimeCapability.CONTINUOUS_TTS_INPUT in eleven_tts.native


def test_openai_realtime_transcription_events_are_normalized():
    partial = normalize_openai_realtime_transcription_event({
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": "msg_1",
        "delta": "hel",
    })
    final = normalize_openai_realtime_transcription_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "msg_1",
        "transcript": "hello",
    })
    speech = normalize_openai_realtime_transcription_event({
        "type": "input_audio_buffer.speech_started",
    })
    response = normalize_openai_realtime_transcription_event({
        "type": "response.created",
    })

    assert partial == RealtimeEvent(RealtimeEventType.TRANSCRIPT_PARTIAL, text="hel", provider_event_id="msg_1")
    assert final == RealtimeEvent(RealtimeEventType.TRANSCRIPT_FINAL, text="hello", provider_event_id="msg_1")
    assert speech == RealtimeEvent(RealtimeEventType.SPEECH_STARTED)
    assert response == RealtimeEvent(
        RealtimeEventType.ERROR,
        message="Realtime transcription sessions must not generate LLM responses",
    )


def test_elevenlabs_realtime_stt_events_are_normalized():
    assert normalize_elevenlabs_realtime_stt_event({
        "type": "partial_transcript",
        "transcript": "hel",
        "id": "seg-1",
    }) == RealtimeEvent(RealtimeEventType.TRANSCRIPT_PARTIAL, text="hel", provider_event_id="seg-1")

    assert normalize_elevenlabs_realtime_stt_event({
        "type": "committed_transcript",
        "text": "hello",
        "segment_id": "seg-1",
    }) == RealtimeEvent(RealtimeEventType.TRANSCRIPT_FINAL, text="hello", provider_event_id="seg-1")

    assert normalize_elevenlabs_realtime_stt_event({"type": "vad_speech_start"}) == RealtimeEvent(
        RealtimeEventType.SPEECH_STARTED
    )
    assert normalize_elevenlabs_realtime_stt_event({"type": "vad_speech_end"}) == RealtimeEvent(
        RealtimeEventType.END_OF_TURN
    )


def test_orchestrator_submits_only_final_transcripts_and_surfaces_partials():
    partials = []
    submitted = []

    orch = RealtimeVoiceOrchestrator(
        submit_user_turn=submitted.append,
        publish_event=lambda event: partials.append(event),
    )

    orch.handle_event(RealtimeEvent(RealtimeEventType.TRANSCRIPT_PARTIAL, text="draft"))
    orch.handle_event(RealtimeEvent(RealtimeEventType.TRANSCRIPT_FINAL, text="final text"))

    assert submitted == ["final text"]
    assert [event.type for event in partials] == [RealtimeEventType.TRANSCRIPT_PARTIAL]


def test_orchestrator_barge_interrupts_generation_cancels_tts_and_preserves_final_text():
    calls = []

    class _Agent:
        def interrupt(self):
            calls.append("interrupt")

    orch = RealtimeVoiceOrchestrator(
        submit_user_turn=lambda text: calls.append(("submit", text)),
        publish_event=lambda event: calls.append(("event", event.type.value)),
        interrupt_active_turn=_Agent().interrupt,
        cancel_tts=lambda: calls.append("cancel_tts"),
        generation_active=lambda: True,
    )

    orch.handle_event(RealtimeEvent(RealtimeEventType.BARGE_IN, text="excuse"))
    orch.handle_event(RealtimeEvent(RealtimeEventType.TRANSCRIPT_FINAL, text="excuse me"))

    assert calls[:3] == ["interrupt", "cancel_tts", ("event", "barge_in")]
    assert calls[-1] == ("submit", "excuse me")


def test_orchestrator_provider_error_cleans_up_and_requests_fallback():
    calls = []

    orch = RealtimeVoiceOrchestrator(
        submit_user_turn=lambda text: calls.append(("submit", text)),
        publish_event=lambda event: calls.append(("event", event.type.value)),
        cleanup_provider=lambda: calls.append("cleanup"),
        fallback_to_batch_stt=lambda: calls.append("fallback"),
    )

    orch.handle_event(RealtimeEvent(RealtimeEventType.ERROR, message="provider down"))

    assert calls == ["cleanup", "fallback", ("event", "error")]


def test_openai_realtime_broker_uses_profile_secret_scope_and_restricted_payload(monkeypatch):
    captured = {}

    def fake_secret():
        captured["resolver"] = True
        return "voice-key"

    def fake_request(api_key, body, timeout):
        captured["api_key"] = api_key
        captured["body"] = body
        captured["timeout"] = timeout
        return {"client_secret": {"value": "eph", "expires_at": 123456}}

    monkeypatch.setattr("agent.voice_realtime.resolve_openai_audio_api_key", fake_secret)

    result = broker_openai_realtime_transcription_session(
        request_client_secret=fake_request,
        session_id="sid",
        model="gpt-4o-transcribe",
        language="en",
        profile="work",
    )

    assert result["client_secret"] == "eph"
    assert result["expires_at"] == 123456
    assert captured["api_key"] == "voice-key"
    assert captured["body"]["session"]["type"] == "transcription"
    assert "tools" not in captured["body"]["session"]
    assert captured["body"]["session"]["input_audio_transcription"]["model"] == "gpt-4o-transcribe"
    assert captured["body"]["session"]["input_audio_transcription"]["language"] == "en"
    assert captured["body"]["session"]["turn_detection"]["type"] == "server_vad"


def test_openai_realtime_broker_rejects_missing_key(monkeypatch):
    monkeypatch.setattr("agent.voice_realtime.resolve_openai_audio_api_key", lambda: "")
    with pytest.raises(ValueError, match="OpenAI Realtime transcription is not configured"):
        broker_openai_realtime_transcription_session(
            request_client_secret=lambda *_args, **_kwargs: {},
            session_id="sid",
            model="gpt-4o-transcribe",
        )


def test_elevenlabs_realtime_broker_returns_backend_bridge_without_api_key(monkeypatch):
    captured = {}

    monkeypatch.setattr("agent.voice_realtime.resolve_provider_secret", lambda env, provider: "eleven-key")

    result = broker_elevenlabs_realtime_stt_bridge_session(
        session_id="sid",
        profile="work",
        model_id="scribe_v2_realtime",
        language="en",
        base_ws_url="ws://testserver",
    )

    assert result["ok"] is True
    assert result["provider"] == "elevenlabs"
    assert result["bridge_url"].startswith("ws://testserver/api/audio/realtime/elevenlabs/ws?")
    assert "eleven-key" not in result["bridge_url"]
    assert result["session_binding"]
    assert result["expires_at"] > 0


def test_elevenlabs_realtime_bridge_token_is_single_use(monkeypatch):
    monkeypatch.setattr("agent.voice_realtime.resolve_provider_secret", lambda env, provider: "eleven-key")

    result = broker_elevenlabs_realtime_stt_bridge_session(
        session_id="sid",
        profile="work",
        model_id="scribe_v2_realtime",
        language="en",
        base_ws_url="ws://testserver",
    )
    token = parse_qs(urlparse(result["bridge_url"]).query)["token"][0]

    first = consume_elevenlabs_realtime_stt_bridge_token(token)
    second = consume_elevenlabs_realtime_stt_bridge_token(token)

    assert first is not None
    assert first["api_key"] == "eleven-key"
    assert second is None
