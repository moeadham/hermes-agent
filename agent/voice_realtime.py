"""Provider-neutral realtime voice orchestration contracts.

This module keeps realtime voice as a transport/capability layer. Hermes still
owns conversation state, turn submission, interruption, tools, memory, and TTS
provider selection.
"""

from __future__ import annotations

import enum
import hmac
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from tools.tool_backend_helpers import resolve_provider_secret
from tools.tool_backend_helpers import resolve_openai_audio_api_key


class RealtimeCapability(enum.Enum):
    MIC_AUDIO_INPUT = "mic_audio_input"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    PROVIDER_VAD = "provider_vad"
    SEMANTIC_END_OF_TURN = "semantic_end_of_turn"
    PROVIDER_BARGE_IN = "provider_barge_in"
    CONTINUOUS_TTS_INPUT = "continuous_tts_input"
    STREAMING_TTS_AUDIO = "streaming_tts_audio"


class RealtimeEventType(enum.Enum):
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    SPEECH_STARTED = "speech_started"
    END_OF_TURN = "end_of_turn"
    BARGE_IN = "barge_in"
    AUDIO_CHUNK = "audio_chunk"
    STATUS = "status"
    ERROR = "error"


@dataclass(frozen=True)
class RealtimeEvent:
    type: RealtimeEventType
    text: str = ""
    provider: str = ""
    provider_event_id: str = ""
    audio: bytes = b""
    message: str = ""
    raw: Optional[Mapping[str, Any]] = field(default=None, compare=False)


@dataclass(frozen=True)
class VoiceProviderCapabilities:
    provider: str
    native: frozenset[RealtimeCapability] = field(default_factory=frozenset)

    def supports(self, capability: RealtimeCapability) -> bool:
        return capability in self.native


_FALLBACKS: Dict[RealtimeCapability, str] = {
    RealtimeCapability.MIC_AUDIO_INPUT: "recording_rms_endpointing",
    RealtimeCapability.TRANSCRIPT_PARTIAL: "no_partials",
    RealtimeCapability.TRANSCRIPT_FINAL: "batch_stt",
    RealtimeCapability.PROVIDER_VAD: "rms_silence_endpointing",
    RealtimeCapability.SEMANTIC_END_OF_TURN: "rms_silence_endpointing",
    RealtimeCapability.PROVIDER_BARGE_IN: "hermes_rms_barge_in",
    RealtimeCapability.CONTINUOUS_TTS_INPUT: "sentence_chunked_tts",
    RealtimeCapability.STREAMING_TTS_AUDIO: "sentence_chunked_tts",
}


@dataclass(frozen=True)
class SelectedVoiceCapabilities:
    stt: VoiceProviderCapabilities
    tts: VoiceProviderCapabilities
    fallbacks: Mapping[RealtimeCapability, str]

    def uses_native(self, capability: RealtimeCapability) -> bool:
        return self.stt.supports(capability) or self.tts.supports(capability)

    def fallback_for(self, capability: RealtimeCapability) -> Optional[str]:
        if self.uses_native(capability):
            return None
        return self.fallbacks.get(capability)


def select_voice_capabilities(
    *,
    stt: VoiceProviderCapabilities,
    tts: VoiceProviderCapabilities,
) -> SelectedVoiceCapabilities:
    """Select native capabilities independently, preserving existing fallbacks."""

    fallbacks = {
        capability: fallback
        for capability, fallback in _FALLBACKS.items()
        if not stt.supports(capability) and not tts.supports(capability)
    }
    return SelectedVoiceCapabilities(stt=stt, tts=tts, fallbacks=fallbacks)


def stt_provider_realtime_capabilities(provider: str) -> VoiceProviderCapabilities:
    """Return native realtime STT capabilities for built-in providers."""

    name = (provider or "").strip().lower()
    if name == "openai":
        return VoiceProviderCapabilities(
            provider="openai",
            native=frozenset({
                RealtimeCapability.MIC_AUDIO_INPUT,
                RealtimeCapability.TRANSCRIPT_PARTIAL,
                RealtimeCapability.TRANSCRIPT_FINAL,
                RealtimeCapability.PROVIDER_VAD,
            }),
        )
    if name == "elevenlabs":
        return VoiceProviderCapabilities(
            provider="elevenlabs",
            native=frozenset({
                RealtimeCapability.MIC_AUDIO_INPUT,
                RealtimeCapability.TRANSCRIPT_PARTIAL,
                RealtimeCapability.TRANSCRIPT_FINAL,
                RealtimeCapability.PROVIDER_VAD,
                RealtimeCapability.SEMANTIC_END_OF_TURN,
            }),
        )
    return VoiceProviderCapabilities(provider=name or "unknown", native=frozenset())


def tts_provider_realtime_capabilities(provider: str) -> VoiceProviderCapabilities:
    """Return native realtime TTS capabilities through the existing TTS resolver."""

    name = (provider or "").strip().lower()
    try:
        from tools.tts_streaming import provider_realtime_tts_capabilities

        caps = provider_realtime_tts_capabilities(name)
    except Exception:
        caps = {}
    native: set[RealtimeCapability] = set()
    if caps.get("streaming_audio_output"):
        native.add(RealtimeCapability.STREAMING_TTS_AUDIO)
    if caps.get("continuous_text_input"):
        native.add(RealtimeCapability.CONTINUOUS_TTS_INPUT)
    return VoiceProviderCapabilities(provider=name or "unknown", native=frozenset(native))


def _event_text(event: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _event_id(event: Mapping[str, Any], *keys: str) -> str:
    return _event_text(event, *keys)


def normalize_openai_realtime_transcription_event(
    event: Mapping[str, Any],
) -> Optional[RealtimeEvent]:
    """Normalize OpenAI Realtime transcription-only events.

    Response-generation events are treated as an error because Hermes must not
    let the realtime provider become a second LLM/tool owner.
    """

    etype = event.get("type") or event.get("message_type")
    if not isinstance(etype, str):
        return None
    if etype.startswith("response.") or "function_call" in etype:
        return RealtimeEvent(
            RealtimeEventType.ERROR,
            message="Realtime transcription sessions must not generate LLM responses",
            raw=event,
        )
    if etype == "input_audio_buffer.speech_started":
        return RealtimeEvent(RealtimeEventType.SPEECH_STARTED, raw=event)
    if etype in {"input_audio_buffer.speech_stopped", "input_audio_buffer.committed"}:
        return RealtimeEvent(RealtimeEventType.END_OF_TURN, raw=event)
    if etype in {
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.partial",
    }:
        text = _event_text(event, "delta", "transcript", "text")
        return RealtimeEvent(
            RealtimeEventType.TRANSCRIPT_PARTIAL,
            text=text,
            provider_event_id=_event_id(event, "item_id", "id"),
            raw=event,
        ) if text else None
    if etype == "conversation.item.input_audio_transcription.completed":
        text = _event_text(event, "transcript", "text")
        return RealtimeEvent(
            RealtimeEventType.TRANSCRIPT_FINAL,
            text=text,
            provider_event_id=_event_id(event, "item_id", "id"),
            raw=event,
        ) if text else None
    if etype in {"conversation.item.input_audio_transcription.failed", "error"}:
        message = _event_text(event, "message", "error") or "Realtime transcription failed"
        return RealtimeEvent(RealtimeEventType.ERROR, message=message, raw=event)
    return None


def normalize_elevenlabs_realtime_stt_event(
    event: Mapping[str, Any],
) -> Optional[RealtimeEvent]:
    """Normalize ElevenLabs Scribe realtime WebSocket events.

    The SDK version pinned by Hermes exposes batch STT and realtime TTS. Scribe
    v2 Realtime is documented as WebSocket with partials, finals, VAD, and
    manual commit; this normalizer accepts those event shapes without claiming
    LLM ownership.
    """

    etype = event.get("type") or event.get("message_type")
    if not isinstance(etype, str):
        return None
    normalized = etype.lower().replace(".", "_")
    if normalized in {"partial_transcript", "transcript_partial", "partial"}:
        text = _event_text(event, "transcript", "text", "delta")
        return RealtimeEvent(
            RealtimeEventType.TRANSCRIPT_PARTIAL,
            text=text,
            provider_event_id=_event_id(event, "segment_id", "id"),
            raw=event,
        ) if text else None
    if normalized in {
        "final_transcript",
        "committed_transcript",
        "committed_transcript_with_timestamps",
        "transcript_final",
        "transcript_committed",
        "final",
    }:
        text = _event_text(event, "transcript", "text")
        return RealtimeEvent(
            RealtimeEventType.TRANSCRIPT_FINAL,
            text=text,
            provider_event_id=_event_id(event, "segment_id", "id"),
            raw=event,
        ) if text else None
    if normalized in {"vad_speech_start", "speech_start", "speech_started"}:
        return RealtimeEvent(RealtimeEventType.SPEECH_STARTED, raw=event)
    if normalized in {"vad_speech_end", "speech_end", "speech_stopped", "commit", "committed"}:
        return RealtimeEvent(RealtimeEventType.END_OF_TURN, raw=event)
    if normalized in {"barge_in", "interruption"}:
        return RealtimeEvent(RealtimeEventType.BARGE_IN, text=_event_text(event, "text"), raw=event)
    if normalized in {
        "auth_error",
        "chunk_size_exceeded",
        "commit_throttled",
        "error",
        "input_error",
        "quota_exceeded",
        "rate_limited",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "transcriber_error",
        "unaccepted_terms",
    }:
        return RealtimeEvent(
            RealtimeEventType.ERROR,
            message=_event_text(event, "message", "error") or "Realtime STT failed",
            raw=event,
        )
    return None


class RealtimeVoiceOrchestrator:
    """Consume normalized realtime events and invoke Hermes-owned seams."""

    def __init__(
        self,
        *,
        submit_user_turn: Callable[[str], None],
        publish_event: Callable[[RealtimeEvent], None],
        interrupt_active_turn: Optional[Callable[[], None]] = None,
        cancel_tts: Optional[Callable[[], None]] = None,
        generation_active: Optional[Callable[[], bool]] = None,
        cleanup_provider: Optional[Callable[[], None]] = None,
        fallback_to_batch_stt: Optional[Callable[[], None]] = None,
    ) -> None:
        self._submit_user_turn = submit_user_turn
        self._publish_event = publish_event
        self._interrupt_active_turn = interrupt_active_turn
        self._cancel_tts = cancel_tts
        self._generation_active = generation_active or (lambda: False)
        self._cleanup_provider = cleanup_provider
        self._fallback_to_batch_stt = fallback_to_batch_stt
        self._seen_finals: set[str] = set()

    def handle_event(self, event: Optional[RealtimeEvent]) -> None:
        if event is None:
            return
        if event.type == RealtimeEventType.TRANSCRIPT_PARTIAL:
            self._publish_event(event)
            return
        if event.type == RealtimeEventType.TRANSCRIPT_FINAL:
            text = event.text.strip()
            if not text:
                return
            key = event.provider_event_id or text
            if key in self._seen_finals:
                return
            self._seen_finals.add(key)
            self._submit_user_turn(text)
            return
        if event.type == RealtimeEventType.BARGE_IN:
            if self._generation_active() and self._interrupt_active_turn is not None:
                self._interrupt_active_turn()
            if self._cancel_tts is not None:
                self._cancel_tts()
            self._publish_event(event)
            return
        if event.type == RealtimeEventType.ERROR:
            if self._cleanup_provider is not None:
                self._cleanup_provider()
            if self._fallback_to_batch_stt is not None:
                self._fallback_to_batch_stt()
            self._publish_event(event)
            return
        if event.type in {
            RealtimeEventType.SPEECH_STARTED,
            RealtimeEventType.END_OF_TURN,
            RealtimeEventType.STATUS,
            RealtimeEventType.AUDIO_CHUNK,
        }:
            self._publish_event(event)


OPENAI_REALTIME_CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"
ELEVENLABS_REALTIME_STT_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


def _default_openai_client_secret_request(
    api_key: str,
    body: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    import requests

    response = requests.post(
        OPENAI_REALTIME_CLIENT_SECRETS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "realtime=v1",
        },
        json=dict(body),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def broker_openai_realtime_transcription_session(
    *,
    session_id: str,
    model: str,
    language: str = "",
    profile: Optional[str] = None,
    ttl_seconds: int = 60,
    vad_threshold: float = 0.5,
    prefix_padding_ms: int = 300,
    silence_duration_ms: int = 500,
    request_client_secret: Callable[[str, Mapping[str, Any], float], Mapping[str, Any]] = (
        _default_openai_client_secret_request
    ),
) -> Dict[str, Any]:
    """Mint an OpenAI transcription-only realtime credential.

    Uses ``resolve_openai_audio_api_key`` so ``VOICE_TOOLS_OPENAI_KEY`` and the
    profile-aware secret scope are honored. The returned credential is the
    short-lived client secret, never the standard API key.
    """

    api_key = resolve_openai_audio_api_key()
    if not api_key:
        raise ValueError("OpenAI Realtime transcription is not configured")

    safe_ttl = max(30, min(int(ttl_seconds or 60), 120))
    transcription: Dict[str, Any] = {"model": model or "gpt-4o-transcribe"}
    if language:
        transcription["language"] = language
    body: Dict[str, Any] = {
        "expires_after": {"anchor": "created_at", "seconds": safe_ttl},
        "session": {
            "type": "transcription",
            "input_audio_transcription": transcription,
            "turn_detection": {
                "type": "server_vad",
                "threshold": max(0.0, min(float(vad_threshold), 1.0)),
                "prefix_padding_ms": max(0, int(prefix_padding_ms)),
                "silence_duration_ms": max(100, int(silence_duration_ms)),
            },
        },
    }
    raw = request_client_secret(api_key, body, 10.0)
    secret = raw.get("client_secret") if isinstance(raw, Mapping) else None
    if isinstance(secret, Mapping):
        value = str(secret.get("value") or "").strip()
        expires_at = int(secret.get("expires_at") or (time.time() + safe_ttl))
    else:
        value = str(raw.get("client_secret") or "").strip() if isinstance(raw, Mapping) else ""
        expires_at = int(raw.get("expires_at") or (time.time() + safe_ttl)) if isinstance(raw, Mapping) else 0
    if not value:
        raise ValueError("OpenAI Realtime did not return a client secret")
    binding_material = f"{profile or ''}|{session_id}|{uuid.uuid4()}".encode("utf-8")
    binding = hmac.new(api_key.encode("utf-8"), binding_material, hashlib.sha256).hexdigest()
    return {
        "ok": True,
        "provider": "openai",
        "client_secret": value,
        "expires_at": expires_at,
        "session_binding": binding,
    }


_ELEVENLABS_BRIDGE_TOKENS: Dict[str, Dict[str, Any]] = {}


def broker_elevenlabs_realtime_stt_bridge_session(
    *,
    session_id: str,
    profile: Optional[str] = None,
    model_id: str = "scribe_v2_realtime",
    language: str = "",
    ttl_seconds: int = 60,
    base_ws_url: str = "",
) -> Dict[str, Any]:
    """Create a backend-owned ElevenLabs realtime STT bridge session.

    ElevenLabs does not provide a browser-safe ephemeral STT credential in the
    shape OpenAI Realtime does. The renderer therefore receives an expiring
    Hermes bridge URL only; the ElevenLabs API key stays in the backend and is
    resolved through the same provider-secret path as the existing voice tools.
    """

    api_key = resolve_provider_secret("ELEVENLABS_API_KEY", "elevenlabs") or ""
    if not api_key:
        raise ValueError("ElevenLabs realtime transcription is not configured")

    safe_ttl = max(30, min(int(ttl_seconds or 60), 120))
    expires_at = int(time.time() + safe_ttl)
    token = uuid.uuid4().hex
    binding_material = f"{profile or ''}|{session_id}|{token}".encode("utf-8")
    binding = hmac.new(api_key.encode("utf-8"), binding_material, hashlib.sha256).hexdigest()
    _ELEVENLABS_BRIDGE_TOKENS[token] = {
        "api_key": api_key,
        "expires_at": expires_at,
        "language": language,
        "model_id": model_id or "scribe_v2_realtime",
        "profile": profile,
        "session_binding": binding,
        "session_id": session_id,
    }
    base = (base_ws_url or "").rstrip("/")
    bridge_path = f"/api/audio/realtime/elevenlabs/ws?token={token}"
    return {
        "ok": True,
        "provider": "elevenlabs",
        "bridge_url": f"{base}{bridge_path}" if base else bridge_path,
        "expires_at": expires_at,
        "session_binding": binding,
    }


def consume_elevenlabs_realtime_stt_bridge_token(token: str) -> Optional[Dict[str, Any]]:
    """Return bridge metadata for a valid token, removing expired entries."""

    now = int(time.time())
    expired = [key for key, value in _ELEVENLABS_BRIDGE_TOKENS.items() if int(value.get("expires_at") or 0) <= now]
    for key in expired:
        _ELEVENLABS_BRIDGE_TOKENS.pop(key, None)

    value = _ELEVENLABS_BRIDGE_TOKENS.pop((token or "").strip(), None)
    if not value:
        return None
    return dict(value)
