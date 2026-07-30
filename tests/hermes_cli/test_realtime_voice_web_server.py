from fastapi.testclient import TestClient


def _client():
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_realtime_voice_session_disabled_by_default(monkeypatch, _isolate_hermes_home):
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(web_server, "load_config", lambda: {"voice": {"input_mode": "legacy"}})

    resp = _client().post(
        "/api/audio/realtime/session",
        json={"session_id": "session-1"},
    )

    assert resp.status_code == 409


def test_realtime_voice_session_uses_profile_scoped_openai_broker(monkeypatch, _isolate_hermes_home):
    from contextlib import contextmanager

    import hermes_cli.web_server as web_server
    import agent.voice_realtime as voice_realtime

    captured = {}

    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {
            "voice": {
                "input_mode": "realtime",
                "realtime": {
                    "enabled": True,
                    "stt_provider": "openai",
                    "transcription_model": "gpt-4o-transcribe",
                    "language": "",
                },
            },
        },
    )

    @contextmanager
    def fake_scope(_profile):
        yield None

    monkeypatch.setattr(web_server, "_config_profile_scope", fake_scope)

    def fake_broker(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "provider": "openai",
            "client_secret": "eph",
            "expires_at": 123,
            "session_binding": "binding",
        }

    monkeypatch.setattr(voice_realtime, "broker_openai_realtime_transcription_session", fake_broker)

    resp = _client().post(
        "/api/audio/realtime/session?profile=work",
        json={"session_id": "session-1", "language": "en"},
    )

    assert resp.status_code == 200
    assert resp.json()["client_secret"] == "eph"
    assert captured["profile"] == "work"
    assert captured["session_id"] == "session-1"
    assert captured["language"] == "en"


def test_realtime_voice_session_uses_profile_scoped_elevenlabs_bridge(monkeypatch, _isolate_hermes_home):
    from contextlib import contextmanager

    import hermes_cli.web_server as web_server
    import agent.voice_realtime as voice_realtime

    captured = {}

    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {
            "voice": {
                "input_mode": "realtime",
                "realtime": {"enabled": True, "stt_provider": "elevenlabs"},
            },
        },
    )

    @contextmanager
    def fake_scope(_profile):
        yield None

    monkeypatch.setattr(web_server, "_config_profile_scope", fake_scope)

    def fake_broker(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "provider": "elevenlabs",
            "bridge_url": "ws://testserver/api/audio/realtime/elevenlabs/ws?token=bridge",
            "expires_at": 123,
            "session_binding": "binding",
        }

    monkeypatch.setattr(voice_realtime, "broker_elevenlabs_realtime_stt_bridge_session", fake_broker)

    resp = _client().post(
        "/api/audio/realtime/session?profile=work",
        json={"session_id": "session-1", "language": "en"},
    )

    assert resp.status_code == 200
    assert resp.json()["provider"] == "elevenlabs"
    assert resp.json()["bridge_url"].startswith("ws://testserver/")
    assert "client_secret" not in resp.json()
    assert captured["profile"] == "work"
    assert captured["session_id"] == "session-1"
    assert captured["language"] == "en"
