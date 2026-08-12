from hosting.engine_host_channel import EngineHostControlChannel


def test_shared_secret_cache_preserves_complete_authentication_result(monkeypatch) -> None:
    channel = EngineHostControlChannel({"engine_host_key_id": "worker-key"})
    monkeypatch.setattr(channel, "_auto_session_cache_key", lambda: "test-cache-key")
    issued = {
        "token": "session-token",
        "role": "worker_user",
        "auth_method": "shared_secret",
        "scope": "control",
        "key_id": "worker-key",
        "expires_at": 4102444800.0,
    }
    channel._store_cached_auto_session("session-token", issued)
    cached = channel._get_cached_auto_session()
    assert cached is not None
    channel.set_session_token(cached["token"])
    channel._set_session_token_meta(cached)
    assert channel.get_session_metadata() == issued


def test_empty_channel_has_no_narrowed_authentication_result() -> None:
    channel = EngineHostControlChannel({})
    assert channel.get_session_metadata() == {}


def test_public_key_cache_returns_structured_metadata(monkeypatch) -> None:
    channel = EngineHostControlChannel({})
    monkeypatch.setattr(channel, "_public_key_session_cache_key", lambda **_kwargs: "public-cache-key")
    monkeypatch.setattr(channel, "_current_ssh_session_binding", lambda: None)
    issued = {"token": "public-token", "role": "admin", "expires_at": 4102444800.0}
    channel._store_cached_public_key_session(
        "public-token", issued, key_id="admin-key", scope="control", bind_to_ssh=False
    )
    cached = channel._get_cached_public_key_session(
        key_id="admin-key", scope="control", bind_to_ssh=False
    )
    assert cached is not None
    assert cached["token"] == "public-token"
    assert cached["auth_method"] == "public_key"
    assert cached["key_id"] == "admin-key"
    assert cached["scope"] == "control"
