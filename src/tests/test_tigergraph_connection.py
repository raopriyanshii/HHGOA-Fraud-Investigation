"""
scrub_secret is a security-relevant function -- it's the only thing
standing between a raw library exception (which can contain a partially
masked token, as pyTigerGraph's own error formatting does) and this
project's terminal output. It gets a real regression test rather than
being trusted by inspection alone.
"""
import os

from src.graph.tigergraph_connection import scrub_secret

FAKE_SECRET = "lpa6xxxxxxxxxxxxxxxxxxxxxxxxu1v"


def _with_fake_secret(monkeypatch):
    monkeypatch.setenv("TG_GSQL_SECRET", FAKE_SECRET)


def test_full_secret_is_redacted(monkeypatch):
    _with_fake_secret(monkeypatch)
    msg = f"full secret leaked: {FAKE_SECRET} end"
    assert FAKE_SECRET not in scrub_secret(msg)
    assert "[REDACTED]" in scrub_secret(msg)


def test_librarys_own_partial_mask_pattern_is_redacted(monkeypatch):
    _with_fake_secret(monkeypatch)
    # exactly the pattern pyTigerGraph produced in practice: a few leading
    # and trailing characters of the secret, separated by asterisks
    msg = "Access Denied because the input token = 'lpa6****u1v' is not in correct format"
    scrubbed = scrub_secret(msg)
    assert "lpa6" not in scrubbed
    assert "u1v" not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_separate_prefix_and_suffix_fragments_are_both_redacted(monkeypatch):
    _with_fake_secret(monkeypatch)
    msg = "prefix lpa6xxx and suffix xxxu1v appear separately"
    scrubbed = scrub_secret(msg)
    assert "lpa6" not in scrubbed
    assert "u1v" not in scrubbed


def test_message_with_no_secret_fragment_is_unchanged():
    assert scrub_secret("connection refused") == "connection refused"


def test_missing_env_var_does_not_crash(monkeypatch):
    monkeypatch.delenv("TG_GSQL_SECRET", raising=False)
    assert scrub_secret("some error") == "some error"


# --- generic credential redaction, independent of TG_GSQL_SECRET ---
# Added after a real incident: a freshly-minted JWT bearer token appeared
# in a raw response body and was printed in full, because scrub_secret at
# the time only knew about the configured secret. These tests lock in the
# fix so this class of leak can't silently regress.

FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImlhdCI6MTIzfQ.fakefakefakefakefakefakefake"


def test_jwt_token_is_redacted_anywhere_in_text():
    msg = f'{{"token":"{FAKE_JWT}"}}'
    scrubbed = scrub_secret(msg)
    assert FAKE_JWT not in scrubbed
    assert "[REDACTED" in scrubbed


def test_bare_jwt_with_no_json_context_is_redacted():
    msg = f"here is a token: {FAKE_JWT} end of message"
    scrubbed = scrub_secret(msg)
    assert FAKE_JWT not in scrubbed


def test_authorization_header_is_redacted():
    msg = "Authorization: Bearer abcdef1234567890.somepayload-here_ok"
    scrubbed = scrub_secret(msg)
    assert "abcdef1234567890" not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_json_secret_and_password_fields_are_redacted():
    msg = '{"secret": "verysecretvalue123", "password": "hunter2hunter2", "apiToken": "abcd1234efgh5678"}'
    scrubbed = scrub_secret(msg)
    assert "verysecretvalue123" not in scrubbed
    assert "hunter2hunter2" not in scrubbed
    assert "abcd1234efgh5678" not in scrubbed


def test_set_cookie_header_is_redacted():
    msg = "Set-Cookie: session=abc123def456; Path=/; HttpOnly"
    scrubbed = scrub_secret(msg)
    assert "abc123def456" not in scrubbed


def test_ordinary_hostname_with_multiple_dots_is_not_over_redacted():
    # a real TG_HOST value from this project -- must survive unredacted,
    # since it's not a secret and diagnostics need to show it
    host = "https://tg-aba9637e-715a-4340-977a-4f9e68733f92.tg-2635877100.i.tgcloud.io"
    assert scrub_secret(host) == host
