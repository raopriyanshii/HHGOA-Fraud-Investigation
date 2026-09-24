"""
The one place a TigerGraphConnection is constructed. Reads credentials
from environment variables (via .env, gitignored) and never logs the
secret/token -- every caller in this project must go through get_connection()
rather than building its own connection object.
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from pyTigerGraph import TigerGraphConnection

load_dotenv()


def scrub_secret(text: str) -> str:
    """Redact every credential-shaped thing from a string before it is ever
    printed or logged -- not just the configured secret. This project has
    twice leaked a credential through a path this function didn't cover
    (a partially-masked secret fragment in a library error message, then a
    freshly-minted JWT bearer token in a raw response body): the fix each
    time was to widen this function, never to trust a call site to be
    careful. Every place in this project that prints an exception message,
    a response body, or any diagnostic text MUST pass it through this
    first, with no exceptions.

    Covers, generically (not just today's known secret value):
      - every configured secret env var this project uses (TG_GSQL_SECRET,
        and G8's GROQ_API_KEY/GEMINI_API_KEY), including partial-fragment
        reveals
      - JWT-shaped tokens (three dot-separated base64url segments) anywhere
        in the text, regardless of context
      - "Authorization: Bearer ..." headers, any casing/quoting
      - common credential-bearing JSON fields: token, apiToken, jwtToken,
        secret, gsqlSecret, password, authorization, cookie
      - Set-Cookie / Cookie header lines
    """
    scrubbed = text

    # G8 note: a real Groq/Gemini API key is a bare alphanumeric string
    # with no JWT shape and no "Authorization: Bearer"/named-JSON-field
    # context of its own -- confirmed empirically that none of the
    # pattern-based rules below would ever catch one if an SDK exception
    # ever embedded it raw. Covered the same way TG_GSQL_SECRET already
    # was: exact-value plus fragment matching, per configured secret. Both
    # providers coexist in this project (src/agent/llm_providers/), so
    # both keys are covered here.
    for secret_env_var in ("TG_GSQL_SECRET", "GROQ_API_KEY", "GEMINI_API_KEY"):
        secret = os.environ.get(secret_env_var, "")
        if secret and len(secret) >= 3:
            scrubbed = scrubbed.replace(secret, "[REDACTED]")
            # catch the library's own "'xxxx****yyy'" masked-token pattern FIRST,
            # before the fragment loop below can partially rewrite it and break
            # the regex's assumption of alphanumeric runs on both sides of '***'
            scrubbed = re.sub(r"'[A-Za-z0-9]{2,8}\*{2,}[A-Za-z0-9]{2,8}'", "'[REDACTED]'", scrubbed)
            # then catch any remaining run of 3+ characters from the secret
            # (prefix or suffix) that appears verbatim anywhere else
            for length in range(min(8, len(secret)), 2, -1):
                for fragment in (secret[:length], secret[-length:]):
                    scrubbed = scrubbed.replace(fragment, "[REDACTED]")

    # JWTs: header.payload.signature, each segment base64url, no context needed
    scrubbed = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED_JWT]", scrubbed)
    # any three-dot-separated base64url-looking segments (covers non-"eyJ"-prefixed tokens too)
    scrubbed = re.sub(r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "[REDACTED_JWT]", scrubbed)

    # Authorization headers in any casing/quoting
    scrubbed = re.sub(
        r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(bearer\s+)?[A-Za-z0-9_.\-]{8,}",
        r"\1[REDACTED]",
        scrubbed,
    )

    # common credential-bearing JSON fields: "token": "...", "secret": "...", etc.
    scrubbed = re.sub(
        r'(?i)("(?:token|apiToken|jwtToken|secret|gsqlSecret|password|cookie)"\s*:\s*")[^"]*(")',
        r"\1[REDACTED]\2",
        scrubbed,
    )

    # Set-Cookie / Cookie header lines
    scrubbed = re.sub(r"(?im)^(set-cookie|cookie):.*$", r"\1: [REDACTED]", scrubbed)

    return scrubbed


def get_connection(graphname: str | None = None) -> TigerGraphConnection:
    """graphname defaults to TG_GRAPH_NAME from .env; pass "" explicitly
    for a graph-less/global-scope connection (e.g. for the smoke test,
    or for listing catalog contents without touching any specific graph).
    """
    host = os.environ["TG_HOST"]
    secret = os.environ["TG_GSQL_SECRET"]
    tg_cloud = os.environ.get("TG_CLOUD", "false").lower() == "true"
    resolved_graphname = os.environ.get("TG_GRAPH_NAME", "") if graphname is None else graphname

    return TigerGraphConnection(
        host=host,
        graphname=resolved_graphname,
        gsqlSecret=secret,
        tgCloud=tg_cloud,
    )
