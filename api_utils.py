"""
API layer for MedAgents, ported from the pre-1.0 OpenAI SDK (Azure) to the
modern `openai>=1.0` client, pointed at an OpenAI-compatible endpoint
(Groq by default).

Only this file was ported. Every LLM call in the pipeline funnels through
`api_handler.get_output_multiagent`, whose signature is unchanged, so
`utils.py` / `prompt_generator.py` / the harness logic are untouched.

Environment variables:
  GROQ_API_KEY         - required, the API key.
  BASE_URL             - OpenAI-compatible base url. Default: Groq.
  REQUEST_TIMEOUT      - per-request timeout in seconds. Default: 120.
  MAX_RETRIES          - max attempts on a *retryable* error. Default: 6.
  MEDAGENTS_USAGE_LOG  - JSONL file for per-call usage. Default: ./outputs/usage_log.jsonl
"""

import os
import re
import sys
import time
import json
import random
import datetime

from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
    APIError,
)

# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("BASE_URL", "https://api.groq.com/openai/v1")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))
USAGE_LOG = os.environ.get("MEDAGENTS_USAGE_LOG", "./outputs/usage_log.jsonl")

_api_key = os.environ.get("GROQ_API_KEY", "")
if not _api_key:
    print(
        "[api_utils] WARNING: GROQ_API_KEY is not set in the environment. "
        "API calls will fail with an auth error.",
        file=sys.stderr,
    )

client = OpenAI(api_key=_api_key, base_url=BASE_URL)

# --------------------------------------------------------------------------- #
# Rate-limit classification
#
# Groq 429 messages look like:
#   "Rate limit reached for model `...` ... on tokens per minute (TPM): Limit 8000 ...
#    Please try again in 7.2s."
#   "Rate limit reached ... on requests per day (RPD): Limit 1000, Used 1000 ..."
#
# Per-minute (TPM/RPM): transient -> wait and retry.
# Per-day   (TPD/RPD): quota is exhausted until it resets -> abort, do NOT retry.
# --------------------------------------------------------------------------- #
_DAILY_MARKERS = ("per day", "(tpd)", "(rpd)", "tpd:", "rpd:", "tokens per day", "requests per day")
_MINUTE_MARKERS = ("per minute", "(tpm)", "(rpm)", "tpm:", "rpm:", "tokens per minute", "requests per minute")


def _is_daily_limit(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _DAILY_MARKERS)


def _retry_after_seconds(err, attempt: int) -> float:
    """How long to wait before a retry. Prefer the server's hint, else backoff."""
    # 1) Retry-After header
    try:
        ra = err.response.headers.get("retry-after")
        if ra:
            return min(float(ra) + 0.5, 65.0)
    except Exception:
        pass
    # 2) "Please try again in 7.2s" (also handles "in 1m2.5s")
    msg = str(getattr(err, "message", "") or "") + " " + str(err)
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg.lower())
    if m:
        minutes = float(m.group(1)) if m.group(1) else 0.0
        seconds = float(m.group(2))
        return min(minutes * 60 + seconds + 0.5, 65.0)
    # 3) exponential backoff with jitter
    return min(2 ** attempt, 30.0) + random.uniform(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Per-call usage logging
# --------------------------------------------------------------------------- #
def _extract_usage(resp):
    try:
        u = resp.usage
        return (
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            getattr(u, "total_tokens", None),
        )
    except Exception:
        return None, None, None


def append_usage_record(record):
    """Append one JSON record (an API call, a vote, ...) to USAGE_LOG. Best-effort;
    never lets a logging failure break a run."""
    try:
        d = os.path.dirname(USAGE_LOG)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(USAGE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[usage] WARNING: could not write usage log: {e}", file=sys.stderr)


def _log_usage(model, prompt_tokens, completion_tokens, total_tokens, latency, status="ok"):
    print(
        f"[usage] model={model} prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} total_tokens={total_tokens} "
        f"latency={latency:.2f}s status={status}"
    )
    append_usage_record({
        "event": "api_call",
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_s": round(latency, 3),
        "status": status,
    })


# --------------------------------------------------------------------------- #
# Raw API calls
# --------------------------------------------------------------------------- #
def generate_response_multiagent(
    engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input
):
    print("Generating response for model: ", engine)
    start_time = time.time()
    # Always send [system, user] to match the original harness behaviour,
    # even when system_role is "".
    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_input},
    ]
    response = client.chat.completions.create(
        model=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        messages=messages,
        timeout=REQUEST_TIMEOUT,
    )
    latency = time.time() - start_time
    pt, ct, tt = _extract_usage(response)
    _log_usage(engine, pt, ct, tt, latency)
    print("Finish! Time taken: ", latency)
    return response


def generate_response(
    engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text
):
    """Single user-message variant. Not used by the harness; kept for completeness."""
    print("Generating response for model: ", engine)
    start_time = time.time()
    response = client.chat.completions.create(
        model=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        messages=[{"role": "user", "content": input_text}],
        timeout=REQUEST_TIMEOUT,
    )
    latency = time.time() - start_time
    pt, ct, tt = _extract_usage(response)
    _log_usage(engine, pt, ct, tt, latency)
    print("Finish! Time taken: ", latency)
    return response


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
# Legacy Azure deployment aliases, kept for backward compatibility. Any name not
# listed here is treated as a literal model id and passed straight to the API
# (this is how Groq catalogue ids such as `llama-3.3-70b-versatile` flow through).
_LEGACY_ALIASES = {
    "instructgpt": "text-davinci-002",
    "instructgpt-gen": "text-davinci-002",
    "newinstructgpt": "text-davinci-003",
    "oldinstructgpt": "text-davinci-001",
    "gpt3": "davinci",
    "codex": "code-davinci-002",
    "gpt3-edit": "text-davinci-edit-001",
    "codex-edit": "code-davinci-edit-001",
    "chatgpt": "gpt-35-turbo-16k",
    "gpt4": "gpt-4",
}


class api_handler:
    def __init__(self, model):
        self.model = model
        self.engine = _LEGACY_ALIASES.get(model, model)

    def _abort_daily(self, err):
        print(
            "\n==================== DAILY QUOTA EXHAUSTED ====================\n"
            f"Model: {self.engine}\n"
            f"Server said: {err}\n"
            "This is a per-DAY rate limit (TPD/RPD). Waiting cannot help; the\n"
            "quota resets at the start of the next UTC day. Aborting now so the\n"
            "retry loop does not spin against an exhausted daily budget.\n"
            "Any items completed before this point were already written to the\n"
            "output file.\n"
            "===============================================================\n",
            file=sys.stderr,
        )
        _log_usage(self.engine, None, None, None, 0.0, status="daily_limit_abort")
        sys.exit(2)

    def get_output_multiagent(
        self,
        system_role,
        user_input,
        max_tokens,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    ):
        for attempt in range(MAX_RETRIES):
            try:
                response = generate_response_multiagent(
                    self.engine,
                    temperature,
                    max_tokens,
                    frequency_penalty,
                    presence_penalty,
                    stop,
                    system_role,
                    user_input,
                )
                if response.choices and response.choices[0].message is not None:
                    content = response.choices[0].message.content
                    return content if content else "ERROR."
                return "ERROR."

            except RateLimitError as error:
                text = str(getattr(error, "message", "") or "") + " " + str(error)
                if _is_daily_limit(text):
                    self._abort_daily(error)  # exits the process
                wait = _retry_after_seconds(error, attempt)
                print(
                    f"[rate-limit] per-minute limit hit (attempt {attempt+1}/{MAX_RETRIES}); "
                    f"waiting {wait:.1f}s then retrying.",
                    file=sys.stderr,
                )
                if attempt == MAX_RETRIES - 1:
                    print("[rate-limit] exhausted retries on a per-minute limit.", file=sys.stderr)
                    return "ERROR."
                time.sleep(wait)

            except (APITimeoutError, APIConnectionError) as error:
                wait = min(2 ** attempt, 30.0) + random.uniform(0.0, 1.0)
                print(
                    f"[network] {type(error).__name__} (attempt {attempt+1}/{MAX_RETRIES}); "
                    f"waiting {wait:.1f}s then retrying.",
                    file=sys.stderr,
                )
                if attempt == MAX_RETRIES - 1:
                    return "ERROR."
                time.sleep(wait)

            except APIStatusError as error:
                # Some providers surface a daily 429 as a generic status error.
                status = getattr(error, "status_code", None)
                text = str(getattr(error, "message", "") or "") + " " + str(error)
                if status == 429 and _is_daily_limit(text):
                    self._abort_daily(error)
                if status == 429:
                    wait = _retry_after_seconds(error, attempt)
                    print(
                        f"[rate-limit] 429 (attempt {attempt+1}/{MAX_RETRIES}); "
                        f"waiting {wait:.1f}s then retrying.",
                        file=sys.stderr,
                    )
                    if attempt == MAX_RETRIES - 1:
                        return "ERROR."
                    time.sleep(wait)
                else:
                    # 4xx other than 429 (e.g. bad request) won't fix on retry.
                    print(f"[api] non-retryable status {status}: {error}", file=sys.stderr)
                    return "ERROR."

            except Exception as error:
                print(
                    f"[api] attempt {attempt+1}/{MAX_RETRIES} failed: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                if attempt == MAX_RETRIES - 1:
                    return "ERROR."
                time.sleep(min(2 ** attempt, 15.0) + random.uniform(0.0, 1.0))

        return "ERROR."

    def get_output(
        self,
        input_text,
        max_tokens,
        temperature=0,
        suffix=None,
        stop=None,
        do_tunc=False,
        echo=False,
        ban_pronoun=False,
        frequency_penalty=0,
        presence_penalty=0,
        return_prob=False,
    ):
        """Single-prompt helper. Not used by the harness; ported for completeness."""
        try:
            response = generate_response(
                self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text
            )
        except Exception:
            print("Error during generation; retrying once.", file=sys.stderr)
            try:
                response = generate_response(
                    self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text
                )
            except Exception:
                print("Second failure. Returning ERROR.", file=sys.stderr)
                return "ERROR."

        if response.choices and response.choices[0].message is not None and response.choices[0].message.content:
            x = response.choices[0].message.content
        else:
            print(response)
            return "ERROR."

        if do_tunc:
            y = x.strip()
            for marker in ("\n", "Q:", "Question:"):
                if marker in y:
                    y = y[: y.find(marker)]
            assert "\n" not in y
            if not return_prob:
                return y

        if not return_prob:
            return x

        output_token_offset_real, output_token_tokens_real, output_token_probs_real = [], [], []
        return x, (output_token_offset_real, output_token_tokens_real, output_token_probs_real)
