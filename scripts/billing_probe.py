#!/usr/bin/env python3
"""
Billing-reachability probe.

Goal: show that each configured API key *authenticates* but may still lack
usable (free) quota. For every platform we:

  1. Hit the provider's /models endpoint (where reachable) to confirm the
     model id -- we do not guess ids; unconfirmed ones are reported.
  2. Send ONE minimal chat completion via the OpenAI-compatible SDK.

Everything is recorded: platform, base_url, model, http_status, error_type,
the verbatim error message, and whether a completion came back. Every
exception is caught -- a failure is a RESULT, not an error -- and we move on
to the next platform. No retries on 4xx (max_retries=0). No key is ever
printed or written to disk.

Reads each key from its own env var. Writes outputs/billing_probe.json and
prints a summary table.
"""

import os
import sys
import json

from openai import OpenAI
import openai as openai_mod

# Each key is read from its own env var; never hardcoded.
PLATFORMS = [
    {
        "platform": "OPENAI",
        "env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-luna",
    },
    {
        "platform": "ANTHROPIC",
        "env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/",
        "model": "claude-sonnet-5-5",   # "Sonnet 5.5" id; confirmed against /models below
    },
    {
        "platform": "DEEPSEEK",
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    {
        "platform": "GEMINI",
        "env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
    },
]

OUT_JSON = os.environ.get("BILLING_PROBE_OUT", "./outputs/billing_probe.json")
TIMEOUT = float(os.environ.get("BILLING_PROBE_TIMEOUT", "30"))


def confirm_model(client, want_model):
    """Return (confirmed, candidates, error).

    confirmed: True if want_model is in /models, False if reachable but absent,
               None if /models could not be reached.
    candidates: a few ids that look related (substring overlap), for reporting.
    """
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

    if want_model in ids:
        return True, [want_model], None

    # Not an exact match -- surface a few plausible neighbours for the report.
    key = want_model.split("-")[0].lower()
    cands = [i for i in ids if key in i.lower()][:8]
    return False, cands, None


def probe_chat(client, model):
    rec = {
        "http_status": None,
        "error_type": None,
        "error_message": None,
        "completion_returned": False,
        "completion_text": None,
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        rec["http_status"] = 200
        rec["completion_returned"] = True
        try:
            rec["completion_text"] = resp.choices[0].message.content
        except Exception:
            rec["completion_text"] = None
    except openai_mod.APIStatusError as e:
        # 4xx/5xx with an HTTP response (covers 401, 403, 429, 404, ...).
        rec["http_status"] = getattr(e, "status_code", None)
        code = getattr(e, "code", None)
        rec["error_type"] = type(e).__name__ + (f"/{code}" if code else "")
        rec["error_message"] = getattr(e, "message", None) or str(e)
    except openai_mod.APIConnectionError as e:
        rec["error_type"] = type(e).__name__
        rec["error_message"] = str(e)
    except Exception as e:
        rec["error_type"] = type(e).__name__
        rec["error_message"] = str(e)
    return rec


def main():
    results = []
    for p in PLATFORMS:
        rec = {
            "platform": p["platform"],
            "base_url": p["base_url"],
            "model": p["model"],
            "model_confirmed": None,
            "model_candidates": None,
            "models_endpoint_error": None,
            "http_status": None,
            "error_type": None,
            "error_message": None,
            "completion_returned": False,
            "completion_text": None,
        }

        key = os.environ.get(p["env"], "")
        if not key:
            rec["error_type"] = "missing_key"
            rec["error_message"] = f"env var {p['env']} is not set"
            results.append(rec)
            print(f"[{p['platform']}] SKIP: {p['env']} not set", file=sys.stderr)
            continue

        try:
            client = OpenAI(
                api_key=key,
                base_url=p["base_url"],
                max_retries=0,      # do NOT retry on 4xx
                timeout=TIMEOUT,
            )
        except Exception as e:
            rec["error_type"] = type(e).__name__
            rec["error_message"] = f"client init failed: {e}"
            results.append(rec)
            continue

        # 1) confirm the model id against /models (best-effort)
        confirmed, cands, merr = confirm_model(client, p["model"])
        rec["model_confirmed"] = confirmed
        rec["model_candidates"] = cands
        rec["models_endpoint_error"] = merr

        # 2) one minimal chat completion
        chat = probe_chat(client, p["model"])
        rec.update(chat)

        results.append(rec)
        print(
            f"[{p['platform']}] status={rec['http_status']} "
            f"completion={rec['completion_returned']} "
            f"model_confirmed={rec['model_confirmed']} "
            f"err={rec['error_type']}",
            file=sys.stderr,
        )

    # write JSON
    d = os.path.dirname(OUT_JSON)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # summary table
    print("\n=== BILLING REACHABILITY SUMMARY ===")
    hdr = f"{'platform':<10} {'status':>6} {'compl':>6} {'model_ok':>9}  {'error_type':<28} message"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        msg = (r["error_message"] or "")
        msg = msg.replace("\n", " ")
        if len(msg) > 70:
            msg = msg[:67] + "..."
        print(
            f"{r['platform']:<10} {str(r['http_status']):>6} "
            f"{str(r['completion_returned']):>6} {str(r['model_confirmed']):>9}  "
            f"{str(r['error_type'] or ''):<28} {msg}"
        )
    print(f"\nWrote {OUT_JSON}")

    # Flag any unconfirmed model ids explicitly.
    unconfirmed = [r for r in results if r["model_confirmed"] is not True
                   and r["error_type"] != "missing_key"]
    if unconfirmed:
        print("\nModel ids NOT confirmed against /models:")
        for r in unconfirmed:
            extra = ""
            if r["models_endpoint_error"]:
                extra = f" (/models unreachable: {r['models_endpoint_error']})"
            elif r["model_candidates"]:
                extra = f" (nearby ids: {r['model_candidates']})"
            print(f"  - {r['platform']}: '{r['model']}'{extra}")


if __name__ == "__main__":
    main()
