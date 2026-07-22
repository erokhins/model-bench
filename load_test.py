#!/usr/bin/env python3
"""
OpenAI Chat API Load Test — interactive UI

Loads model endpoint configs from ~/.junie-local/models/*.json,
presents a Rich-based menu to pick an endpoint and tune parameters,
then runs concurrent load tests and prints statistics.

Uses Python stdlib (urllib, concurrent.futures) plus
transformers (optional, for accurate token counting via HuggingFace tokenizers)
and Rich (for terminal UI).

Usage:
    python load_test.py
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from glob import glob
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window, HSplit, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.shortcuts import choice as pt_choice
from prompt_toolkit.styles import Style

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False

console = Console()

# ---------------------------------------------------------------------------
# Model config loader
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.expanduser("~/.junie-local/models")


def load_model_configs() -> list:
    configs = []
    for path in sorted(glob(os.path.join(MODELS_DIR, "*.json"))):
        try:
            with open(path) as f:
                cfg = json.load(f)
            cfg["_source_file"] = os.path.basename(path)
            configs.append(cfg)
        except Exception as e:
            console.print(f"[yellow]Warning: could not load {path}: {e}[/]")
    return configs


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Cache for tokenizers keyed by base model family
_tokenizer_cache: dict = {}


def _get_tokenizer_for_model(model_id: str):
    """Return a HuggingFace tokenizer for the given model, with caching.

    Falls back to char/4 estimation if transformers is not available
    or the tokenizer cannot be loaded.
    """
    if not HAS_TOKENIZER:
        return None

    # Normalize model ID to find the base tokenizer
    normalized = model_id.lower().replace("-", "").replace("_", "")

    # Check cache first
    if normalized in _tokenizer_cache:
        return _tokenizer_cache[normalized]

    # Try loading the tokenizer for the exact model ID first
    tokenizer = None
    candidates = [model_id]

    # Map known model families to their tokenizer sources
    if "qwen3" in normalized or "qwen2" in normalized:
        candidates.append("Qwen/Qwen2.5-32B")
    elif "llama" in normalized:
        candidates.append("meta-llama/Llama-3.2-3B")
    elif "mistral" in normalized or "mixtral" in normalized:
        candidates.append("mistralai/Mistral-7B-v0.3")

    for candidate in candidates:
        if candidate.lower() in _tokenizer_cache:
            tokenizer = _tokenizer_cache[candidate.lower()]
            break
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            break
        except Exception:
            continue

    _tokenizer_cache[normalized] = tokenizer
    return tokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens using the provided tokenizer, falling back to char/4."""
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text) // 4


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------

FILLER = """
The following is a placeholder paragraph used to build up context length.
It contains natural-sounding but meaningless text designed to consume tokens
without triggering any particular pattern in the model.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore
eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt
in culpa qui officia deserunt mollit anim id est laborum.

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore
veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim
ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia
consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.

Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur,
adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et
dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis
nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex
ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea
voluptate velit esse quam nihil molestiae consequatur.
"""

DEFAULT_ESSAY_WORDS = 2000


def build_prompt(essay_words: int = DEFAULT_ESSAY_WORDS) -> str:
    return (
        f"Please write a detailed essay of approximately {essay_words} words "
        "about the history and impact of artificial intelligence on "
        "modern society, covering topics such as machine learning, "
        "natural language processing, computer vision, robotics, and "
        "the ethical implications of AI development."
    )


def generate_context(target_tokens: int = 30_000, model_id: str = "") -> str:
    tokenizer = _get_tokenizer_for_model(model_id) if model_id else None
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f %Z")
    header = f"[UNIQUE TIMESTAMP: {ts}] This section ensures no prompt caching. "

    if tokenizer is not None:
        header_tokens = len(tokenizer.encode(header))
        filler_tokens = len(tokenizer.encode(FILLER))
        if filler_tokens > 0:
            repeats = max(1, (target_tokens - header_tokens) // filler_tokens)
        else:
            repeats = max(1, (target_tokens * 4 - len(header)) // len(FILLER))
    else:
        target_chars = target_tokens * 4
        repeats = max(1, (target_chars - len(header)) // len(FILLER))
    return header + FILLER * repeats


# ---------------------------------------------------------------------------
# HTTP streaming request (stdlib only)
# ---------------------------------------------------------------------------

def send_request(
    base_url: str,
    model: str,
    context: str,
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    extra_body: dict,
    api_type: str = "OpenAICompletion",
    essay_words: int = DEFAULT_ESSAY_WORDS,
) -> dict:
    prompt = build_prompt(essay_words)
    tokenizer = _get_tokenizer_for_model(model)
    input_tokens = count_tokens(f"{context}\n\n{prompt}", tokenizer)
    start = time.monotonic()

    try:
        if api_type == "OpenAIResponses":
            result = _send_responses_api(
                base_url, model, context, api_key, temperature, max_tokens,
                extra_body, start, prompt, tokenizer
            )
        else:
            result = _send_completion_api(
                base_url, model, context, api_key, temperature, max_tokens,
                extra_body, start, prompt, tokenizer
            )
    except Exception as e:
        return {"error": str(e)}

    if "error" not in result:
        result["input_tokens"] = input_tokens
    return result


def _send_completion_api(
    base_url, model, context, api_key, temperature, max_tokens, extra_body, start, prompt, tokenizer
) -> dict:
    """Standard OpenAI /chat/completions with SSE streaming."""
    messages = [{"role": "user", "content": f"{context}\n\n{prompt}"}]
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for k, v in extra_body.items():
        if k not in body:
            body[k] = v

    first_tok = None
    last_tok = None
    tok_count = 0
    chunk_count = 0
    output_text = ""
    done = False

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url, data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            debug_count = 0
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if done:
                    break
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    now = time.monotonic()
                    if first_tok is None:
                        first_tok = now
                    last_tok = now
                    output_text += content
                    chunk_count += 1
                    if chunk_count % 50 == 0 or chunk_count == 1:
                        tok_count = count_tokens(output_text, tokenizer)
                    if tok_count >= max_tokens:
                        done = True
                elif debug_count < 3:
                    print(f"  [DEBUG chunk] keys={list(chunk.keys())} choices_empty={not choices}")
                    debug_count += 1
                    sys.stdout.flush()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:200]
        return {"error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        return {"error": str(e)}

    end = time.monotonic()
    if output_text:
        tok_count = count_tokens(output_text, tokenizer)
    prefill = (first_tok - start) if first_tok else None
    gen = (last_tok - first_tok) if (first_tok and last_tok) else None
    return {
        "prefill_time_s": round(prefill, 3) if prefill else None,
        "generation_time_s": round(gen, 3) if gen else None,
        "total_time_s": round(end - start, 3),
        "token_count": tok_count,
    }


def _send_responses_api(
    base_url, model, context, api_key, temperature, max_tokens, extra_body, start, prompt, tokenizer
) -> dict:
    """OpenAI /responses API with streaming."""
    input_text = f"{context}\n\n{prompt}"
    body = {
        "model": model,
        "input": input_text,
        "stream": True,
        "temperature": temperature,
    }
    for k, v in extra_body.items():
        if k not in body:
            body[k] = v

    first_tok = None
    last_tok = None
    tok_count = 0
    chunk_count = 0
    output_text = ""
    done = False

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url, data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
                if event_type == "response.done":
                    break

                if done:
                    break

                if event_type == "response.output_text.delta":
                    text = event.get("delta", "")
                    if text:
                        now = time.monotonic()
                        if first_tok is None:
                            first_tok = now
                        last_tok = now
                        output_text += text
                        chunk_count += 1
                        if chunk_count % 50 == 0 or chunk_count == 1:
                            tok_count = count_tokens(output_text, tokenizer)
                        if tok_count >= max_tokens:
                            done = True
                elif event_type == "response.text.delta":
                    text = event.get("delta", "")
                    if text:
                        now = time.monotonic()
                        if first_tok is None:
                            first_tok = now
                        last_tok = now
                        output_text += text
                        chunk_count += 1
                        if chunk_count % 50 == 0 or chunk_count == 1:
                            tok_count = count_tokens(output_text, tokenizer)
                        if tok_count >= max_tokens:
                            done = True
                elif event_type == "response.output_item.delta":
                    delta = event.get("delta", "")
                    if isinstance(delta, str) and delta:
                        now = time.monotonic()
                        if first_tok is None:
                            first_tok = now
                        last_tok = now
                        output_text += delta
                        chunk_count += 1
                        if chunk_count % 50 == 0 or chunk_count == 1:
                            tok_count = count_tokens(output_text, tokenizer)
                        if tok_count >= max_tokens:
                            done = True
                    elif isinstance(delta, dict):
                        for v in delta.values():
                            if isinstance(v, str) and v:
                                now = time.monotonic()
                                if first_tok is None:
                                    first_tok = now
                                last_tok = now
                                output_text += v
                                chunk_count += 1
                                if chunk_count % 50 == 0 or chunk_count == 1:
                                    tok_count = count_tokens(output_text, tokenizer)
                                if tok_count >= max_tokens:
                                    done = True
                                break
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:200]
        return {"error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        return {"error": str(e)}

    end = time.monotonic()
    if output_text:
        tok_count = count_tokens(output_text, tokenizer)
    prefill = (first_tok - start) if first_tok else None
    gen = (last_tok - first_tok) if (first_tok and last_tok) else None
    return {
        "prefill_time_s": round(prefill, 3) if prefill else None,
        "generation_time_s": round(gen, 3) if gen else None,
        "total_time_s": round(end - start, 3),
        "token_count": tok_count,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(results) -> dict:
    def pct(values, p):
        if not values:
            return 0
        s = sorted(values)
        return s[int(len(s) * p)]

    pre = [r["prefill_time_s"] for r in results if r.get("prefill_time_s") is not None]
    gen = [r["generation_time_s"] for r in results if r.get("generation_time_s") is not None]

    pre_tok_s = []
    for r in results:
        it = r.get("input_tokens") or 0
        pt = r.get("prefill_time_s")
        if it and pt and pt > 0:
            pre_tok_s.append(round(it / pt, 1))

    gen_tok_s = []
    for r in results:
        tc = r.get("token_count") or 0
        gt = r.get("generation_time_s")
        if tc and gt and gt > 0:
            gen_tok_s.append(round(tc / gt, 1))

    def s(vals):
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "mean": round(sum(vals) / len(vals), 3),
            "median": round(pct(vals, 0.5), 3),
            "p95": round(pct(vals, 0.95), 3),
        }

    return {"prefill": s(pre), "prefill_tok_s": s(pre_tok_s), "gen": s(gen), "gen_tok_s": s(gen_tok_s)}


def print_stats(st) -> None:
    table = Table(title="STATISTICS", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Min", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Max", justify="right")

    for label, key in [
        ("Prefill (s)", "prefill"),
        ("Prefill tok/s", "prefill_tok_s"),
        ("Generation (s)", "gen"),
        ("Generation tok/s", "gen_tok_s"),
    ]:
        v = st[key]
        if v["count"] == 0:
            table.add_row(label, "N/A", "N/A", "N/A", "N/A", "N/A")
        else:
            table.add_row(label, str(v["min"]), str(v["mean"]), str(v["median"]), str(v["p95"]), str(v["max"]))

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Interactive menu (prompt_toolkit)
# ---------------------------------------------------------------------------

def select_endpoint(configs: list) -> Optional[dict]:
    """Arrow-key navigable endpoint selection using prompt_toolkit choice."""
    options = []
    for cfg in configs:
        name = cfg.get("_source_file", "unknown")
        model_id = cfg.get("id", "?")
        api_type = cfg.get("apiType", "?")
        label = f"{name}  [{model_id} / {api_type}]"
        options.append((cfg, label))

    result = pt_choice(
        message="Select Endpoint (Esc=quit)",
        options=options,
        show_frame=True,
    )
    return result


def edit_params(selected: dict) -> dict:
    """Arrow-key navigable parameter editor using prompt_toolkit."""
    model_id = selected.get("id", "")
    base_url = selected.get("baseUrl", "")
    api_key = selected.get("apiKey", os.environ.get("OPENAI_API_KEY", ""))
    temperature = selected.get("temperature", 0.6)
    extra_body = selected.get("extraBody", {})
    concurrency = 1
    num_runs = 5
    context_tokens = 30_000
    max_tokens = 4096
    essay_words = DEFAULT_ESSAY_WORDS

    params = [
        ("API Key", api_key, str),
        ("Temperature", temperature, float),
        ("Concurrency", concurrency, int),
        ("Batches", num_runs, int),
        ("Context Tokens", context_tokens, int),
        ("Max Gen Tokens", max_tokens, int),
        ("Essay Words", essay_words, int),
    ]
    RUN_ACTION = len(params)  # sentinel index for the "Run Load Test" menu item

    sel = [0]
    run = [False]
    editing = [False]

    style = Style.from_dict({
        "title": "#00bfff bold",
        "param_name": "#ffff00",
        "param_value": "#ffffff",
        "selected": "#00ff00 bold",
        "hint": "#888888",
        "editbar": "#cccccc",
        "editbar.text": "#ffffff",
    })

    edit_buffer = Buffer()
    buffer_control = BufferControl(buffer=edit_buffer)
    edit_label = Window(
        content=FormattedTextControl(lambda: " Value: " if editing[0] else ""),
        height=1,
        width=9,
        style="class:editbar",
    )
    edit_input = Window(
        content=buffer_control,
        style="class:editbar",
        height=1,
    )
    edit_window = VSplit([edit_label, edit_input], height=1)

    name_col_width = max(len(name) for name, _, _ in params) + 1

    def get_lines():
        lines = []
        hint = " Editing — Enter=confirm  Esc=cancel" if editing[0] else \
               " Configuration  (↑↓=navigate  Enter=edit/run  Esc=quit)"
        lines.append(("class:title", hint + "\n\n"))
        lines.append(("class:param_name", f" Endpoint: "))
        lines.append(("class:param_value", f"{model_id}\n"))
        lines.append(("class:param_name", f" URL:      "))
        lines.append(("class:param_value", f"{base_url}\n\n"))
        for i, (name, val, typ) in enumerate(params):
            if name == "API Key":
                display = ("*" * 8 + str(val)[-4:]) if len(str(val)) > 4 else "(empty)"
            elif isinstance(val, int):
                display = f"{val:,}"
            else:
                display = str(val)
            padded_name = f"{name}:".ljust(name_col_width + 1)
            marker = ">" if i == sel[0] else " "
            style = "class:selected" if i == sel[0] else "class:param_name"
            lines.append((style, f" {marker} {padded_name}"))
            lines.append(("class:param_value" if i != sel[0] else "class:selected", f" {display}\n"))
        lines.append(("", "\n"))
        run_marker = ">" if sel[0] == RUN_ACTION else " "
        run_style = "class:selected" if sel[0] == RUN_ACTION else "class:hint"
        lines.append((run_style, f" {run_marker} ▶ Run Load Test\n"))
        return lines

    top_window = Window(
        content=FormattedTextControl(get_lines),
        always_hide_cursor=Condition(lambda: not editing[0]),
    )

    layout = Layout(HSplit([top_window, edit_window]), focused_element=top_window)

    bindings = KeyBindings()

    @bindings.add("up", filter=~Condition(lambda: editing[0]))
    def go_up(event):
        sel[0] = (sel[0] - 1) % (len(params) + 1)

    @bindings.add("down", filter=~Condition(lambda: editing[0]))
    def go_down(event):
        sel[0] = (sel[0] + 1) % (len(params) + 1)

    @bindings.add("enter", filter=~Condition(lambda: editing[0]))
    def start_edit(event):
        if sel[0] == RUN_ACTION:
            run[0] = True
            event.app.exit()
            return
        name, val, typ = params[sel[0]]
        if name == "API Key":
            default = api_key
        else:
            default = str(val)
        edit_buffer.text = default
        edit_buffer.cursor_position = len(default)
        editing[0] = True
        event.app.layout.focus(edit_input)

    @bindings.add("enter", filter=Condition(lambda: editing[0]))
    def confirm_edit(event):
        name, val, typ = params[sel[0]]
        text = edit_buffer.text
        try:
            new_val = typ(text)
            params[sel[0]] = (name, new_val, typ)
        except (ValueError, TypeError):
            pass
        editing[0] = False
        edit_buffer.text = ""
        event.app.layout.focus(top_window)

    @bindings.add("escape", filter=Condition(lambda: editing[0]))
    def cancel_edit(event):
        editing[0] = False
        edit_buffer.text = ""
        event.app.layout.focus(top_window)

    @bindings.add("escape", filter=~Condition(lambda: editing[0]))
    def quit_app(event):
        event.app.exit()

    app = Application(
        layout=layout,
        key_bindings=bindings,
        style=style,
        full_screen=True,
    )
    app.run()

    if not run[0]:
        return None

    api_key = params[0][1]
    temperature = params[1][1]
    concurrency = params[2][1]
    num_runs = params[3][1]
    context_tokens = params[4][1]
    max_tokens = params[5][1]
    essay_words = params[6][1]

    return {
        "model_id": model_id,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "concurrency": concurrency,
        "num_runs": num_runs,
        "context_tokens": context_tokens,
        "max_tokens": max_tokens,
        "essay_words": essay_words,
        "extra_body": extra_body,
        "api_type": selected.get("apiType", "OpenAICompletion"),
    }


def run_load_test(params: dict) -> None:
    """Run the actual load test and print results."""
    model_id = params["model_id"]
    base_url = params["base_url"]
    api_key = params["api_key"]
    temperature = params["temperature"]
    concurrency = params["concurrency"]
    num_runs = params["num_runs"]
    context_tokens = params["context_tokens"]
    max_tokens = params["max_tokens"]
    essay_words = params["essay_words"]
    extra_body = params["extra_body"]
    api_type = params["api_type"]

    console.print()
    console.rule("[bold cyan]Load Test: {}[/]".format(model_id))
    console.print(f"  URL:         {base_url}")
    console.print(f"  Concurrency: {concurrency}  |  Batches: {num_runs}  |  Context: ~{context_tokens:,} tokens")
    console.print(f"  Temperature: {temperature}  |  Max gen: {max_tokens}  |  Essay words: {essay_words}")
    console.print()

    all_results = []
    total_errors = 0

    for batch in range(num_runs):
        batch_num = batch + 1
        console.print(f"[bold]--- Batch {batch_num}/{num_runs} ({concurrency} concurrent) ---[/]")

        contexts = [generate_context(context_tokens, model_id) for _ in range(concurrency)]

        def do_req(req_id, ctx):
            return (req_id, send_request(
                base_url, model_id, ctx, api_key, temperature, max_tokens, extra_body, api_type, essay_words
            ))

        batch_results = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(do_req, batch * concurrency + i + 1, ctx)
                for i, ctx in enumerate(contexts)
            ]
            for fut in as_completed(futures, timeout=7200):
                try:
                    req_id, result = fut.result()
                    batch_results.append((req_id, result))
                except Exception as e:
                    batch_results.append((0, {"error": str(e)}))

        batch_results.sort(key=lambda x: x[0])
        for req_id, r in batch_results:
            if "error" in r:
                total_errors += 1
                console.print(f"  [red]Req {req_id:>3}: ERROR[/] — {r['error'][:120]}")
            else:
                pre = r.get("prefill_time_s")
                gen = r.get("generation_time_s")
                tok = r.get("token_count", 0)
                inp = r.get("input_tokens", 0)
                pre_tok_s = round(inp / pre, 1) if (inp and pre and pre > 0) else 0
                gen_tok_s = round(tok / gen, 1) if (tok and gen and gen > 0) else 0
                status = "OK"
                if pre is None:
                    status = "NO_TOKENS"
                console.print(
                    f"  [green]Req {req_id:>3}[/]: prefill={pre}s ({pre_tok_s} tok/s)  "
                    f"gen={gen}s ({gen_tok_s} tok/s)  in={inp}  out={tok}  [{status}]"
                )
            all_results.append(r)

    # Statistics
    st = compute_stats(all_results)
    print_stats(st)
    succ = st["prefill"]["count"]
    console.print(f"\n[bold]Successful:[/bold] {succ}/{len(all_results)}  |  [red]Errors:[/red] {total_errors}")
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not sys.stdout.isatty():
        console.print("Error: this script requires a terminal (tty).")
        sys.exit(1)

    configs = load_model_configs()
    if not configs:
        console.print(f"[yellow]No model configs found in {MODELS_DIR}[/]")
        sys.exit(1)

    selected = select_endpoint(configs)
    if selected is None:
        sys.exit(0)

    params = edit_params(selected)
    if params is None:
        sys.exit(0)

    run_load_test(params)


if __name__ == "__main__":
    main()