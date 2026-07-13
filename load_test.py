#!/usr/bin/env python3
"""
OpenAI Chat API Load Test — interactive UI

Loads model endpoint configs from ~/.junie-local/models/*.json,
presents a curses-based menu to pick an endpoint and tune parameters,
then runs concurrent load tests and prints statistics.

Uses Python stdlib (urllib, curses, concurrent.futures) plus
transformers (optional, for accurate token counting via HuggingFace tokenizers).

Usage:
    python load_test.py
"""

import curses
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from glob import glob
from typing import Optional

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False


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
            print(f"Warning: could not load {path}: {e}")
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
    # e.g. "mlx-community/Qwen3.6-27B-8bit" -> try "Qwen/Qwen2.5-32B" as fallback
    normalized = model_id.lower().replace("-", "").replace("_", "")
    
    # Check cache first
    if normalized in _tokenizer_cache:
        return _tokenizer_cache[normalized]
    
    # Try loading the tokenizer for the exact model ID first
    tokenizer = None
    candidates = [model_id]
    
    # Map known model families to their tokenizer sources
    if "qwen3" in normalized or "qwen2" in normalized:
        # Qwen3.6 uses Qwen2Tokenizer (verified: Qwen/Qwen3.6-27B tokenizer_config.json
        # declares tokenizer_class=Qwen2Tokenizer). Qwen2.5 tokenizer is identical
        # in token counts and much lighter/faster to download.
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
        # Use actual tokenizer to build context close to target token count
        header_tokens = len(tokenizer.encode(header))
        filler_tokens = len(tokenizer.encode(FILLER))
        if filler_tokens > 0:
            repeats = max(1, (target_tokens - header_tokens) // filler_tokens)
        else:
            repeats = max(1, (target_tokens * 4 - len(header)) // len(FILLER))
    else:
        # Fallback: char/4 estimation
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
                    continue
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
    # Final accurate token count
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
                    continue

                # OpenAI Responses API: text content arrives in these event types
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
                        # delta may be a dict with content field
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
    # Final accurate token count
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

    # Prefill tok/s = input_tokens / prefill_time
    pre_tok_s = []
    for r in results:
        it = r.get("input_tokens") or 0
        pt = r.get("prefill_time_s")
        if it and pt and pt > 0:
            pre_tok_s.append(round(it / pt, 1))

    # Generation tok/s = token_count / generation_time
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
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("STATISTICS")
    lines.append("=" * 78)
    fmt = "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8}"
    lines.append(fmt.format("Metric", "Min", "Mean", "Median", "P95", "Max"))
    lines.append("-" * 78)
    for label, key in [("Prefill (s)", "prefill"), ("Prefill tok/s", "prefill_tok_s"), ("Generation (s)", "gen"), ("Generation tok/s", "gen_tok_s")]:
        v = st[key]
        if v["count"] == 0:
            lines.append(f"{label:<20} {'N/A':>8}")
        else:
            lines.append(fmt.format(label, v["min"], v["mean"], v["median"], v["p95"], v["max"]))
    lines.append("=" * 78)
    for line in lines:
        print(line)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Curses UI helpers
# ---------------------------------------------------------------------------

def draw_str(win, y, x, text, attr=0):
    h, w = win.getmaxyx()
    if y >= h or x >= w:
        return
    max_len = w - x - 1
    win.addstr(max(0, y), max(0, x), text[:max_len], attr)


def prompt_input(stdscr, label: str, default: str) -> str:
    h, w = stdscr.getmaxyx()
    box_w = min(60, w - 4)
    box_h = 7
    box_y = (h - box_h) // 2
    box_x = (w - box_w) // 2
    win = curses.newwin(box_h, box_w, box_y, box_x)
    win.keypad(True)
    win.border()
    draw_str(win, 0, 2, label, curses.A_BOLD)
    draw_str(win, 1, 2, "Enter=OK  Esc=Cancel")

    buf = list(str(default))
    cur = len(buf)

    while True:
        val = "".join(buf)
        prefix = "> "
        # Clear the line first so deleted chars don't leave ghost artifacts
        win.move(3, 2)
        win.clrtoeol()
        draw_str(win, 3, 2, f"{prefix}{val}")
        vis_x = 2 + len(prefix) + cur
        if 2 + len(prefix) <= vis_x < 2 + len(prefix) + len(val):
            try:
                ch = win.inch(3, vis_x)
                win.addch(3, vis_x, ch & 0xFF, curses.A_REVERSE)
            except curses.error:
                pass
        win.refresh()

        key = win.getch()
        if key == 27:
            return default
        elif key in (curses.KEY_ENTER, 10, 13):
            return "".join(buf)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if cur > 0:
                cur -= 1
                buf.pop(cur)
        elif key == curses.KEY_DC:
            if cur < len(buf):
                buf.pop(cur)
        elif 32 <= key <= 126:
            buf.insert(cur, chr(key))
            cur += 1
        elif key == curses.KEY_LEFT and cur > 0:
            cur -= 1
        elif key == curses.KEY_RIGHT and cur < len(buf):
            cur += 1


def select_menu(stdscr, title: str, items: list) -> int:
    h, w = stdscr.getmaxyx()
    menu_h = min(len(items) + 5, h - 2)
    menu_w = min(max(len(i) for i in items) + 4, w - 2)
    my = (h - menu_h) // 2
    mx = (w - menu_w) // 2
    win = curses.newwin(menu_h, menu_w, my, mx)
    win.keypad(True)
    win.border()
    draw_str(win, 0, 2, title, curses.A_BOLD)
    draw_str(win, 1, 2, "Arrows=select  Enter=OK  Esc=cancel", curses.A_DIM)

    sel = 0
    while True:
        for i, item in enumerate(items):
            prefix = "> " if i == sel else "  "
            attr = curses.A_BOLD if i == sel else 0
            draw_str(win, 3 + i, 2, f"{prefix}{item}", attr)
        win.refresh()

        key = win.getch()
        if key == 27:
            return -1
        elif key in (curses.KEY_ENTER, 10, 13):
            return sel
        elif key == curses.KEY_UP:
            sel = (sel - 1) % len(items)
        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(items)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def run_ui(stdscr) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

    configs = load_model_configs()
    if not configs:
        h, w = stdscr.getmaxyx()
        draw_str(stdscr, h // 2, 2, f"No model configs found in {MODELS_DIR}")
        stdscr.refresh()
        stdscr.getch()
        return

    # --- Step 1: Select endpoint ---
    labels = []
    for cfg in configs:
        name = cfg.get("_source_file", "unknown")
        model_id = cfg.get("id", "?")
        url = cfg.get("baseUrl", "")
        api_type = cfg.get("apiType", "?")
        labels.append(f"{name}  [{model_id} / {api_type}]\n  {url}")

    idx = select_menu(stdscr, "Select Endpoint", labels)
    if idx == -1:
        return
    selected = configs[idx]

    # --- Step 2: Parameters ---
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

    sel = 0
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        draw_str(stdscr, 0, 2, "OpenAI Chat API Load Test — Configuration", curses.A_BOLD | curses.color_pair(1))
        draw_str(stdscr, 1, 2, f"Endpoint: {model_id}", curses.A_DIM)
        draw_str(stdscr, 2, 2, f"URL:      {base_url}", curses.A_DIM)

        for i, (name, val, _) in enumerate(params):
            if name == "API Key":
                display = ("*" * 8 + str(val)[-4:]) if len(str(val)) > 4 else "(empty)"
            else:
                display = f"{val:,}" if isinstance(val, int) else str(val)
            prefix = " >" if i == sel else "  "
            attr = curses.A_REVERSE if i == sel else 0
            draw_str(stdscr, 4 + i, 2, f"{prefix}{name}: {display}", attr)

        draw_str(stdscr, 12, 2, "Up/Down=select  Enter=edit  Ctrl+R=run  Esc=quit", curses.color_pair(2) | curses.A_BOLD)
        stdscr.refresh()

        key = stdscr.getch()
        if key == 27:
            return
        elif key == 18:  # Ctrl+R
            break
        elif key == curses.KEY_UP:
            sel = (sel - 1) % len(params)
        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(params)
        elif key in (curses.KEY_ENTER, 10, 13):
            name, val, typ = params[sel]
            old = str(val)
            new = prompt_input(stdscr, f"Edit: {name}", old)
            try:
                params[sel] = (name, typ(new), typ)
            except ValueError:
                pass
            api_key = params[0][1]
            temperature = params[1][1]
            concurrency = params[2][1]
            num_runs = params[3][1]
            context_tokens = params[4][1]
            max_tokens = params[5][1]
            essay_words = params[6][1]

    # --- Step 3: Run load test ---
    curses.endwin()
    # Restore terminal: exit alternate screen, show cursor, reset scrolling, clear
    sys.stdout.write("\033[?1049l\033[?47l\033[?25h\033[r\033[H\033[2J")
    sys.stdout.flush()

    print()
    print("=" * 78)
    print(f"Load Test: {model_id}")
    print(f"URL:       {base_url}")
    print(f"Concurrency: {concurrency}  |  Batches: {num_runs}  |  Context: ~{context_tokens:,} tokens")
    print(f"Temperature: {temperature}  |  Max gen: {max_tokens}  |  Essay words: {essay_words}")
    print("=" * 78)
    sys.stdout.flush()

    all_results = []
    total_errors = 0

    for batch in range(num_runs):
        batch_num = batch + 1
        print(f"\n--- Batch {batch_num}/{num_runs} ({concurrency} concurrent) ---")
        sys.stdout.flush()

        contexts = [generate_context(context_tokens, model_id) for _ in range(concurrency)]

        api_type = selected.get("apiType", "OpenAICompletion")

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
                print(f"  Req {req_id:>3}: ERROR — {r['error'][:120]}")
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
                print(f"  Req {req_id:>3}: prefill={pre}s ({pre_tok_s} tok/s)  gen={gen}s ({gen_tok_s} tok/s)  in={inp}  out={tok}  [{status}]")
            sys.stdout.flush()
            all_results.append(r)

    sys.stdout.flush()

    # --- Step 4: Statistics ---
    st = compute_stats(all_results)
    print_stats(st)
    succ = st["prefill"]["count"]
    print(f"\nSuccessful: {succ}/{len(all_results)}  |  Errors: {total_errors}")
    print()
    sys.stdout.flush()


def main():
    if not sys.stdout.isatty():
        print("Error: this script requires a terminal (tty).")
        sys.exit(1)
    curses.wrapper(run_ui)


if __name__ == "__main__":
    main()