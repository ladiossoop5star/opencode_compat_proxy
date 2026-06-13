#!/usr/bin/env python3
"""OpenAI-compatible tool-call compatibility proxy.

Converts DeepSeek DSML and Qwen XML raw tool calls into standard
OpenAI tool_calls format. Sits between opencode (port 9526) and
llama.cpp (port 8000).
"""

import html
import json
import logging
import os
import re
import uuid

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("proxy")

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:9527")
HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("PROXY_PORT", "9526"))

HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-length", "content-type",
})

DSML_BAR = chr(0xFF5C)
DSML_OPEN = "<" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"
DSML_CLOSE = "</" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"

SECTION_SIZE = 32
GUARD_SECTIONS = 2


def normalize_raw_tool_calls(text):
    """Normalize various DSML/Qwen XML formats to standard <｜DSML｜tool_calls> format."""
    bar = DSML_BAR
    # Format 1: <DSML>tool_calls> (DSML pseudo-namespace without bars)
    if "<DSML>tool_calls>" in text:
        text = text.replace("<DSML>tool_calls>", "<" + bar + "DSML" + bar + "tool_calls>", 1)
        text = re.sub(r'</DSML[:\s]+tool_calls\s*>', "</" + bar + "DSML" + bar + "tool_calls>", text)
        text = re.sub(r'<DSML[:\s]+(invoke)\s+', "<" + bar + "DSML" + bar + r"\1 ", text)
        text = re.sub(r'<DSML[:\s]+(parameter)\s+', "<" + bar + "DSML" + bar + r"\1 ", text)
        text = re.sub(r'</DSML[:\s]+(invoke|parameter)\s*>', "</" + bar + "DSML" + bar + r"\1>", text)
    # Format 2: <tool_calls> (bare XML, no DSML prefix)
    elif "<tool_calls>" in text and "</tool_calls>" in text:
        text = text.replace("<tool_calls>", "<" + bar + "DSML" + bar + "tool_calls>", 1)
        text = text.replace("</tool_calls>", "</" + bar + "DSML" + bar + "tool_calls>", 1)
        text = re.sub(r'<invoke\s+', "<" + bar + "DSML" + bar + "invoke ", text)
        text = re.sub(r'</invoke\s*>', "</" + bar + "DSML" + bar + "invoke>", text)
        text = re.sub(r'<parameter\s+', "<" + bar + "DSML" + bar + "parameter ", text)
        text = re.sub(r'</parameter\s*>', "</" + bar + "DSML" + bar + "parameter>", text)
    return text

app = FastAPI()


def strip_hop_by_hop_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def normalize_arg_value(val):
    if isinstance(val, str):
        val = html.unescape(val)
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
        return val
    return json.dumps(val, ensure_ascii=False)


def make_tool_call(name, arguments, call_id=None):
    return {
        "id": call_id or "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {
            "name": name,
            "arguments": normalize_arg_value(arguments),
        },
    }


def has_complete_raw_tool_block(text):
    if DSML_OPEN in text and DSML_CLOSE in text:
        return True
    if "<DSML>tool_calls>" in text:
        return True
    if "<tool_calls>" in text and "</tool_calls>" in text:
        return True
    if "<tool_call>" in text and "</tool_call>" in text:
        return True
    return False


def has_any_dsml_prefix(text):
    """Check if text may contain the start of a raw tool block."""
    if not text:
        return False
    tail = text[-150:] if len(text) > 150 else text
    if DSML_OPEN[:8] in tail:
        return True
    if "<DSML>" in tail or "<DSML:" in tail:
        return True
    if "<tool_calls" in tail:
        return True
    if "<tool_call" in tail:
        return True
    return False


def parse_dsml_tool_calls(text):
    results = []
    for m in re.finditer(
        re.escape(DSML_OPEN) + r"(.*?)" + re.escape(DSML_CLOSE), text, re.DOTALL
    ):
        block = m.group(1)

        # Format 1: <name>fn</name><parameters>...</parameters>
        for tc in re.finditer(
            r"<name>\s*(.*?)\s*</name>.*?<parameters>\s*(.*?)\s*</parameters>",
            block,
            re.DOTALL,
        ):
            results.append(make_tool_call(tc.group(1).strip(), tc.group(2).strip()))

        # Format 2: <DSML invoke name="fn"><DSML parameter name="k" string="t">v</DSML parameter></DSML invoke>
        invoke_pat = re.escape("<" + DSML_BAR + "DSML" + DSML_BAR + "invoke") + r'\s+name="([^"]+)"\s*>'
        for inv in re.finditer(invoke_pat, block):
            fn_name = inv.group(1)
            after_invoke = block[inv.end():]
            end_invoke = re.search(re.escape("</" + DSML_BAR + "DSML" + DSML_BAR + "invoke>"), after_invoke)
            if not end_invoke:
                continue
            param_block = after_invoke[:end_invoke.start()]
            params = {}
            for p in re.finditer(
                re.escape("<" + DSML_BAR + "DSML" + DSML_BAR + "parameter") +
                r'\s+name="([^"]+)"\s*(?:string="[^"]*"\s*)?' +
                r">(.*?)</" + DSML_BAR + "DSML" + DSML_BAR + "parameter>",
                param_block,
                re.DOTALL,
            ):
                params[p.group(1)] = p.group(2).strip()
            results.append(make_tool_call(fn_name, json.dumps(params, ensure_ascii=False)))
    return results


def parse_qwen_xml_tool_calls(text):
    results = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        block = m.group(1)
        name_m = re.search(r"<name>\s*(.*?)\s*</name>", block, re.DOTALL)
        args_m = re.search(r"<parameters>\s*(.*?)\s*</parameters>", block, re.DOTALL)
        if name_m:
            results.append(make_tool_call(
                name_m.group(1).strip(),
                (args_m.group(1).strip() if args_m else "{}"),
            ))
    return results


def parse_raw_tool_calls(text):
    tc = parse_dsml_tool_calls(text)
    if tc:
        return tc
    tc = parse_qwen_xml_tool_calls(text)
    if tc:
        return tc
    # tool block found but nothing parsed — log the block for debugging
    if has_complete_raw_tool_block(text):
        log.warning("parse_raw_tool_calls failed on: %s", text[:800])
    return []


def collect_text_fields(delta):
    parts = []
    for field in ("content", "reasoning_content", "reasoning"):
        val = delta.get(field)
        if isinstance(val, str) and val:
            parts.append(val)
    return "".join(parts)


def convert_non_streaming_response(body):
    msg = body.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        log.info("Upstream already has tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False)[:500])
    if not tool_calls and content:
        if DSML_CLOSE in content or "</tool_calls>" in content or "</tool_call>" in content:
            log.info("Raw tool close tag found in content (len=%d): %s", len(content), content[:500])
        if has_complete_raw_tool_block(content):
            normalized = normalize_raw_tool_calls(content)
            tool_calls = parse_raw_tool_calls(normalized)
            if tool_calls:
                log.info("Converted %d tool_calls from non-stream content", len(tool_calls))
                msg["tool_calls"] = tool_calls
                msg["content"] = None
                body["choices"][0]["finish_reason"] = "tool_calls"
            else:
                log.warning("has_complete_raw_tool_block true but parse empty! block=[%s]", content[:600])
    return body


def sse_json(line):
    prefix = "data: "
    if not line.startswith(prefix):
        return None
    payload = line[len(prefix):]
    if payload.strip() == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return {"raw": payload, "done": False}


def build_stream_tool_call_chunks(tool_calls, chunk_id, model):
    chunks = []
    for tc in tool_calls:
        tc_id = tc["id"]
        fn = tc["function"]
        chunks.append({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": ""},
                }]},
                "finish_reason": None,
            }],
        })
        args = fn["arguments"]
        chunk_size = 32
        for i in range(0, len(args), chunk_size):
            chunks.append({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": args[i:i + chunk_size]},
                    }]},
                    "finish_reason": None,
                }],
            })
    chunks.append({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    })
    return chunks


def _make_content_sse(chunk_id, model, text):
    return "data: " + json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text, "tool_calls": []}}],
    }, ensure_ascii=False) + "\n\n"


def _find_dsml_start(text):
    """Return the index of the first DSML open marker, or len(text)."""
    for marker in (DSML_OPEN, "<DSML>tool_calls>", "<tool_calls>", "<tool_call>"):
        i = text.find(marker)
        if i != -1:
            return i
    return len(text)


async def stream_with_sections(upstream_req, forwarded_for=""):
    chunk_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    model = upstream_req.get("model", "deepseek")

    async def generate():
        nonlocal chunk_id, model
        buffer = ""
        unflushed = ""
        pending = []
        dsml_mode = False
        content_collected = False

        req_headers = {"Accept": "text/event-stream"}
        if forwarded_for:
            req_headers["x-forwarded-for"] = forwarded_for

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                UPSTREAM + "/v1/chat/completions",
                json=upstream_req,
                headers=req_headers,
                timeout=None,
            ) as resp:
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue

                    ev = sse_json(raw_line)
                    if ev is None:
                        yield raw_line + "\n\n"
                        continue
                    if ev.get("done"):
                        continue

                    choices = ev.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    if ev.get("id"):
                        chunk_id = ev["id"]
                    if ev.get("model"):
                        model = ev["model"]

                    if "role" in delta:
                        yield raw_line + "\n\n"
                        continue

                    reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                    text = delta.get("content", "")

                    if reasoning:
                        yield raw_line + "\n\n"
                    if not text:
                        if not reasoning and not dsml_mode:
                            yield raw_line + "\n\n"
                        continue

                    buffer += text
                    content_collected = True

                    if dsml_mode:
                        if has_complete_raw_tool_block(buffer):
                            idx = _find_dsml_start(buffer)
                            if idx > 0:
                                yield _make_content_sse(chunk_id, model, buffer[:idx])
                            tcs = parse_raw_tool_calls(normalize_raw_tool_calls(buffer))
                            if tcs:
                                for tc in build_stream_tool_call_chunks(tcs, chunk_id, model):
                                    yield "data: " + json.dumps(tc) + "\n\n"
                            return
                        continue

                    if has_any_dsml_prefix(buffer):
                        dsml_mode = True
                        continue

                    unflushed += text
                    while len(unflushed) >= SECTION_SIZE:
                        pending.append(unflushed[:SECTION_SIZE])
                        unflushed = unflushed[SECTION_SIZE:]
                        if len(pending) > GUARD_SECTIONS:
                            yield _make_content_sse(chunk_id, model, pending.pop(0))

                # Stream ended
                if dsml_mode:
                    if content_collected:
                        idx = _find_dsml_start(buffer)
                        if idx > 0:
                            yield _make_content_sse(chunk_id, model, buffer[:idx])
                else:
                    for s in pending:
                        yield _make_content_sse(chunk_id, model, s)
                    if unflushed:
                        yield _make_content_sse(chunk_id, model, unflushed)

                yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.api_route("/v1/chat/completions", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request):
    body_bytes = await request.body()
    stream = False
    try:
        j = json.loads(body_bytes)
        stream = j.get("stream", False)
        log.info("REQUEST: model=%s stream=%s msgs=%d tools=%d",
                 j.get("model", "?"), stream,
                 len(j.get("messages", [])),
                 len(j.get("tools", [])))
        for i, msg in enumerate(j.get("messages", [])):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            tc = msg.get("tool_calls")
            log.info("  msg[%d] role=%s content_len=%d tool_calls=%s",
                     i, role, len(content or ""),
                     len(tc) if tc else 0)
    except Exception:
        j = None
        log.info("REQUEST (raw): %s", body_bytes.decode("utf-8", errors="replace")[:1000])

    client_host = request.client.host if request.client else "unknown"
    upstream_headers = strip_hop_by_hop_headers(dict(request.headers))
    upstream_headers.pop("host", None)
    upstream_headers["content-type"] = "application/json"
    if not upstream_headers.get("x-forwarded-for"):
        upstream_headers["x-forwarded-for"] = client_host

    if stream:
        if j is None:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    UPSTREAM + "/v1/chat/completions",
                    content=body_bytes,
                    headers=upstream_headers,
                    timeout=None,
                )
            return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
        log.info("RESPONSE: streaming via stream_with_sections")
        return await stream_with_sections(j, client_host)

    async with httpx.AsyncClient() as client:
        upstream_resp = await client.post(
            UPSTREAM + "/v1/chat/completions",
            content=body_bytes,
            headers=upstream_headers,
            timeout=None,
        )

    try:
        result = upstream_resp.json()
        result = convert_non_streaming_response(result)
        msg = result.get("choices", [{}])[0].get("message", {})
        finish = result.get("choices", [{}])[0].get("finish_reason", "?")
        content_preview = (msg.get("content") or "")[:200].replace("\n", "\\n")
        if msg.get("tool_calls"):
            log.info("RESPONSE: non-stream tool_calls=%d finish=%s",
                     len(msg["tool_calls"]), finish)
        else:
            log.info("RESPONSE: non-stream finish=%s content=%s", finish, content_preview)
        return JSONResponse(content=result, status_code=upstream_resp.status_code)
    except Exception as e:
        log.info("RESPONSE: non-stream error=%s, raw_len=%d", e, len(upstream_resp.content))
        return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, headers=dict(upstream_resp.headers))


@app.get("/v1/models")
async def models_list():
    async with httpx.AsyncClient() as client:
        resp = await client.get(UPSTREAM + "/v1/models", timeout=None)
    log.info("MODELS: %d models returned", len(resp.json().get("data", [])))
    return JSONResponse(content=resp.json(), status_code=resp.status_code)





@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all_proxy(request: Request, path: str):
    body_bytes = await request.body()
    log.info("CATCH-ALL %s /%s body=%s", request.method, path,
             body_bytes.decode("utf-8", errors="replace")[:500])
    upstream_headers = strip_hop_by_hop_headers(dict(request.headers))
    upstream_headers.pop("host", None)
    client_host = request.client.host if request.client else "unknown"
    if not upstream_headers.get("x-forwarded-for"):
        upstream_headers["x-forwarded-for"] = client_host
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method,
            UPSTREAM + "/" + path,
            headers=upstream_headers,
            content=body_bytes,
            timeout=None,
        )
    log.info("CATCH-ALL %s /%s -> %d (len=%d)", request.method, path,
             resp.status_code, len(resp.content))
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
