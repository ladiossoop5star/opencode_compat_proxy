#!/usr/bin/env python3
"""OpenAI-compatible tool-call compatibility proxy.

Converts DeepSeek DSML and Qwen XML raw tool calls into standard
OpenAI tool_calls format. Sits between opencode (port 9526) and
llama.cpp (port 8000).
"""

import html
import json
import os
import re
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8000")
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
    if "<tool_call>" in text and "</tool_call>" in text:
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
    if not tool_calls and content and has_complete_raw_tool_block(content):
        tool_calls = parse_raw_tool_calls(content)
        if tool_calls:
            msg["tool_calls"] = tool_calls
            msg["content"] = None
            body["choices"][0]["finish_reason"] = "tool_calls"
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


async def stream_compat_response(upstream_req, forwarded_for=""):
    chunk_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    model = upstream_req.get("model", "deepseek")
    raw_buffer = ""
    deferred_chunks = []

    async def generate():
        nonlocal raw_buffer
        sent_role = False
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
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    parsed = sse_json(line)
                    if not parsed:
                        continue
                    if parsed.get("done"):
                        break

                    chunk = parsed
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # Strip empty tool_calls array from vLLM (it sends tool_calls: [] on every content chunk)
                    tc_val = delta.get("tool_calls")
                    has_real_tc = bool(tc_val)
                    if "tool_calls" in delta and not has_real_tc:
                        delta = {k: v for k, v in delta.items() if k != "tool_calls"}
                        patched = dict(chunk)
                        patched["choices"] = [dict(choice, delta=delta)]
                        chunk = patched

                    has_tc_field = has_real_tc
                    has_role_only = "role" in delta and len(delta) == 1
                    has_empty_content = (
                        "content" in delta
                        and not delta.get("content")
                        and len(delta) == 1
                    )
                    has_reasoning_only = (
                        "reasoning" in delta
                        and "content" not in delta
                        and "tool_calls" not in delta
                    )
                    has_content_only = (
                        "content" in delta
                        and delta.get("content")
                        and len(delta) == 1
                    )

                    if has_tc_field:
                        for c in deferred_chunks:
                            yield "data: " + json.dumps(c) + "\n\n"
                        deferred_chunks.clear()
                        raw_buffer = ""
                        yield "data: " + json.dumps(chunk) + "\n\n"
                        continue

                    if has_role_only or has_empty_content:
                        stripped = {k: v for k, v in delta.items() if v}
                        if stripped:
                            patched = dict(chunk)
                            patched["choices"] = [dict(choice, delta=stripped)]
                            yield "data: " + json.dumps(patched) + "\n\n"
                        continue

                    if has_content_only:
                        text = delta["content"]
                        raw_buffer += text
                        yield "data: " + json.dumps(chunk) + "\n\n"

                        if has_complete_raw_tool_block(raw_buffer):
                            raw_tool_calls = parse_raw_tool_calls(raw_buffer)
                            if raw_tool_calls:
                                raw_buffer = ""
                                for tc_chunk in build_stream_tool_call_chunks(
                                    raw_tool_calls, chunk_id, model
                                ):
                                    yield "data: " + json.dumps(tc_chunk) + "\n\n"
                                return
                        continue

                    if has_reasoning_only:
                        reasoning_text = delta.get("reasoning") or delta.get("reasoning_content") or ""
                        raw_buffer += reasoning_text
                        yield "data: " + json.dumps(chunk) + "\n\n"

                        if has_complete_raw_tool_block(raw_buffer):
                            raw_tool_calls = parse_raw_tool_calls(raw_buffer)
                            if raw_tool_calls:
                                raw_buffer = ""
                                for tc_chunk in build_stream_tool_call_chunks(
                                    raw_tool_calls, chunk_id, model
                                ):
                                    yield "data: " + json.dumps(tc_chunk) + "\n\n"
                                return
                        continue

                    yield "data: " + json.dumps(chunk) + "\n\n"

                if raw_buffer and has_complete_raw_tool_block(raw_buffer):
                    raw_tool_calls = parse_raw_tool_calls(raw_buffer)
                    if raw_tool_calls:
                        deferred_chunks.clear()
                        raw_buffer = ""
                        for tc_chunk in build_stream_tool_call_chunks(
                            raw_tool_calls, chunk_id, model
                        ):
                            yield "data: " + json.dumps(tc_chunk) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return

                for c in deferred_chunks:
                    yield "data: " + json.dumps(c) + "\n\n"
                deferred_chunks.clear()

                yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.api_route("/v1/chat/completions", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    client_host = request.client.host if request.client else "unknown"
    upstream_headers = strip_hop_by_hop_headers(dict(request.headers))
    upstream_headers.pop("host", None)
    upstream_headers["content-type"] = "application/json"
    if not upstream_headers.get("x-forwarded-for"):
        upstream_headers["x-forwarded-for"] = client_host

    upstream_req = dict(body)

    if stream:
        return await stream_compat_response(upstream_req, upstream_headers["x-forwarded-for"])

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            UPSTREAM + "/v1/chat/completions",
            json=upstream_req,
            headers=upstream_headers,
            timeout=None,
        )

    try:
        result = resp.json()
    except Exception:
        return JSONResponse(content=resp.text, status_code=resp.status_code)

    result = convert_non_streaming_response(result)
    return JSONResponse(content=result, status_code=resp.status_code)


@app.api_route("/v1/models/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
@app.get("/v1/models")
async def models_proxy(request: Request, path: str = ""):
    upstream_path = f"/v1/models/{path}" if path else "/v1/models"
    async with httpx.AsyncClient() as client:
        resp = await client.get(UPSTREAM + upstream_path, timeout=None)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all_proxy(request: Request, path: str):
    body_bytes = await request.body()
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
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
