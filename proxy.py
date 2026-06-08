import html
import json
import os
import re
import uuid
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "http://0.0.0.0:8004")

app = FastAPI()
global_client = httpx.AsyncClient(timeout=None)

def strip_hop_by_hop_headers(headers: Dict[str, str]) -> Dict[str, str]:
    blocked = {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "accept-encoding",
    }
    return {k: v for k, v in headers.items() if k.lower() not in blocked}

def normalize_arg_value(value: str) -> str:
    return html.unescape(value or "").strip()

def make_tool_call(name: str, args: Dict[str, str], call_id: Optional[str] = None) -> Dict:
    clean_args = {}
    for key, value in args.items():
        key = str(key or "").strip()
        if not key:
            continue
        clean_args[key] = normalize_arg_value(value)

    return {
        "id": call_id or ("call_" + uuid.uuid4().hex[:24]),
        "type": "function",
        "function": {
            "name": str(name or "").strip(),
            "arguments": json.dumps(clean_args, ensure_ascii=False),
        },
    }

def has_complete_raw_tool_block(text: str) -> bool:
    if not text:
        return False
    if "<｜DSML｜tool_calls>" in text and "</｜DSML｜tool_calls>" in text:
        return True
    if "<tool_call" in text and "</tool_call>" in text:
        return True
    return False

def parse_dsml_tool_calls(text: str) -> List[Dict]:
    if not text or "DSML" not in text:
        return []

    calls = []
    tool_block_re = re.compile(
        r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>",
        re.DOTALL,
    )
    invoke_re = re.compile(
        r"<｜DSML｜invoke\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</｜DSML｜invoke>",
        re.DOTALL,
    )
    param_re = re.compile(
        r"<｜DSML｜parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</｜DSML｜parameter>",
        re.DOTALL,
    )

    for block in tool_block_re.finditer(text):
        block_body = block.group(1)
        for invoke in invoke_re.finditer(block_body):
            name = invoke.group(1).strip()
            body = invoke.group(2)
            args = {}
            for param in param_re.finditer(body):
                args[param.group(1).strip()] = param.group(2)
            if name and args:
                calls.append(make_tool_call(name, args))

    return calls

def parse_qwen_xml_tool_calls(text: str) -> List[Dict]:
    if not text or "<tool_call" not in text:
        return []

    calls = []
    block_re = re.compile(
        r"<tool_call(?:[^>]*)?>(.*?)</tool_call>",
        re.DOTALL,
    )
    func_re = re.compile(
        r"<function=([A-Za-z0-9_\-\.]+)\s*>(.*?)(?:</function>|$)",
        re.DOTALL,
    )
    param_re = re.compile(
        r"<parameter=([A-Za-z0-9_\-\.]+)\s*>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    for block in block_re.finditer(text):
        block_body = block.group(1)
        func = func_re.search(block_body)
        if not func:
            continue
        name = func.group(1).strip()
        body = func.group(2)
        args = {}
        for param in param_re.finditer(body):
            args[param.group(1).strip()] = param.group(2)
        if name and args:
            calls.append(make_tool_call(name, args))

    return calls

def parse_raw_tool_calls(text: str) -> List[Dict]:
    if not has_complete_raw_tool_block(text):
        return []
    calls = []
    calls.extend(parse_dsml_tool_calls(text))
    calls.extend(parse_qwen_xml_tool_calls(text))
    return calls

def collect_text_fields(obj) -> str:
    parts = []

    def walk(x):
        if isinstance(x, dict):
            for key, value in x.items():
                if key == "tool_calls":
                    continue
                if isinstance(value, str):
                    parts.append(value)
                else:
                    walk(value)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return "\n".join(parts)

def convert_non_streaming_response(obj):
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return obj

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue

        if msg.get("tool_calls"):
            choice["finish_reason"] = "tool_calls"
            continue

        raw_text = collect_text_fields(msg)
        calls = parse_raw_tool_calls(raw_text)

        if calls:
            msg["tool_calls"] = calls
            msg["content"] = ""
            choice["finish_reason"] = "tool_calls"

    return obj

def sse_json(obj: Dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n"

def build_stream_tool_call_chunks(base_obj: Dict, calls: List[Dict]) -> List[str]:
    created = base_obj.get("created")
    model = base_obj.get("model")
    system_fingerprint = base_obj.get("system_fingerprint")
    object_name = base_obj.get("object", "chat.completion.chunk")
    chat_id = base_obj.get("id", "chatcmpl_proxy_" + uuid.uuid4().hex[:16])

    out = []

    role_obj = {
        "choices": [
            {
                "finish_reason": None,
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                },
            }
        ],
        "created": created,
        "id": chat_id,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "object": object_name,
    }
    out.append(sse_json(role_obj))

    for idx, call in enumerate(calls):
        tool_obj = {
            "choices": [
                {
                    "finish_reason": None,
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": idx,
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["function"]["name"],
                                    "arguments": call["function"]["arguments"],
                                },
                            }
                        ]
                    },
                }
            ],
            "created": created,
            "id": chat_id,
            "model": model,
            "system_fingerprint": system_fingerprint,
            "object": object_name,
        }
        out.append(sse_json(tool_obj))

    final_obj = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "delta": {},
            }
        ],
        "created": created,
        "id": chat_id,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "object": object_name,
    }
    out.append(sse_json(final_obj))
    out.append("data: [DONE]\n\n")
    return out

async def stream_compat_response(upstream_response: httpx.Response):
    raw_buffer = ""
    saw_structured_tool_calls = False

    try:
        async for line in upstream_response.aiter_lines():
            if line == "":
                continue

            if not line.startswith("data: "):
                yield line + "\n\n"
                continue

            data = line[len("data: "):]

            if data == "[DONE]":
                yield "data: [DONE]\n\n"
                continue

            try:
                obj = json.loads(data)

                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        continue

                    if delta.get("tool_calls") is not None:
                        saw_structured_tool_calls = True

                    raw_buffer += collect_text_fields(delta)

                if not saw_structured_tool_calls:
                    calls = parse_raw_tool_calls(raw_buffer)
                    if calls:
                        for chunk in build_stream_tool_call_chunks(obj, calls):
                            yield chunk
                        return

                for choice in obj.get("choices", []):
                    if choice.get("finish_reason") == "stop" and saw_structured_tool_calls:
                        choice["finish_reason"] = "tool_calls"

                yield sse_json(obj)

            except Exception:
                yield line + "\n\n"
    except Exception as e:
        import logging
        logging.error(f"Upstream read error: {repr(e)}")
        return

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    upstream_url = f"{UPSTREAM_BASE}/{path}"
    headers = strip_hop_by_hop_headers(dict(request.headers))
    body_bytes = await request.body()
    method = request.method

    is_chat_completions = path.endswith("v1/chat/completions") or path.endswith("chat/completions")
    is_stream = False
    json_body = None

    if body_bytes and "application/json" in request.headers.get("content-type", ""):
        try:
            json_body = json.loads(body_bytes.decode("utf-8"))
            is_stream = bool(json_body.get("stream", False))
        except Exception:
            json_body = None

    client = global_client
    if True:
        if is_chat_completions and is_stream and json_body is not None:
            stream_ctx = client.stream(
                method,
                upstream_url,
                headers=headers,
                json=json_body,
            )
            upstream_response = await stream_ctx.__aenter__()

            async def generator():
                try:
                    async for chunk in stream_compat_response(upstream_response):
                        yield chunk
                finally:
                    try:
                        await stream_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass

            return StreamingResponse(
                generator(),
                status_code=upstream_response.status_code,
                media_type="text/event-stream",
            )

        upstream_response = await client.request(
            method,
            upstream_url,
            headers=headers,
            content=body_bytes,
        )

        content_type = upstream_response.headers.get("content-type", "")

        if is_chat_completions and "application/json" in content_type:
            try:
                obj = upstream_response.json()
                obj = convert_non_streaming_response(obj)
                return JSONResponse(content=obj, status_code=upstream_response.status_code)
            except Exception:
                pass

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=strip_hop_by_hop_headers(dict(upstream_response.headers)),
            media_type=content_type or None,
        )
