#!/usr/bin/env python3
import html
import json
import logging
import os
import re
import uuid
import asyncio

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
log = logging.getLogger("proxy")

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8000")
HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("PROXY_PORT", "9526"))

HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "content-length", "content-type"})

DSML_BAR = chr(0xFF5C)
DSML_OPEN = "<" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"
DSML_CLOSE = "</" + DSML_BAR + "DSML" + DSML_BAR + "tool_calls>"
SECTION_SIZE = 32
GUARD_SECTIONS = 2

app = FastAPI()

def strip_hop_by_hop_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}

def normalize_arg_value(val):
    if isinstance(val, str):
        val = html.unescape(val)
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)): return json.dumps(parsed, ensure_ascii=False)
        except Exception: pass
        return val
    return json.dumps(val, ensure_ascii=False)

def make_tool_call(name, arguments, call_id=None):
    return {"id": call_id or "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": normalize_arg_value(arguments)}}

# ---------------------------------------------------------
# LÓGICA DE PARSEO DSML (ESTABLE)
# ---------------------------------------------------------
def has_complete_raw_tool_block(text):
    if re.search(r"</[^>]*?(?:invoke|calls)\s*>", text, re.IGNORECASE): return True
    return False

def find_raw_tool_start(text):
    match = re.search(r"(?:invoke|calls)\s+(?:name\s*=|[^>]*>)", text, re.IGNORECASE)
    if match: return max(0, match.start() - 20)
    return len(text)

def has_any_dsml_prefix(text):
    if not text: return False
    if re.search(r"(?:invoke|calls|tool_calls)\s+(?:name\s*=|[^>]*>)", text, re.IGNORECASE): return True
    if "DSML" in text or DSML_BAR in text:
        if "name=" in text or "invoke" in text: return True
    return False

def parse_dsml_tool_calls(text):
    results = []
    invoke_pat = r"(?:invoke|tool_calls|calls)\s+name\s*=\s*[\"']([^\"']+)[\"']"
    for inv in re.finditer(invoke_pat, text, re.IGNORECASE):
        fn_name = inv.group(1).strip()
        next_inv = re.search(invoke_pat, text[inv.end():], re.IGNORECASE)
        block_end = inv.end() + next_inv.start() if next_inv else len(text)
        param_block = text[inv.end():block_end]
        params = {}
        param_pat = r"(?:parameter)\s+name\s*=\s*[\"']([^\"']+)[\"'][^>]*?>(.*?)(?:<\/[^>]*?parameter\s*>|$)"
        for p in re.finditer(param_pat, param_block, re.DOTALL | re.IGNORECASE):
            param_val = re.sub(r"</[^>]*?>", "", p.group(2).strip()).strip()
            params[p.group(1).strip()] = param_val
        if params or not param_block.strip():
            results.append(make_tool_call(fn_name, json.dumps(params, ensure_ascii=False)))
    return results

def parse_raw_tool_calls(text):
    return parse_dsml_tool_calls(text)

# ---------------------------------------------------------
# LÓGICA DE LIMPIEZA Y UI (ESTABLE)
# ---------------------------------------------------------
def clean_dsml_garbage(text):
    if not text: return ""
    lines = []
    for line in text.split('\n'):
        has_dsml = "DSML" in line
        has_pipe = "|" in line or "｜" in line
        has_angle = "<" in line or ">" in line
        if has_dsml and has_pipe and has_angle: continue
        if has_dsml and ("invoke" in line or "calls" in line or "parameter" in line): continue
        lines.append(line)
    return "\n".join(lines).strip()

def format_tool_calls_for_display(tcs):
    if not tcs: return ""
    md = "\n\n> 🛠️ **Acción del agente:**\n"
    for tc in tcs:
        md += f"> - Ejecutando `{tc['function']['name']}`\n"
        try:
            args = json.loads(tc['function']['arguments'])
            for k, v in args.items():
                val_str = str(v).replace('\n', ' ')
                md += f">   - *{k}*: `{val_str[:77]}...`\n" if len(val_str) > 80 else f">   - *{k}*: `{val_str}`\n"
        except Exception: pass
    return md + "\n"

def sse_json(line):
    if not line.startswith("data: "): return None
    payload = line[6:]
    if payload.strip() == "[DONE]": return {"done": True}
    try: return json.loads(payload)
    except Exception: return {"raw": payload, "done": False}

def _make_content_sse(chunk_id, model, text):
    return "data: " + json.dumps({"id": chunk_id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": text, "tool_calls": []}}]}, ensure_ascii=False) + "\n\n"

def build_stream_tool_call_chunks(tool_calls, chunk_id, model):
    chunks = []
    for tc in tool_calls:
        chunks.append({"id": chunk_id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": ""}}]}, "finish_reason": None}]})
        args = tc["function"]["arguments"]
        for i in range(0, len(args), 32):
            chunks.append({"id": chunk_id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[i:i+32]}}]}, "finish_reason": None}]})
    chunks.append({"id": chunk_id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    return chunks

# ---------------------------------------------------------
# STREAMING PRINCIPAL
# ---------------------------------------------------------
async def stream_with_sections(upstream_req, forwarded_for="", auth_header=None, user_agent=None):
    chunk_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    model = upstream_req.get("model", "deepseek")

    req_headers = {"Accept": "text/event-stream"}
    if auth_header: req_headers["Authorization"] = auth_header
    if user_agent: req_headers["User-Agent"] = user_agent
    if forwarded_for: req_headers["x-forwarded-for"] = forwarded_for

    client = httpx.AsyncClient(timeout=None)
    try:
        resp = await client.send(client.build_request("POST", UPSTREAM + "/v1/chat/completions", json=upstream_req, headers=req_headers), stream=True)
    except Exception as e:
        await client.aclose()
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})

    if resp.status_code != 200:
        err_content = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return Response(content=err_content, status_code=resp.status_code, media_type="application/json")

    async def generate():
        nonlocal chunk_id, model
        buffer, unflushed_text, pending = "", "", []
        dsml_mode, content_collected, thinking_state = False, False, 0

        try:
            async for raw_line in resp.aiter_lines():
                if not raw_line: continue

                ev = sse_json(raw_line)
                if ev is None:
                    yield raw_line + "\n\n"
                    continue
                if ev.get("done"): continue

                choices = ev.get("choices", [])
                if not choices: continue

                delta = choices[0].get("delta", {})
                if ev.get("id"): chunk_id = ev["id"]
                if ev.get("model"): model = ev["model"]

                if "role" in delta:
                    yield raw_line + "\n\n"
                    continue

                reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                text = delta.get("content", "")

                is_reason = bool(reasoning)
                raw_chunk_text = reasoning if is_reason else text
                if not raw_chunk_text:
                    if not reasoning and not dsml_mode: yield raw_line + "\n\n"
                    continue

                chunk_text = ""
                if is_reason:
                    if thinking_state == 0: chunk_text += "<think>\n"; thinking_state = 1
                    chunk_text += raw_chunk_text
                else:
                    if thinking_state == 1: chunk_text += "\n</think>\n"; thinking_state = 2
                    chunk_text += raw_chunk_text

                buffer += chunk_text
                content_collected = True

                if dsml_mode:
                    if has_complete_raw_tool_block(buffer):
                        if thinking_state == 1: thinking_state = 2

                        tcs = parse_raw_tool_calls(buffer) 
                        formatted_calls = format_tool_calls_for_display(tcs)
                        
                        if formatted_calls:
                            yield _make_content_sse(chunk_id, model, formatted_calls)

                        if tcs:
                            for tc in build_stream_tool_call_chunks(tcs, chunk_id, model):
                                yield "data: " + json.dumps(tc) + "\n\n"
                        else:
                            log.warning("Parse empty for DSML block")
                        
                        yield "data: [DONE]\n\n" 
                        return
                    continue

                if has_any_dsml_prefix(buffer):
                    dsml_mode = True
                    all_pending_text = "".join(pending) + unflushed_text
                    
                    idx = find_raw_tool_start(all_pending_text)
                    if idx > 0:
                        text_to_yield = clean_dsml_garbage(all_pending_text[:idx])
                        if text_to_yield:
                            yield _make_content_sse(chunk_id, model, text_to_yield + "\n\n")
                            
                    pending.clear()
                    unflushed_text = ""
                    
                    idx_buffer = find_raw_tool_start(buffer)
                    if idx_buffer != -1 and idx_buffer < len(buffer):
                        buffer = buffer[idx_buffer:]
                    else:
                        buffer = chunk_text
                    continue

                unflushed_text += chunk_text
                while len(unflushed_text) >= SECTION_SIZE:
                    pending.append(unflushed_text[:SECTION_SIZE])
                    unflushed_text = unflushed_text[SECTION_SIZE:]
                    if len(pending) > GUARD_SECTIONS:
                        yield _make_content_sse(chunk_id, model, pending.pop(0))

            if thinking_state == 1: unflushed_text += "\n</think>\n"; thinking_state = 2

            if dsml_mode:
                if content_collected:
                    idx = find_raw_tool_start(buffer)
                    raw_prefix = buffer[:idx] if idx > 0 else ""
                    code_to_parse = buffer[idx:] if idx > 0 else buffer
                    
                    tcs = parse_raw_tool_calls(code_to_parse)
                    formatted_calls = format_tool_calls_for_display(tcs)
                    
                    prefix = clean_dsml_garbage(raw_prefix)
                    
                    if prefix or formatted_calls:
                        yield _make_content_sse(chunk_id, model, prefix + ("\n\n" if prefix else "") + formatted_calls)

                    if tcs:
                        for tc in build_stream_tool_call_chunks(tcs, chunk_id, model):
                            yield "data: " + json.dumps(tc) + "\n\n"
                    else:
                        clean_buffer = clean_dsml_garbage(buffer)
                        if clean_buffer and not clean_buffer.strip() == "": 
                            yield _make_content_sse(chunk_id, model, clean_buffer)
            else:
                for s in pending:
                    yield _make_content_sse(chunk_id, model, s)
                if unflushed_text:
                    yield _make_content_sse(chunk_id, model, unflushed_text)

            yield "data: " + json.dumps({"id": chunk_id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": "", "tool_calls": []}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
            
        except asyncio.CancelledError:
            log.info("Cliente desconectado")
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.api_route("/v1/chat/completions", methods=["POST"])
async def proxy(request: Request):
    body_bytes = await request.body()
    j = None
    try:
        j = json.loads(body_bytes)
        if j and "tools" in j and isinstance(j["tools"], list):
            j["tools"] = [t for t in j["tools"] if isinstance(t, dict) and t.get("type") == "function"]
            if not j["tools"]: j.pop("tools")
    except Exception: pass

    client_host = request.client.host if request.client else "unknown"
    upstream_headers = strip_hop_by_hop_headers(dict(request.headers))
    upstream_headers.pop("host", None)
    upstream_headers["content-type"] = "application/json"
    if not upstream_headers.get("x-forwarded-for"): upstream_headers["x-forwarded-for"] = client_host

    if j and j.get("stream", False):
        return await stream_with_sections(j, client_host, request.headers.get("Authorization"), request.headers.get("User-Agent"))

    async with httpx.AsyncClient() as client:
        upstream_resp = await client.post(UPSTREAM + "/v1/chat/completions", json=j if j else json.loads(body_bytes), headers=upstream_headers, timeout=None)
    return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, headers=dict(upstream_resp.headers))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
