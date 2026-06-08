# OpenCode Compatibility Proxy

A FastAPI proxy that translates non-standard tool call formats (DeepSeek DSML, Qwen XML) from LLM backends into OpenAI-compatible `tool_calls` structure, enabling [OpenCode](https://opencode.ai) to work with models that output raw markup instead of structured JSON.

## Problem

Many open-source LLMs (DeepSeek, Qwen, and their fine-tunes) output tool calls as raw XML or DSML tags embedded in the text content — for example:

```
<｜DSML｜tool_calls>
<name>search</name>
<parameters>{"query": "hello"}</parameters>
</｜DSML｜tool_calls>
```

OpenCode (and most OpenAI-compatible clients) expects tool calls in structured JSON format (`tool_calls` array in the response). Without a conversion layer, these models are incompatible with OpenCode's function calling feature — OpenCode cannot parse the raw markup, so it hangs indefinitely waiting for the assistant to respond, effectively treating it as waiting for user input. This breaks any automated workflow that depends on tool-call-driven agent loops.

This proxy solves that by sitting between OpenCode and the LLM backend, intercepting the response, and converting raw markup into standard OpenAI `tool_calls` on the fly — without modifying either the client or the model.

## Architecture

```
OpenCode  →  Proxy (e.g. :9526)  →  LLM Backend (e.g. :8000)
```

All ports and addresses shown below are examples — adjust them to match your environment.

The proxy sits between OpenCode and the LLM backend (e.g. vLLM, llama.cpp). It intercepts streaming/non-streaming responses, detects raw tool-call tags (`<｜DSML｜tool_calls>` or `<tool_call>`), and converts them on-the-fly into the standard OpenAI `tool_calls` JSON format.

## Supported Tool Call Formats

| Format | Tags | Example |
|--------|------|---------|
| DeepSeek DSML (name/parameters) | `<｜DSML｜tool_calls>` | `<name>ls</name><parameters>{"path":"."}</parameters>` |
| DeepSeek DSML (invoke/parameter) | `<｜DSML｜tool_calls>` | `<｜DSML｜invoke name="ls"><｜DSML｜parameter name="path">.</｜DSML｜parameter></｜DSML｜invoke>` |
| Qwen XML | `<tool_call>` | `<name>ls</name><parameters>{"path":"."}</parameters>` |
| OpenAI standard | `tool_calls` array | Passed through unchanged |

## Features

- **DeepSeek DSML** — Detects and converts both `<name>/<parameters>` and `<invoke>/<parameter>` formats
- **Qwen XML** — Detects and converts `<tool_call>` blocks
- **vLLM compat** — Strips spurious empty `tool_calls: []` arrays that vLLM sends on every content chunk
- **Reasoning content** — Accumulates `reasoning`/`reasoning_content` fields and still catches tool calls within them
- **Chunked streaming** — Streams function arguments in small chunks so OpenCode receives progressive JSON
- **Native passthrough** — If the upstream already outputs standard `delta.tool_calls`, passes it through unchanged
- **SSE compression fix** — Strips `Accept-Encoding` header to prevent upstream GZIP/Deflate from breaking the stream parser

## Installation

### Option A — Manual

```bash
git clone https://github.com/ladiossoop5star/opencode_compat_proxy.git
cd opencode_compat_proxy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option B — Let AI do it

Give this entire README to any capable AI coding assistant (Claude, ChatGPT, etc.) with a prompt like:

> Read this README and install the project on my machine.

The AI will handle cloning, setting up the environment, and starting the proxy.

## Usage

### 1. Configure upstream backend

Set the `UPSTREAM_URL` environment variable (defaults to `http://127.0.0.1:8000`):

```bash
export UPSTREAM_URL="http://127.0.0.1:8000"
```

### 2. Start the proxy

```bash
python proxy.py
```

Or via uvicorn directly:

```bash
uvicorn proxy:app --host 0.0.0.0 --port 9526
```

### 3. Configure OpenCode

Add a provider entry pointing to the proxy in your `opencode.jsonc`:

```jsonc
"provider": {
    "my-model": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "My Model",
      "options": {
        "baseURL": "http://127.0.0.1:9526/v1",
        "apiKey": "dummy"
      },
      "models": {
        "my-model": {
          "name": "My Model",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
```

### Run as a systemd service (optional)

```
[Unit]
Description=OpenCode Compat Proxy
After=network.target

[Service]
Type=simple
ExecStart=/path/to/venv/bin/python /path/to/proxy.py
WorkingDirectory=/path/to/project
Restart=always

[Install]
WantedBy=multi-user.target
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_URL` | `http://127.0.0.1:8000` | Base URL of the upstream LLM backend |
| `PROXY_HOST` | `0.0.0.0` | Address to bind the proxy to |
| `PROXY_PORT` | `9526` | Port to listen on |

## How It Works (Streaming)

1. Proxy receives the request, opens a streaming connection to the upstream LLM backend
2. Each SSE chunk is inspected:
   - Empty `tool_calls: []` from vLLM are stripped
   - Role-only and empty-content chunks are cleaned up
   - `content` and `reasoning`/`reasoning_content` fields are accumulated into a raw buffer
3. When the raw buffer contains a complete DSML or Qwen XML tool call block, the proxy:
   - Parses the block into structured `tool_calls`
   - Emits a role chunk, then function name + chunked argument chunks, then a final `finish_reason: "tool_calls"` chunk
   - Sends `[DONE]` and closes the stream
4. If no raw tool call is detected, the stream passes through normally

### Non-Streaming

The proxy checks `message.content` for a complete raw tool call block. If found, it parses and converts it to `tool_calls` in the JSON response.

## Testing

```bash
# List models
curl http://127.0.0.1:9526/v1/models

# Non-streaming chat
curl -s http://127.0.0.1:9526/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hi"}]}'

# Streaming chat
curl -sN http://127.0.0.1:9526/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hi"}],"stream":true}'

# Tool call test
curl -sN http://127.0.0.1:9526/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"list files"}],"stream":true,"tools":[{"type":"function","function":{"name":"ls","description":"List directory","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}'
```

## License

MIT
