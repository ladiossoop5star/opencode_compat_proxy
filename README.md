# OpenCode Compatibility Proxy

A FastAPI proxy that translates non-standard tool call formats (DeepSeek DSML, Qwen XML) from LLM backends into OpenAI-compatible `tool_calls` structure, enabling [OpenCode](https://opencode.ai) to work with models that output raw markup instead of structured JSON.

## Problem

Many open-source LLMs (DeepSeek, Qwen, and their fine-tunes) output tool calls as raw XML or DSML tags embedded in the text content — for example:

```
<tool_call>
<function=search>
<parameter=query>hello</parameter>
</function>
</tool_call>
```

OpenCode (and most OpenAI-compatible clients) expects tool calls in structured JSON format (`tool_calls` array in the response). Without a conversion layer, these models are incompatible with OpenCode's function calling feature — OpenCode cannot parse the raw markup, so it hangs indefinitely waiting for the assistant to respond, effectively treating it as waiting for user input. This breaks any automated workflow that depends on tool-call-driven agent loops.

This proxy solves that by sitting between OpenCode and the LLM backend, intercepting the response, and converting raw markup into standard OpenAI `tool_calls` on the fly — without modifying either the client or the model.

## Architecture

```
OpenCode  →  Proxy (e.g. :9526)  →  LLM Backend (e.g. :8004)
```

All ports and addresses shown below are examples — adjust them to match your environment.

The proxy sits between OpenCode and the LLM backend (e.g. vLLM, llama.cpp). It intercepts streaming/non-streaming responses, detects raw tool-call tags (`<tool_call>` or `<｜DSML｜tool_calls>`), and converts them on-the-fly into the standard OpenAI `tool_calls` JSON format.

## Features

- **Streaming Support** — Uses a persistent `httpx.AsyncClient` to avoid connection drops
- **DeepSeek DSML** — Automatically detects and converts `<｜DSML｜tool_calls>` blocks
- **Qwen XML** — Automatically detects and converts `<tool_call>` blocks
- **Native Passthrough** — If the upstream model already outputs standard `delta.tool_calls`, the proxy passes it through unchanged
- **Timeout-free** — Sets timeout to `None` for long LLM inference sessions
- **SSE Compression Fix** — Strips `Accept-Encoding` header to prevent upstream GZIP/Deflate compression from breaking the stream parser

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

Set the `UPSTREAM_BASE` environment variable (defaults to `http://127.0.0.1:8004`):

```bash
export UPSTREAM_BASE="http://127.0.0.1:8004"
```

### 3. Start the proxy

```bash
uvicorn proxy:app --host 0.0.0.0 --port 9526
```

### 4. Configure OpenCode

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
ExecStart=/path/to/venv/bin/uvicorn proxy:app --host 0.0.0.0 --port 9526
WorkingDirectory=/path/to/project
Restart=always

[Install]
WantedBy=multi-user.target
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_BASE` | `http://127.0.0.1:8004` | Base URL of the upstream LLM backend |

## How It Works

1. OpenCode sends a standard OpenAI chat completions request to the proxy
2. The proxy forwards the request to the upstream LLM backend
3. The proxy inspects the response for raw tool-call markup in the text content
4. If non-standard markup is found, it parses the tags and emits structured `tool_calls` chunks
5. The converted response is sent back to OpenCode in standard OpenAI format

## License

MIT
