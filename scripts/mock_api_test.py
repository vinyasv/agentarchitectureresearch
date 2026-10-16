"""Exercise the real SDK code path against a local fake Messages endpoint.

Checks request serialization (output_config, replayed content blocks including thinking
blocks), response parsing and usage accounting, all without an API key or spend.
"""

import asyncio
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUESTS = []


def _text_of(msg):
    c = msg["content"]
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if b.get("type") == "text")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        REQUESTS.append(body)
        schema = (body.get("output_config") or {}).get("format", {}).get("schema", {})
        props = schema.get("properties", {})
        last = _text_of(body["messages"][-1])
        if "shared_fact_ids" in props:
            m = re.search(r"Facts available to share this turn:\n\[(\d+)\] (.*)", last)
            text = json.dumps({"shared_fact_ids": [int(m.group(1))] if m else [],
                               "message": m.group(2) if m else "go ahead"})
        elif "spec" in props:
            n_user = sum(1 for x in body["messages"] if x["role"] == "user")
            text = json.dumps({"spec": {"objective": "o", "requirements": ["r"], "assumptions": [],
                                        "open_questions": []},
                               "ready_to_execute": n_user >= 3, "message_to_user": "Lead-in."})
        else:
            text = "```sql\nSELECT 1\n```" if "SQLite" in body["system"] else "Noted. What else?"
        resp = {"id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "thinking", "thinking": "", "signature": "sig"},
                            {"type": "text", "text": text}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1000, "output_tokens": 200}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{srv.server_port}"
    os.environ["ANTHROPIC_API_KEY"] = "test-key-for-local-mock"

    from convarch.runner import Config, run_episode, _make_llms
    from convarch.llm import make_client
    from convarch.tasks import all_tasks

    task = next(t for t in all_tasks() if t.id == "sql_unsold_electronics")
    cfg = Config(agent_model="claude-sonnet-5-5", user_model="claude-haiku-5-5")

    async def go():
        llms = _make_llms(cfg, make_client(max_retries=0), asyncio.Semaphore(4))
        return [await run_episode(task, c, 0, cfg, *llms) for c in ["single", "two_agent"]]

    for r in asyncio.run(go()):
        print(f"{r['condition']}: turns={r['n_assistant_turns']} attempts={r['n_attempts']} "
              f"cost=${r['cost_usd']:.4f} usage_calls={ {k: v['calls'] for k, v in r['usage'].items()} }")
    replayed = [b for req in REQUESTS for m in req["messages"] if m["role"] == "assistant"
                for b in (m["content"] if isinstance(m["content"], list) else [])]
    assert any(b["type"] == "thinking" for b in replayed), "thinking blocks were not replayed"
    assert all("effort" in req["output_config"] for req in REQUESTS)
    print(f"{len(REQUESTS)} requests OK; thinking blocks replayed; models used:",
          sorted({r['model'] for r in REQUESTS}))
    srv.shutdown()


if __name__ == "__main__":
    main()
