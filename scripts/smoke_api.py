"""Smoke test script for ContextForge FastAPI server.
===================================================
Tests GET /health and 3 sequential POST /ask requests (Base, RAG, Fine-Tuned).
Uses standard library urllib.request for direct HTTP calls.
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

QUESTIONS = [
    {
        "pipeline": "base",
        "question": "What does the principle of least privilege mean in cybersecurity?",
    },
    {
        "pipeline": "rag",
        "question": "What is the Govern function in NIST CSF 2.0?",
    },
    {
        "pipeline": "finetuned",
        "question": "How does multi-factor authentication prevent unauthorized account access?",
    },
]


def wait_for_server(timeout=180):
    print("Waiting for ContextForge server to respond on /health...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    print(f"Server is ready! Health: {data}")
                    return True
        except Exception:
            time.sleep(1)
    print("Server failed to respond within timeout.")
    return False


def run_smoke():
    if not wait_for_server():
        sys.exit(1)

    results = []
    for item in QUESTIONS:
        pipeline = item["pipeline"]
        question = item["question"]
        print(f"\n--- Testing Pipeline: {pipeline.upper()} ---")
        print(f"Question: {question}")

        payload = json.dumps({"question": question, "pipeline": pipeline}).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/ask",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                elapsed = (time.time() - t0) * 1000
                data = json.loads(resp.read().decode("utf-8"))
                print(f"Status: {resp.status}")
                print(f"Client elapsed: {elapsed:.1f}ms | API latency: {data.get('latency_ms', 0):.1f}ms")
                print(f"Model ID: {data.get('model_id')}")
                print(f"Sources count: {len(data.get('sources', []))}")
                print(f"Answer preview: {data.get('answer', '')[:160]}...")
                results.append({"pipeline": pipeline, "status": resp.status, "data": data})
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            print(f"HTTP ERROR {e.code}: {err_body}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    print("\n=== SMOKE COMPLETED SUCCESSFULLY ===")
    return results


if __name__ == "__main__":
    run_smoke()
