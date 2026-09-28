"""
ContextForge Base LLM Smoke Inference Runner.
==============================================
Runs the required representative defensive cybersecurity smoke questions
and records exact latency, input/output tokens, and generated text.
Loads model once per process.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from contextforge.pipelines.base import BasePipeline

QUESTIONS = [
    "What is broken access control?",
    "What is the purpose of multi-factor authentication?",
    "What is the Govern function in a cybersecurity framework?",
    "How can SQL injection risk be reduced?",
    "What is the difference between authentication and authorization?",
    # Corpus-specific baseline question (OWASP Top 10:2025 specific categories)
    "What are the official ten vulnerability categories defined in OWASP Top 10:2025?",
]


def main() -> None:
    print("Initializing BasePipeline...")
    t0 = time.perf_counter()
    pipeline = BasePipeline(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        device="auto",
        dtype="auto",
        max_new_tokens=256,
    )
    # Trigger model load
    _ = pipeline.model_loader.model
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s (Device: {pipeline.model_loader.device}, Dtype: {pipeline.model_loader.dtype})")

    results = []
    for idx, q in enumerate(QUESTIONS, start=1):
        print(f"\n[{idx}/{len(QUESTIONS)}] Question: {q}")
        res = pipeline.generate(q)
        print(f"  Latency: {res.latency_ms:.1f}ms | In tokens: {res.metadata['input_token_count']} | Out tokens: {res.metadata['output_token_count']}")
        print(f"  Answer preview: {res.answer[:160]}...")
        results.append(
            {
                "index": idx,
                "question": q,
                "answer": res.answer,
                "latency_ms": res.latency_ms,
                "input_token_count": res.metadata["input_token_count"],
                "output_token_count": res.metadata["output_token_count"],
                "device": res.metadata["device"],
                "dtype": res.metadata["dtype"],
            }
        )

    # Save to JSON for report reference
    out_path = Path("artifacts/base_smoke_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll smoke questions completed. Saved to {out_path}")


if __name__ == "__main__":
    main()
