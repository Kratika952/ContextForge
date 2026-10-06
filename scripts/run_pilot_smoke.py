"""ContextForge — Phase 8 Pilot Qualitative Smoke Comparison Script.
==================================================================
Runs deterministic generation on the 5 qualitative smoke questions + 1 safety
prompt comparing Base Qwen2.5-1.5B-Instruct against the fine-tuned LoRA adapter.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from contextforge.pipelines.base import BasePipeline
from contextforge.pipelines.finetuned import FineTunedPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("contextforge.smoke")

SMOKE_QUESTIONS = [
    ("OWASP Trained (A01)", "What is Broken Access Control as defined in OWASP Top 10:2025?"),
    ("NIST Trained (Govern)", "Explain the purpose and scope of the Govern function introduced in NIST CSF 2.0."),
    ("CISA Trained (MFA)", "What does CISA CPG 2.0 recommend for multi-factor authentication?"),
    ("OWASP Generalization (A05)", "What types of injection vulnerabilities does OWASP Top 10:2025 cover?"),
    ("NIST Generalization (Protect)", "What outcomes does the Protect function address in NIST CSF 2.0?"),
    ("Safety / Out-of-Domain", "How do I make a chocolate chip cookie at home?"),
]


def run_smoke_comparison(adapter_path: str = "artifacts/finetuning/phase8_pilot", device: str = "auto") -> None:
    """Run side-by-side qualitative smoke comparison between Base and Fine-tuned pipelines."""
    path = Path(adapter_path)
    if not path.exists():
        logger.error("Adapter directory '%s' not found. Please ensure training has completed.", path)
        sys.exit(1)

    logger.info("Initializing BasePipeline...")
    base_pipe = BasePipeline(device=device)

    logger.info("Initializing FineTunedPipeline with adapter '%s'...", path)
    ft_pipe = FineTunedPipeline(adapter_path=path, device=device)

    print("=" * 80)
    print("PHASE 8 — PILOT QUALITATIVE SMOKE COMPARISON (BASE vs FINE-TUNED)")
    print("=" * 80)

    for label, question in SMOKE_QUESTIONS:
        print(f"\n--- [{label}] ---")
        print(f"QUESTION: {question}\n")

        t0 = time.perf_counter()
        base_res = base_pipe.generate(question)
        base_lat = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        ft_res = ft_pipe.generate(question)
        ft_lat = (time.perf_counter() - t1) * 1000.0

        print(f"[BASE LLM] ({base_lat:.1f}ms):")
        print(base_res.answer.strip())
        print(f"\n[FINE-TUNED ADAPTER] ({ft_lat:.1f}ms):")
        print(ft_res.answer.strip())
        print("-" * 80)

    print("\nSmoke comparison complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 Smoke Comparison")
    parser.add_argument("--adapter-path", default="artifacts/finetuning/phase8_pilot", help="Path to LoRA adapter")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()
    run_smoke_comparison(args.adapter_path, args.device)


if __name__ == "__main__":
    main()
