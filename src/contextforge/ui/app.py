"""ContextForge — Streamlit Frontend Application.
==============================================
Phase 11: Decoupled interactive UI and evaluation dashboard.

Architecture rule:
This UI module acts STRICTLY as an HTTP client to the ContextForge FastAPI backend.
It NEVER directly loads language models, adapters, FAISS indices, or CUDA code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from contextforge.ui.client import (
    ApiConnectionError,
    ApiError,
    ApiResponseError,
    ContextForgeClient,
    DEFAULT_API_URL,
)

# ---------------------------------------------------------------------------
# Frozen Phase 9 Benchmark Metrics (Ground Truth Constants)
# Evaluated on 9 held-out supported cybersecurity questions under BF16 CUDA.
# ---------------------------------------------------------------------------
FROZEN_PHASE9_METRICS: dict[str, Any] = {
    "limitation": "9 held-out supported cybersecurity questions only.",
    "pipelines": {
        "Base LLM": {
            "pipeline_key": "base",
            "token_f1": 0.2870,
            "semantic_similarity": 0.8641,
            "mean_latency_ms": 8784.4,
            "input_tokens_mean": 64.1,
            "output_tokens_mean": 177.2,
        },
        "RAG Pipeline": {
            "pipeline_key": "rag",
            "token_f1": 0.3159,
            "semantic_similarity": 0.8715,
            "mean_latency_ms": 11837.3,
            "input_tokens_mean": 1346.8,
            "output_tokens_mean": 173.7,
            "hit_at_5": 1.0000,
            "mrr": 0.7722,
            "citation_presence": 0.3333,
            "citation_validity": 1.0000,
        },
        "Fine-Tuned LLM": {
            "pipeline_key": "finetuned",
            "token_f1": 0.3274,
            "semantic_similarity": 0.8692,
            "mean_latency_ms": 5633.3,
            "input_tokens_mean": 64.1,
            "output_tokens_mean": 73.0,
        },
    },
    "interpretation": {
        "finetuned": "Fine-Tuned achieved the highest token F1 (0.3274) and lowest mean latency (5633.3 ms).",
        "rag": "RAG achieved the highest semantic similarity (0.8715) and provides retrieval/document grounding with 100% citation validity.",
        "base": "Base LLM serves as the unadapted baseline (Token F1: 0.2870, Mean latency: 8784.4 ms).",
    },
}

PIPELINE_OPTIONS = {
    "Base LLM": "base",
    "RAG Pipeline": "rag",
    "Fine-Tuned LLM": "finetuned",
}


def get_client(api_url: str) -> ContextForgeClient:
    """Return a configured ContextForgeClient instance."""
    return ContextForgeClient(base_url=api_url)


def render_sidebar() -> tuple[str, str]:
    """Render sidebar navigation and API connectivity settings."""
    st.sidebar.title("⚙️ ContextForge")
    st.sidebar.markdown("**Architecture:** Decoupled Client")

    page = st.sidebar.radio(
        "Navigation",
        ["💬 Domain Assistant", "📊 Research Results"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backend Configuration")
    default_url = os.getenv("CONTEXTFORGE_API_URL", DEFAULT_API_URL).strip()
    api_url = st.sidebar.text_input(
        "FastAPI Server URL",
        value=default_url,
        help="Target URL for the ContextForge FastAPI inference backend.",
    ).strip()

    if st.sidebar.button("Test API Connection", use_container_width=True):
        client = get_client(api_url)
        try:
            health = client.check_health()
            st.sidebar.success(f"Connected (status: {health.get('status', 'ok')})")
        except ApiConnectionError as e:
            st.sidebar.error(f"Cannot connect: {e}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small>Domain: **Cybersecurity**<br>"
        "Corpus: OWASP Top 10, NIST CSF 2.0, CISA CPG 2.0<br>"
        "Model: Qwen2.5-1.5B-Instruct</small>",
        unsafe_allow_html=True,
    )
    return page, api_url


def collapse_display_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Presentation-only duplicate collapsing for identical (document_id, excerpt) chunks.

    Does NOT modify the underlying retrieval results or ranking.
    Collapses exact duplicate cards when normalized excerpt + document ID are identical.
    """
    display_sources: list[dict[str, Any]] = []
    seen_excerpts: dict[tuple[str, str], int] = {}

    for src in sources:
        doc_id = src.get("document_id") or src.get("doc_id") or "Unknown"
        text_snippet = (src.get("text") or src.get("content") or "").strip()
        norm_text = " ".join(text_snippet.split())
        key = (doc_id, norm_text)

        if key not in seen_excerpts:
            seen_excerpts[key] = len(display_sources)
            item = dict(src)
            item["_collapse_count"] = 1
            labels = [src["source_label"]] if src.get("source_label") else []
            item["_labels"] = labels
            display_sources.append(item)
        else:
            existing_idx = seen_excerpts[key]
            display_sources[existing_idx]["_collapse_count"] += 1
            lbl = src.get("source_label")
            if lbl and lbl not in display_sources[existing_idx]["_labels"]:
                display_sources[existing_idx]["_labels"].append(lbl)

    return display_sources


def extract_source_metadata(src: dict[str, Any]) -> dict[str, Any]:
    """Extract display fields from a retrieved source chunk without fake placeholders.

    Prefers real document title, real document_id, section_heading, page ranges,
    and relevance score. Never returns 'Unknown' or 'N/A' when real data exists.
    """
    src_meta = src.get("metadata") or {}
    doc_id = src.get("document_id") or src.get("doc_id")
    title = (
        src_meta.get("document_title")
        or src_meta.get("title")
        or src.get("title")
        or doc_id
        or "Source Document"
    )
    source_label = src.get("source_label") or ""
    section = src_meta.get("section_heading") or src.get("section")
    if section and str(section).strip().upper() in ("NONE", "N/A", "NULL"):
        section = None

    page_start = src_meta.get("page_start") or src.get("page")
    page_end = src_meta.get("page_end")
    page_str = None
    if page_start is not None and str(page_start).strip().upper() not in ("NONE", "N/A", "NULL"):
        if page_end is not None and page_end != page_start:
            page_str = f"{page_start}–{page_end}"
        else:
            page_str = str(page_start)

    score = src.get("score")
    excerpt = (src.get("text") or src.get("content") or "").strip()

    return {
        "doc_id": doc_id,
        "title": title,
        "source_label": source_label,
        "section": section,
        "page": page_str,
        "score": score,
        "excerpt": excerpt,
    }


def extract_response_metrics(
    response: dict[str, Any],
    pipeline_key: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Extract real latency, pipeline, token counts, and caption metadata.

    Only includes fields that have real values (no 'N/A' placeholders).
    """
    meta = response.get("metadata") or {}
    latency_ms = response.get("latency_ms", 0.0)
    model_id = response.get("model_id") or meta.get("model_id")
    prompt_version = response.get("prompt_version") or meta.get("prompt_version")

    token_usage = response.get("token_usage") or {}
    in_tokens = (
        meta.get("input_token_count")
        if meta.get("input_token_count") is not None
        else meta.get("input_tokens")
        if meta.get("input_tokens") is not None
        else token_usage.get("input_tokens")
    )
    out_tokens = (
        meta.get("output_token_count")
        if meta.get("output_token_count") is not None
        else meta.get("output_tokens")
        if meta.get("output_tokens") is not None
        else token_usage.get("output_tokens")
    )

    metrics: list[tuple[str, str]] = [
        ("Latency", f"{latency_ms:.1f} ms"),
        ("Pipeline", response.get("pipeline", pipeline_key)),
    ]
    if in_tokens is not None and str(in_tokens).upper() != "N/A":
        metrics.append(("Input Tokens", str(in_tokens)))
    if out_tokens is not None and str(out_tokens).upper() != "N/A":
        metrics.append(("Output Tokens", str(out_tokens)))

    caption_items: list[str] = []
    if model_id and str(model_id).upper() != "UNKNOWN":
        caption_items.append(f"Model ID: `{model_id}`")
    if prompt_version and str(prompt_version).upper() != "N/A":
        caption_items.append(f"Prompt Version: `{prompt_version}`")

    return metrics, caption_items


def render_assistant_page(api_url: str) -> None:
    """Render interactive QA assistant page."""
    st.title("ContextForge")
    st.subheader("Domain-Specific AI Knowledge Assistant")
    st.caption("Current Domain: **Cybersecurity** (OWASP Top 10, NIST CSF 2.0, CISA CPG 2.0)")

    col_pipe, col_info = st.columns([2, 3])
    with col_pipe:
        selected_label = st.selectbox(
            "Select Inference Pipeline",
            options=list(PIPELINE_OPTIONS.keys()),
            index=1,  # Default to RAG
            help="Choose which model pipeline will generate the answer via FastAPI.",
        )
        pipeline_key = PIPELINE_OPTIONS[selected_label]

    with col_info:
        if pipeline_key == "base":
            st.info("ℹ️ **Base LLM:** Zero-shot causal generation without document retrieval.")
        elif pipeline_key == "rag":
            st.info("📚 **RAG Pipeline:** Dense FAISS retrieval over domain corpus with verified citations.")
        else:
            st.info("🎯 **Fine-Tuned LLM:** QLoRA adapter specialized on domain-grounded QA pairs.")

    question = st.text_area(
        "Question",
        placeholder="e.g., How does NIST CSF 2.0 define the Govern function?",
        height=100,
    )

    ask_clicked = st.button("Ask Assistant", type="primary", use_container_width=True)

    if ask_clicked:
        clean_q = question.strip()
        if not clean_q:
            st.warning("Please enter a question before clicking Ask.")
            return

        client = get_client(api_url)
        with st.spinner(f"Querying backend via {selected_label}..."):
            try:
                response = client.ask(question=clean_q, pipeline=pipeline_key)
            except ApiConnectionError as e:
                st.error(
                    f"**Connection Error:** {e}\n\n"
                    f"Please verify that the FastAPI backend is running at `{api_url}`."
                )
                return
            except ApiResponseError as e:
                st.error(f"**API Response Error (HTTP {e.status_code}):** {e.detail}")
                return
            except Exception as e:
                st.error(f"**Unexpected Error:** {e}")
                return

        # Display successful response
        st.markdown("### Answer")
        answer_text = response.get("answer", "")
        st.markdown(answer_text)

        # Performance and metadata metrics (only show fields with real values)
        st.markdown("---")
        metrics, caption_items = extract_response_metrics(response, pipeline_key)
        cols = st.columns(len(metrics))
        for col, (label, val) in zip(cols, metrics):
            col.metric(label, val)

        if caption_items:
            st.caption(" | ".join(caption_items))

        # RAG Sources Display
        sources = response.get("sources", [])
        if pipeline_key == "rag":
            st.markdown("### Retrieved Sources")
            if sources:
                display_sources = collapse_display_sources(sources)

                if len(display_sources) < len(sources):
                    st.caption(
                        f"*{len(sources)} chunks retrieved · {len(display_sources)} unique source excerpts shown*"
                    )
                else:
                    st.caption(f"*{len(sources)} chunks retrieved*")

                for idx, item in enumerate(display_sources, start=1):
                    info = extract_source_metadata(item)
                    score = info["score"]
                    score_str = f" | Score: {score:.4f}" if score is not None else ""

                    header_parts = [f"Source [{idx}]: {info['title']}"]
                    if info["source_label"]:
                        header_parts.append(info["source_label"])
                    collapse_count = item.get("_collapse_count", 1)
                    if collapse_count > 1:
                        header_parts.append(f"({collapse_count} identical chunks)")

                    expander_title = " · ".join(header_parts) + score_str

                    with st.expander(expander_title):
                        doc_text = f"{info['title']}"
                        if info["doc_id"] and info["doc_id"] != info["title"]:
                            doc_text += f" (`{info['doc_id']}`)"
                        st.markdown(f"**Document:** {doc_text}")

                        if info["source_label"]:
                            labels_to_show = item.get("_labels") or [info["source_label"]]
                            if len(labels_to_show) > 1:
                                st.markdown(f"**Source Labels:** {', '.join(labels_to_show)}")
                            else:
                                st.markdown(f"**Source Label:** `{info['source_label']}`")

                        if info["section"]:
                            st.markdown(f"**Section:** {info['section']}")

                        if info["page"]:
                            st.markdown(f"**Page:** {info['page']}")

                        if score is not None:
                            st.markdown(f"**Relevance Score:** {score:.4f}")

                        st.markdown("**Excerpt:**")
                        st.text(info["excerpt"])
            else:
                st.info("No external document sources were retrieved for this query.")


def render_research_page() -> None:
    """Render frozen Phase 9 evaluation results and comparative benchmarks."""
    st.title("ContextForge Research Evaluation")
    st.subheader("Phase 9 Held-Out Three-Way Comparative Benchmark")
    st.caption("Base LLM vs. RAG Pipeline vs. QLoRA Fine-Tuned LLM")

    st.warning(
        "⚠️ **Benchmark Limitation:** Evaluated strictly on 9 held-out supported cybersecurity "
        "questions under identical precision (BF16, CUDA, unquantized base)."
    )

    # Core Metrics Table
    st.markdown("### Comparative Performance Summary")

    p_data = FROZEN_PHASE9_METRICS["pipelines"]
    summary_rows = [
        {
            "Pipeline": "Base LLM",
            "Token F1": f"{p_data['Base LLM']['token_f1']:.4f}",
            "Semantic Similarity": f"{p_data['Base LLM']['semantic_similarity']:.4f}",
            "Mean Latency (ms)": f"{p_data['Base LLM']['mean_latency_ms']:.1f}",
            "Mean Input Tokens": f"{p_data['Base LLM']['input_tokens_mean']:.1f}",
            "Mean Output Tokens": f"{p_data['Base LLM']['output_tokens_mean']:.1f}",
        },
        {
            "Pipeline": "RAG Pipeline",
            "Token F1": f"{p_data['RAG Pipeline']['token_f1']:.4f}",
            "Semantic Similarity": f"{p_data['RAG Pipeline']['semantic_similarity']:.4f}",
            "Mean Latency (ms)": f"{p_data['RAG Pipeline']['mean_latency_ms']:.1f}",
            "Mean Input Tokens": f"{p_data['RAG Pipeline']['input_tokens_mean']:.1f}",
            "Mean Output Tokens": f"{p_data['RAG Pipeline']['output_tokens_mean']:.1f}",
        },
        {
            "Pipeline": "Fine-Tuned LLM",
            "Token F1": f"{p_data['Fine-Tuned LLM']['token_f1']:.4f}",
            "Semantic Similarity": f"{p_data['Fine-Tuned LLM']['semantic_similarity']:.4f}",
            "Mean Latency (ms)": f"{p_data['Fine-Tuned LLM']['mean_latency_ms']:.1f}",
            "Mean Input Tokens": f"{p_data['Fine-Tuned LLM']['input_tokens_mean']:.1f}",
            "Mean Output Tokens": f"{p_data['Fine-Tuned LLM']['output_tokens_mean']:.1f}",
        },
    ]

    st.table(summary_rows)

    # Key Findings Cards
    st.markdown("### Key Scientific Findings")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Fine-Tuned Token F1",
            value="0.3274",
            delta="+0.0404 vs Base",
            help="Fine-Tuned achieved the highest token F1 among all three pipelines.",
        )
        st.markdown(
            "**Fastest Inference:** 5,633.3 ms mean latency (~1.5x faster than Base, "
            "~2.1x faster than RAG) with concise, domain-adapted outputs."
        )

    with c2:
        st.metric(
            label="RAG Semantic Similarity",
            value="0.8715",
            delta="+0.0074 vs Base",
            help="RAG achieved the highest semantic similarity to reference answers.",
        )
        st.markdown(
            "**Grounding & Verification:** Provides explicit document sources with "
            "100% citation validity on emitted citations."
        )

    with c3:
        st.metric(
            label="Base Baseline F1",
            value="0.2870",
            help="Unadapted baseline performance without retrieval or fine-tuning.",
        )
        st.markdown(
            "**Baseline Standard:** Zero-shot performance of Qwen2.5-1.5B-Instruct "
            "prior to domain adaptation."
        )

    st.markdown("---")
    st.markdown("### RAG Retrieval and Citation Quality")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Hit@5", "100%", help="Top 5 retrieved documents contained the relevant source")
    r_col2.metric("MRR", "0.7722", help="Mean Reciprocal Rank of the first relevant document")
    r_col3.metric("Citation Presence", "33.3%", help="Rate of explicit citations in RAG answers")
    r_col4.metric("Citation Validity", "100%", help="Validity rate of emitted citations against retrieved documents")

    st.markdown("---")
    st.markdown("### Qualitative and Architectural Interpretation")
    interp = FROZEN_PHASE9_METRICS["interpretation"]
    st.markdown(f"- 🎯 **Fine-Tuned Pipeline:** {interp['finetuned']}")
    st.markdown(f"- 📚 **RAG Pipeline:** {interp['rag']}")
    st.markdown(f"- ℹ️ **Base LLM Pipeline:** {interp['base']}")
    st.info(
        "💡 **Conclusion:** Fine-Tuning delivers concise, fast responses with superior token overlap, "
        "while RAG offers grounded semantic alignment and source verification for high-assurance tasks."
    )


def main() -> None:
    """Streamlit application main entry point."""
    st.set_page_config(
        page_title="ContextForge — AI Knowledge Assistant",
        page_icon="🛡️",
        layout="wide",
    )

    page, api_url = render_sidebar()

    if page == "💬 Domain Assistant":
        render_assistant_page(api_url)
    else:
        render_research_page()


if __name__ == "__main__":
    main()
