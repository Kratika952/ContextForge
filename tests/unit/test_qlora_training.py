"""Unit tests for Phase 8 QLoRA fine-tuning training components.
================================================================
Validates configuration loading, dataset parsing, chat formatting,
token length calculations, QLoRA/LoRA config construction, and CPU
refusal without loading heavy model weights or GPU training stacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from scripts.train_qlora import (
    DEFAULT_CONFIG_PATH,
    build_bnb_config,
    build_lora_config,
    compute_token_length_stats,
    format_chat_text,
    get_environment_info,
    load_config,
    load_dataset_records,
    resolve_compute_dtype,
    run_preflight,
    train_qlora,
    validate_config,
)


# ── Configuration Tests ──────────────────────────────────────────────────────


def test_load_default_config():
    """Verify default config loads successfully with all required sections."""
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config["model"]["id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["dataset"]["train_path"] == "data/qa/finetune_train.jsonl"
    assert config["dataset"]["val_path"] == "data/qa/finetune_val.jsonl"
    assert config["dataset"]["max_seq_length"] == 512
    assert config["qlora"]["load_in_4bit"] is True
    assert config["qlora"]["bnb_4bit_quant_type"] == "nf4"
    assert config["lora"]["r"] == 16
    assert config["lora"]["lora_alpha"] == 32
    assert config["lora"]["target_modules"] == "all-linear"
    assert config["training"]["num_train_epochs"] == 3
    assert config["training"]["per_device_train_batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 4


def test_validate_config_missing_sections():
    """Verify ValueError is raised when required sections are missing."""
    with pytest.raises(ValueError, match="Missing required config section"):
        validate_config({"model": {"id": "test"}})

    with pytest.raises(ValueError, match="Config model.id must not be empty"):
        validate_config({
            "model": {"id": ""},
            "dataset": {"train_path": "a", "val_path": "b", "max_seq_length": 512},
            "qlora": {},
            "lora": {"r": 16},
            "training": {},
        })


# ── Dataset Loading & Formatting Tests ──────────────────────────────────────


def test_load_dataset_records(tmp_path: Path):
    """Verify dataset loader parses valid JSONL records and validates roles."""
    sample_data = [
        {
            "id": "test_01",
            "messages": [
                {"role": "system", "content": "Sys prompt"},
                {"role": "user", "content": "User q"},
                {"role": "assistant", "content": "Asst ans"},
            ],
        }
    ]
    p = tmp_path / "test.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in sample_data:
            f.write(json.dumps(r) + "\n")

    records = load_dataset_records(p)
    assert len(records) == 1
    assert records[0]["id"] == "test_01"


def test_load_dataset_records_invalid_roles(tmp_path: Path):
    """Verify error when messages do not follow system/user/assistant order."""
    invalid_data = [
        {
            "id": "bad_01",
            "messages": [
                {"role": "user", "content": "No system prompt"},
                {"role": "assistant", "content": "Ans"},
            ],
        }
    ]
    p = tmp_path / "bad.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in invalid_data:
            f.write(json.dumps(r) + "\n")

    with pytest.raises(ValueError, match="invalid roles"):
        load_dataset_records(p)


def test_format_chat_text_fallback():
    """Verify chat formatting renders special delimiters correctly."""
    messages = [
        {"role": "system", "content": "System text"},
        {"role": "user", "content": "User text"},
        {"role": "assistant", "content": "Asst text"},
    ]
    mock_tok = MagicMock(spec=[])  # No apply_chat_template
    formatted = format_chat_text(messages, mock_tok)
    assert "<|im_start|>system\nSystem text<|im_end|>\n" in formatted
    assert "<|im_start|>user\nUser text<|im_end|>\n" in formatted
    assert "<|im_start|>assistant\nAsst text<|im_end|>\n" in formatted


# ── Token Length Computation Tests ──────────────────────────────────────────


def test_compute_token_length_stats():
    """Verify token statistics calculation and truncation logic."""
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "formatted text"
    mock_tok.encode.side_effect = lambda t: [1] * 200  # 200 tokens

    records = [
        {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
        for _ in range(5)
    ]
    stats = compute_token_length_stats(records, mock_tok, max_seq_length=512)
    assert stats["count"] == 5
    assert stats["min"] == 200
    assert stats["max"] == 200
    assert stats["median"] == 200.0
    assert stats["truncated_at_max"] == 0


def test_compute_token_length_stats_with_truncation():
    """Verify truncation detection when sequence exceeds max_seq_length."""
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "long text"
    mock_tok.encode.return_value = [1] * 600  # 600 tokens > 512

    records = [
        {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
    ]
    stats = compute_token_length_stats(records, mock_tok, max_seq_length=512)
    assert stats["truncated_at_max"] == 1


# ── QLoRA and LoRA Configuration Tests ──────────────────────────────────────


def test_build_bnb_config():
    """Verify BitsAndBytesConfig sets 4-bit, NF4, double quant, and compute dtype."""
    config = {
        "qlora": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    }
    bnb_cfg = build_bnb_config(config, torch.float16)
    assert bnb_cfg.load_in_4bit is True
    assert bnb_cfg.bnb_4bit_quant_type == "nf4"
    assert bnb_cfg.bnb_4bit_use_double_quant is True
    assert bnb_cfg.bnb_4bit_compute_dtype == torch.float16


def test_build_lora_config():
    """Verify PEFT LoraConfig creates correct rank, alpha, and module targets."""
    config = {
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": "all-linear",
        }
    }
    lora_cfg = build_lora_config(config)
    assert lora_cfg.r == 16
    assert lora_cfg.lora_alpha == 32
    assert lora_cfg.lora_dropout == 0.05
    assert lora_cfg.bias == "none"
    assert lora_cfg.target_modules == "all-linear"


def test_resolve_compute_dtype():
    """Verify programmatic dtype selection based on GPU capability."""
    assert resolve_compute_dtype({"bf16_supported": True}, "auto") == torch.bfloat16
    assert resolve_compute_dtype({"bf16_supported": False}, "auto") == torch.float16
    assert resolve_compute_dtype({}, "float16") == torch.float16
    assert resolve_compute_dtype({}, "bfloat16") == torch.bfloat16


# ── Strict CPU Training Refusal Test ────────────────────────────────────────


def test_train_qlora_cpu_refusal():
    """Verify train_qlora strictly raises RuntimeError when CUDA is unavailable."""
    with patch("scripts.train_qlora.get_environment_info") as mock_env:
        mock_env.return_value = {"cuda_available": False}
        with pytest.raises(RuntimeError, match="prohibit running QLoRA training on local CPU"):
            train_qlora(DEFAULT_CONFIG_PATH)


# ── Preflight Dry-Run Test ──────────────────────────────────────────────────


def test_run_preflight_success():
    """Verify run_preflight executes cleanly and returns 0 exit code."""
    result = run_preflight(DEFAULT_CONFIG_PATH)
    assert result == 0
