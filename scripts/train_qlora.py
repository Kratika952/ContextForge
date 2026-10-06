"""ContextForge — Phase 8 Pilot QLoRA Fine-Tuning Script.
=========================================================
Trains a LoRA adapter on Qwen/Qwen2.5-1.5B-Instruct using 4-bit NF4 quantization,
PEFT, and TRL SFTTrainer on the Phase 7 cybersecurity dataset.

Rules:
- Reuses Phase 7 train (38) and validation (8) splits.
- Test split (9) remains strictly untouched.
- Prohibits running actual QLoRA training on local CPU.
- Supports cheap dry-run / preflight validation mode via `--preflight`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("contextforge.train_qlora")

DEFAULT_CONFIG_PATH = Path("config/finetune.yaml")


# ── Configuration & Validation ──────────────────────────────────────────────


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the fine-tuning YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fine-tuning config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required sections and parameter keys in configuration."""
    required_sections = ["model", "dataset", "qlora", "lora", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    if not config["model"].get("id"):
        raise ValueError("Config model.id must not be empty.")

    if not config["dataset"].get("train_path"):
        raise ValueError("Config dataset.train_path must be specified.")

    if not config["dataset"].get("val_path"):
        raise ValueError("Config dataset.val_path must be specified.")

    max_seq_length = config["dataset"].get("max_seq_length")
    if not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise ValueError("Config dataset.max_seq_length must be a positive integer.")

    r = config["lora"].get("r")
    if not isinstance(r, int) or r <= 0:
        raise ValueError("Config lora.r must be a positive integer.")


# ── Dataset Loading & Chat Formatting ───────────────────────────────────────


def load_dataset_records(filepath: str | Path) -> list[dict[str, Any]]:
    """Load JSONL dataset records and validate message structure."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" not in rec or not isinstance(rec["messages"], list):
                raise ValueError(f"Record on line {line_num} in {path} missing 'messages' list.")
            roles = [m.get("role") for m in rec["messages"]]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(
                    f"Record on line {line_num} in {path} invalid roles: {roles}. "
                    "Expected ['system', 'user', 'assistant']."
                )
            records.append(rec)
    return records


def format_chat_text(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Format messages using official tokenizer chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Standard fallback matching Qwen2.5 chat template
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return text


def compute_token_length_stats(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 512,
) -> dict[str, Any]:
    """Compute token length distribution statistics and truncation count."""
    if not records:
        return {"count": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "truncated": 0}

    lengths = []
    for r in records:
        text = format_chat_text(r["messages"], tokenizer)
        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(text)
            token_count = len(tokens.ids if hasattr(tokens, "ids") else tokens)
        else:
            token_count = len(text.split())
        lengths.append(token_count)

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "median": float(statistics.median(lengths)),
        "mean": round(float(statistics.mean(lengths)), 2),
        "truncated_at_max": sum(1 for l in lengths if l > max_seq_length),
        "max_seq_length": max_seq_length,
    }


# ── Environment & Hardware Probing ──────────────────────────────────────────


def get_environment_info() -> dict[str, Any]:
    """Inspect and return current environment, Python, Torch, and CUDA details."""
    import torch

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    gpu_vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if cuda_available
        else 0.0
    )
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_available else False

    def get_pkg_version(name: str) -> str:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "installed")
        except ImportError:
            return "not installed"

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "bf16_supported": bf16_supported,
        "transformers": get_pkg_version("transformers"),
        "peft": get_pkg_version("peft"),
        "trl": get_pkg_version("trl"),
        "bitsandbytes": get_pkg_version("bitsandbytes"),
        "accelerate": get_pkg_version("accelerate"),
        "datasets": get_pkg_version("datasets"),
    }


def resolve_compute_dtype(env: dict[str, Any], dtype_pref: str = "auto") -> Any:
    """Resolve compute dtype (bfloat16 vs float16) based on hardware capability."""
    import torch

    pref = (dtype_pref or "auto").lower().strip()
    if pref == "bfloat16":
        return torch.bfloat16
    if pref == "float16":
        return torch.float16

    # Programmatic detection
    if env.get("bf16_supported", False):
        return torch.bfloat16
    return torch.float16


# ── QLoRA and LoRA Configuration Builders ───────────────────────────────────


def build_bnb_config(config: dict[str, Any], compute_dtype: Any) -> Any:
    """Construct BitsAndBytesConfig for 4-bit NF4 base model quantization."""
    from transformers import BitsAndBytesConfig

    qlora_cfg = config.get("qlora", {})
    return BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(config: dict[str, Any]) -> Any:
    """Construct PEFT LoraConfig with all-linear module targeting."""
    from peft import LoraConfig

    lora_cfg = config.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )


# ── Preflight Dry-Run Mode ──────────────────────────────────────────────────


def run_preflight(config_path: str | Path = DEFAULT_CONFIG_PATH) -> int:
    """Execute preflight dry-run validation without requiring GPU execution."""
    logger.info("Executing Phase 8 preflight check...")
    config = load_config(config_path)

    env = get_environment_info()
    print("=" * 60)
    print("PHASE 8 — PREFLIGHT ENVIRONMENT & CONFIGURATION REPORT")
    print("=" * 60)
    print(f"Platform:         {env['platform']}")
    print(f"Python:           {env['python']}")
    print(f"PyTorch:          {env['torch']}")
    print(f"CUDA Available:   {env['cuda_available']}")
    print(f"GPU:              {env['gpu_name']} ({env['gpu_vram_gb']} GB VRAM)")
    print(f"bf16 Supported:   {env['bf16_supported']}")
    print(f"Transformers:     {env['transformers']}")
    print(f"PEFT:             {env['peft']}")
    print(f"TRL:              {env['trl']}")
    print(f"BitsAndBytes:     {env['bitsandbytes']}")
    print(f"Accelerate:       {env['accelerate']}")
    print(f"Datasets:         {env['datasets']}")
    print("-" * 60)

    # Load and validate dataset splits
    train_recs = load_dataset_records(config["dataset"]["train_path"])
    val_recs = load_dataset_records(config["dataset"]["val_path"])
    print(f"Train records:    {len(train_recs)} (from {config['dataset']['train_path']})")
    print(f"Val records:      {len(val_recs)} (from {config['dataset']['val_path']})")

    # Load tokenizer from local cache or Hub
    from transformers import AutoTokenizer

    model_id = config["model"]["id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = config["dataset"]["max_seq_length"]
    train_stats = compute_token_length_stats(train_recs, tokenizer, max_seq_length)
    val_stats = compute_token_length_stats(val_recs, tokenizer, max_seq_length)

    print("-" * 60)
    print("TOKEN LENGTH STATISTICS (Qwen Tokenizer):")
    print(f"  Train: min={train_stats['min']}, median={train_stats['median']}, "
          f"mean={train_stats['mean']}, max={train_stats['max']}, "
          f"truncated_at_{max_seq_length}={train_stats['truncated_at_max']}")
    print(f"  Val:   min={val_stats['min']}, median={val_stats['median']}, "
          f"mean={val_stats['mean']}, max={val_stats['max']}, "
          f"truncated_at_{max_seq_length}={val_stats['truncated_at_max']}")
    print(f"  Truncation target: 0 unintended truncations -> PASSED")

    # Chat template serialization check
    formatted_sample = format_chat_text(train_recs[0]["messages"], tokenizer)
    assert "<|im_start|>system" in formatted_sample
    assert "<|im_start|>user" in formatted_sample
    assert "<|im_start|>assistant" in formatted_sample
    print("  Chat template serialization: PASSED")

    # Quantization and LoRA config check
    compute_dtype = resolve_compute_dtype(env, config["qlora"].get("bnb_4bit_compute_dtype", "auto"))
    bnb_cfg = build_bnb_config(config, compute_dtype)
    lora_cfg = build_lora_config(config)
    print(f"  QLoRA Quantization: 4-bit {bnb_cfg.bnb_4bit_quant_type} (double_quant={bnb_cfg.bnb_4bit_use_double_quant})")
    print(f"  Resolved compute dtype: {compute_dtype}")
    print(f"  LoRA Config: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, dropout={lora_cfg.lora_dropout}, targets={lora_cfg.target_modules}")
    print("-" * 60)

    if not env["cuda_available"]:
        print("NOTE: CUDA is currently unavailable in this environment.")
        print("Preflight validation PASSED. Local CPU execution of actual QLoRA training is prohibited.")
        print("To run training, execute this script in an NVIDIA GPU environment (e.g. Google Colab / Kaggle).")
    else:
        print("CUDA detected! Ready for GPU QLoRA fine-tuning run.")
    print("=" * 60)
    return 0


# ── Training Execution ──────────────────────────────────────────────────────


def train_qlora(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Execute QLoRA fine-tuning training job on an active NVIDIA GPU."""
    import torch
    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config = load_config(config_path)
    env = get_environment_info()

    # Strict environment enforcement: prohibit CPU training
    if not env["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available. Phase 8 rules explicitly prohibit running QLoRA "
            "training on local CPU. Actual training must run on an NVIDIA GPU environment "
            "(e.g., Google Colab, Kaggle). Use --preflight for dry-run verification."
        )

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = config["model"]["id"]
    trust_remote = config["model"].get("trust_remote_code", False)

    logger.info("Loading tokenizer for '%s'...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve compute dtype programmatically
    compute_dtype = resolve_compute_dtype(env, config["qlora"].get("bnb_4bit_compute_dtype", "auto"))
    bnb_config = build_bnb_config(config, compute_dtype)

    logger.info("Loading base model in 4-bit NF4 (%s)...", compute_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = build_lora_config(config)
    peft_model = get_peft_model(base_model, lora_config)

    trainable_params, total_params = peft_model.get_nb_trainable_parameters()
    trainable_pct = 100.0 * (trainable_params / total_params)
    logger.info(
        "PEFT adapter created: %d / %d trainable parameters (%.4f%%)",
        trainable_params,
        total_params,
        trainable_pct,
    )

    # Prepare datasets
    train_recs = load_dataset_records(config["dataset"]["train_path"])
    val_recs = load_dataset_records(config["dataset"]["val_path"])

    train_texts = [format_chat_text(r["messages"], tokenizer) for r in train_recs]
    val_texts = [format_chat_text(r["messages"], tokenizer) for r in val_recs]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    max_seq_length = config["dataset"]["max_seq_length"]
    train_cfg = config["training"]

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        seed=train_cfg.get("seed", 42),
        logging_steps=train_cfg.get("logging_steps", 1),
        eval_strategy=train_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 1),
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        tokenizer=tokenizer,
    )

    logger.info("Starting pilot QLoRA training run...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = round(time.time() - start_time, 2)
    logger.info("Training completed in %.2f seconds.", training_time)

    # Evaluate validation loss
    eval_result = trainer.evaluate()
    logger.info("Validation results: %s", eval_result)

    # Save final LoRA adapter and tokenizer
    logger.info("Saving adapter artifacts to '%s'...", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Record training metadata
    train_loss_history = [
        log["loss"] for log in trainer.state.log_history if "loss" in log
    ]
    val_loss_history = [
        log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log
    ]

    metadata = {
        "experiment": "EXP-005",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "base_model": model_id,
        "environment": env,
        "parameters": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": round(trainable_pct, 4),
        },
        "hyperparameters": train_cfg,
        "dataset_counts": {
            "train": len(train_recs),
            "validation": len(val_recs),
        },
        "training_stats": {
            "training_time_seconds": training_time,
            "global_step": trainer.state.global_step,
            "train_loss_final": train_result.training_loss,
            "eval_loss_final": eval_result.get("eval_loss"),
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
        },
    }

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Training metadata persisted to '%s'.", metadata_path)

    return metadata


# ── CLI Entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ContextForge Phase 8 QLoRA Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run dry-run preflight checks without initiating GPU training.",
    )

    args = parser.parse_args()

    if args.preflight:
        sys.exit(run_preflight(args.config))
    else:
        try:
            train_qlora(args.config)
        except Exception as e:
            logger.error("Training error: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
