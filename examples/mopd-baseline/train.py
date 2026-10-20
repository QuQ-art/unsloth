#!/usr/bin/env python3
"""Full-parameter MOPD with synchronous, single-resident-teacher offload."""
import argparse
import statistics
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

# Import Unsloth first: it patches some training paths at import time.
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GKDConfig, GKDTrainer
from opd_telemetry import OPDTelemetry
from opd_memory import MemoryAudit, model_inventory


def get_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--teacher-code", required=True)
    p.add_argument("--teacher-if", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-rows", type=int, default=0, help="0 uses the entire Open-MOPD training parquet")
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--lmbda", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    assert args.batch_size == 1 and args.grad_accum == 1, "Stage B requires batch=1, GA=1"
    assert torch.cuda.is_available(), "CUDA GPU not visible; check CUDA_VISIBLE_DEVICES and nvidia-smi"
    assert args.beta == 1.0, "This experiment requires reverse KL"
    assert args.steps > 0 and args.grad_accum > 0 and args.batch_size > 0
    assert args.max_prompt_length + args.max_new_tokens <= args.max_length
    assert args.lmbda == 1.0, "This experiment must use 100% on-policy sampling"
    assert args.max_new_tokens > 0 and args.max_length > args.max_new_tokens
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_monitor = None
    from opd_offload import ProcessMemoryMonitor
    cpu_monitor = ProcessMemoryMonitor(args.output_dir)
    cpu_monitor.start()

    t0 = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu, flush=True)
    print("Visible GPUs:", torch.cuda.device_count(), flush=True)
    assert torch.cuda.device_count() == 1, "Use CUDA_VISIBLE_DEVICES to select one GPU"

    metadata = {
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "<unset>"),
        "gpu": gpu,
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
        "torch_cuda_version": torch.version.cuda,
        "versions": {p: get_version(p) for p in (
            "unsloth", "unsloth_zoo", "trl", "transformers", "peft", "datasets", "torch", "accelerate")},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("VERSIONS:", metadata["versions"], flush=True)

    memory = MemoryAudit(args.output_dir)
    memory.capture("before_student_load")
    student, processor = FastLanguageModel.from_pretrained(
        model_name=args.student,
        max_seq_length=args.max_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=True,
        use_gradient_checkpointing="unsloth",
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not type(processor).__name__.endswith("Processor"):
        tokenizer = processor
    student.to(dtype=torch.bfloat16)
    memory.student = student
    memory.capture("after_student_load")

    torch.cuda.empty_cache()
    memory.capture("before_teacher_load")
    teacher, teacher_processor = FastLanguageModel.from_pretrained(
        model_name=args.teacher,
        max_seq_length=args.max_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    memory.teacher = teacher
    memory.capture("after_teacher_load")
    teachers = {0: teacher}
    teacher.to('cpu')
    for teacher_id, path in ((1, args.teacher_code), (2, args.teacher_if)):
        extra, extra_processor = FastLanguageModel.from_pretrained(
            model_name=path, max_seq_length=args.max_length,
            dtype=torch.bfloat16, load_in_4bit=False)
        extra.eval().requires_grad_(False)
        extra.to('cpu')
        extra_tokenizer = getattr(extra_processor, "tokenizer", extra_processor)
        assert tokenizer.get_vocab() == extra_tokenizer.get_vocab()
        assert extra.config.vocab_size == student.config.vocab_size
        teachers[teacher_id] = extra
        memory.teachers = teachers
        memory.capture(f"after_teacher_{teacher_id}_load")
    # Loading the frozen teacher resets this global Unsloth switch. Restore the
    # student mode before Trainer preparation; the teacher stays explicitly frozen.
    os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"
    assert all(p.requires_grad for p in student.parameters()), "Student was partially frozen"
    assert not getattr(student, "peft_config", None), "Full mode must not have PEFT adapters"
    assert not any("lora_" in n for n, _ in student.named_parameters())

    # GKD requires same token IDs at the same vocabulary positions.
    t_tokenizer = getattr(teacher_processor, "tokenizer", None)
    if t_tokenizer is None or not type(teacher_processor).__name__.endswith("Processor"):
        t_tokenizer = teacher_processor
    probe = "The quick brown fox jumps over 0123456789 lazy dogs."
    assert tokenizer(probe).input_ids == t_tokenizer(probe).input_ids, "Tokenizer IDs mismatch"
    assert student.config.vocab_size == teacher.config.vocab_size, "Vocabulary widths differ"

    assert tokenizer.get_vocab() == t_tokenizer.get_vocab(), "Full tokenizer vocabulary mismatch"
    raw = load_dataset("parquet", data_files=args.dataset, split="train")
    if args.data_rows:
        raw = raw.select(range(min(args.data_rows, len(raw))))

    def prepare_prompt(example):
        # Open-MOPD parquet contains prompt-only conversations. Never drop the
        # final user message, as the default GKD conversational collator would.
        text = tokenizer.apply_chat_template(example["prompt"], tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        result = {"input_ids": ids, "prompt": text}
        from opd_routing import resolve_teacher_id
        result["teacher_id"] = resolve_teacher_id(example)
        return result

    data = raw.map(prepare_prompt, remove_columns=raw.column_names)
    data = data.filter(lambda x: 0 < len(x["input_ids"]) <= args.max_prompt_length)
    print(f"Dataset kept {len(data)}/{len(raw)} rows", flush=True)
    assert len(data) > 0, "No prompts fit the configured budget"
    # Balanced deterministic validation subset, with two distinct examples
    # per teacher for the default 6-step run. Original parquet is untouched.
    needed = max(1, (args.steps + 2) // 3)
    buckets = {i: [] for i in teachers}
    for index, teacher_id in enumerate(data["teacher_id"]):
        if len(buckets[teacher_id]) < needed:
            buckets[teacher_id].append(index)
        if all(len(v) == needed for v in buckets.values()):
            break
    assert all(len(v) == needed for v in buckets.values())
    selected = [buckets[i][j] for j in range(needed) for i in teachers]
    (args.output_dir / "routing_dataset.json").write_text(json.dumps({
        "mapping": {0: "math", 1: "code", 2: "if"},
        "teacher_id_source": "existing teacher_id validated against domain, otherwise derived from domain",
        "filtered_dataset_indices": selected,
        "teacher_ids": [data[i]["teacher_id"] for i in selected],
        "full_filtered_rows": len(data), "purpose": "balanced route coverage validation",
    }, indent=2))
    data = data.select(selected)
    (args.output_dir / "dataset.json").write_text(json.dumps({
        "source": args.dataset, "input_rows": len(raw), "kept_rows": len(data),
        "fingerprint": data._fingerprint, "max_prompt_length": args.max_prompt_length,
    }, indent=2))

    def collate_prompts(examples):
        tokenizer.padding_side = "left"
        batch = tokenizer.pad({"input_ids": [x["input_ids"] for x in examples]},
                              padding=True, return_tensors="pt")
        result = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                "labels": torch.full_like(batch["input_ids"], -100),
                "prompts": batch["input_ids"], "prompt_attention_mask": batch["attention_mask"]}
        result["teacher_id"] = torch.tensor([x["teacher_id"] for x in examples])
        return result

    bf16 = torch.cuda.is_bf16_supported()
    config = GKDConfig(
        output_dir=str(args.output_dir / "trainer_output"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        logging_steps=1,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="constant",
        seed=args.seed,
        max_length=args.max_length,
        lmbda=args.lmbda,
        beta=args.beta,
        seq_kd=False,
        temperature=1.0,
        max_new_tokens=args.max_new_tokens,
        bf16=bf16,
        fp16=not bf16,
        remove_unused_columns=False,
    )

    # Non-invasive wrapper around TRL's documented on-policy generation path.
    # If a future Unsloth release bypasses this method, check before interpreting a zero.
    rollout = {"calls": 0, "sequences": 0, "generated_tokens_upper_bound": 0, "examples": []}

    def record_rollout(out, inputs):
        rollout["calls"] += 1
        batch = int(out[0].shape[0])
        rollout["sequences"] += batch
        prompt_len = int(inputs["prompts"].shape[1])
        generated_len = max(0, int(out[0].shape[1]) - prompt_len)
        # This includes padding and is a rough upper bound, not exact completion tokens.
        rollout["generated_tokens_upper_bound"] += batch * generated_len
        if len(rollout["examples"]) < 2:
            text = tokenizer.decode(out[0][0][prompt_len:], skip_special_tokens=True)
            rollout["examples"].append(text[:400])
        return out

    timer = OPDTelemetry(args.output_dir)
    timer.on_rollout = record_rollout
    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=config,
        train_dataset=data,
        processing_class=tokenizer,
        data_collator=collate_prompts,
        callbacks=[timer],
    )
    from opd_bf16_optimizer import BF16StochasticAdamW
    import hashlib
    import inspect
    # Trainer/Unsloth preparation may promote norm tensors. Enforce the
    # Open-MOPD parameter storage dtype after preparation, before optimizer.
    student.to(dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in student.parameters())
    trainer.optimizer = BF16StochasticAdamW(
        student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01, foreach=False)
    source = Path(inspect.getsourcefile(BF16StochasticAdamW))
    (args.output_dir / "precision_profile.json").write_text(json.dumps({
        "profile": "open_mopd", "optimizer": "BF16StochasticAdamW",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameter_dtype": "bfloat16", "moment_dtype": "float32",
        "update_compute_dtype": "float32", "writeback": "stochastic_bfloat16",
        "persistent_master_weights": False, "optimizer_offload": False,
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
        "lr": args.learning_rate, "max_grad_norm": config.max_grad_norm,
    }, indent=2))
    # Unsloth generation can return inference tensors. A trainable embedding
    # must save token indices for backward, which rejects inference tensors.
    # Clone only such outputs, outside inference mode, without changing IDs.
    generate_outputs = trainer.generate_on_policy_outputs
    def autograd_safe_generate(*call_args, **call_kwargs):
        outputs = generate_outputs(*call_args, **call_kwargs)
        with torch.inference_mode(False):
            return tuple(x.clone() if isinstance(x, torch.Tensor) and torch.is_inference(x)
                         else x for x in outputs)
    trainer.generate_on_policy_outputs = autograd_safe_generate

    # Rollout runs without gradients; checkpointing applies only to training.
    trainer.generation_config.use_cache = True
    trainer.generation_config.top_p = 1.0
    trainer.generation_config.top_k = 0
    assert type(trainer.args).__name__ == "GKDConfig", type(trainer.args).__name__
    assert trainer.args.lmbda == 1.0 and trainer.args.beta == args.beta
    (args.output_dir / "gkd_config.json").write_text(config.to_json_string())
    (args.output_dir / "generation_config.json").write_text(trainer.generation_config.to_json_string())
    print("GKD config verified:", type(trainer.args).__name__,
          "lmbda=", trainer.args.lmbda, "beta=", trainer.args.beta, flush=True)

    memory.trainer = trainer
    timer.memory_audit = memory
    timer.install(trainer)
    router = None
    teachers[0] = trainer.teacher_model
    for i in (1, 2):
        teachers[i] = trainer.accelerator.prepare_model(teachers[i], evaluation_mode=True,
                                                      device_placement=False)
    memory.teachers = teachers
    from opd_offload import OffloadedTeacherRouter
    teachers[0].to('cpu')
    router = OffloadedTeacherRouter(trainer, teachers, timer, args.output_dir, cpu_monitor)
    router.check_migration_logits(tokenizer)
    router.install()
    memory.capture("trainer_ready")

    # CPU copies of all trainable tensors for exact, full-tensor update checks.
    # Full mode costs about 5.7 GiB host RAM; copies are outside timed training.
    before = {n: p.detach().cpu().clone() for n, p in student.named_parameters() if p.requires_grad}
    print("Trainable tensors:", len(before), flush=True)
    assert before, "No trainable student parameters"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory.capture("before_train")
    if cpu_monitor is not None:
        cpu_monitor.begin_training()
    train_t0 = time.perf_counter()
    try:
        result = trainer.train()
    except Exception as error:
        (args.output_dir / "failure.json").write_text(json.dumps({
            "type": type(error).__name__, "message": str(error),
            "completed_steps": trainer.state.global_step,
            "training_mode": "full",
        }, indent=2))
        raise
    finally:
        timer.write_evidence()
        if router is not None:
            router.restore()
        timer.restore()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_t0
    total_seconds = time.perf_counter() - t0
    memory.capture("after_train")
    training_peak_allocated = torch.cuda.max_memory_allocated()
    training_peak_reserved = torch.cuda.max_memory_reserved()
    if cpu_monitor is not None:
        cpu_monitor.stop()

    changed_names = [n for n, p in student.named_parameters()
                     if p.requires_grad and not torch.equal(p.detach().cpu(), before[n])]
    changed = len(changed_names)
    parameter_report = model_inventory(student)
    parameter_report["changed_parameter_names"] = changed_names
    parameter_report["unchanged_parameter_names"] = [n for n in before if n not in changed_names]
    (args.output_dir / "parameter_updates.json").write_text(json.dumps(parameter_report, indent=2))
    full_ok = (
        parameter_report["trainable_numel"] == parameter_report["total_numel"]
        and not parameter_report["has_peft_config"] and not parameter_report["quantized"]
        and all(row.get("optimizer_matches_trainable_student", True) and
                not row.get("optimizer_contains_teacher", False) for row in memory.rows))
    history = trainer.state.log_history
    losses = [float(x["loss"]) for x in history if "loss" in x]
    gradients = [float(x["grad_norm"]) for x in history if "grad_norm" in x]
    loss_ok = len(losses) == args.steps and all(math.isfinite(x) for x in losses)
    grad_ok = len(gradients) == args.steps and all(math.isfinite(x) and x > 0 for x in gradients)
    teacher_grad_free = all(p.grad is None and not p.requires_grad for t in teachers.values() for p in t.parameters())
    routing_ok = router.validate() if router is not None else True
    precision_ok = True
    opt = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    states = list(opt.state.values())
    precision_ok = (
        type(opt).__name__ == "BF16StochasticAdamW"
        and all(p.dtype == torch.bfloat16 for p in student.parameters())
        and len(states) == len(before)
        and all(set(s) == {"step", "exp_avg", "exp_avg_sq"}
                and s["step"] == args.steps
                and all(s[k].dtype == torch.float32 and torch.isfinite(s[k]).all().item()
                        for k in ("exp_avg", "exp_avg_sq")) for s in states))
    (args.output_dir / "trainer_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str))
    (args.output_dir / "rollout_samples.json").write_text(json.dumps(rollout, ensure_ascii=False, indent=2))

    chunked = timer.evidence()
    timing_ok = (len(timer.rows) == args.steps and all(
        r["optimizer_calls"] == 1 and r["rollout_tokens"] > 0 and r["rollout_seconds"] > 0
        and r["training_seconds"] > 0 and r["optimizer_seconds"] > 0 and r["other_seconds"] >= 0
        for r in timer.rows))
    metrics = {
        "success": bool(math.isfinite(result.training_loss) and loss_ok and grad_ok and teacher_grad_free
                        and chunked["verified"] and timing_ok and full_ok and precision_ok and routing_ok
                        and trainer.state.global_step == args.steps and changed > 0 and rollout["calls"] > 0),
        "training_mode": "full",
        "precision_profile": "open_mopd",
        "routing_checks_passed": routing_ok if router is not None else None,
        "precision_checks_passed": precision_ok,
        "full_parameter_checks_passed": full_ok,
        "student_parameter_inventory": parameter_report,
        "chunked_jsd": chunked,
        "stage_timing_checks_passed": timing_ok,
        "generated_tokens": sum(r["rollout_tokens"] for r in timer.rows),
        "generated_tokens_per_second": (sum(r["rollout_tokens"] for r in timer.rows) /
                                         sum(r["rollout_seconds"] for r in timer.rows)),
        "stage_seconds": {k: sum(r[k] for r in timer.rows) for k in
                          ("rollout_seconds", "training_seconds", "optimizer_seconds", "other_seconds")},
        "loss_checks_passed": loss_ok,
        "gradient_checks_passed": grad_ok,
        "teacher_gradient_free": teacher_grad_free,
        "grad_norm_min": min(gradients) if gradients else None,
        "grad_norm_max": max(gradients) if gradients else None,
        "training_loss": float(result.training_loss),
        "train_seconds": round(train_seconds, 3),
        "total_seconds_including_setup": round(total_seconds, 3),
        "seconds_per_optimizer_step": round(train_seconds / args.steps, 4),
        "median_step_wall_seconds_excl_first_5": (round(statistics.median(
            [x["step_wall_seconds"] for x in timer.rows[5:]]), 5)
            if len(timer.rows) > 5 else None),
        "optimizer_steps": args.steps,
        "sample_count_after_filter": len(data),
        "rollout": rollout,
        "trainable_tensors": len(before),
        "changed_trainable_tensors": changed,
        "peak_torch_allocated_gib": round(training_peak_allocated / 2**30, 3),
        "peak_torch_reserved_gib": round(training_peak_reserved / 2**30, 3),
        "train_metrics": result.metrics,
        "end_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    print("\nEXPERIMENT_SUMMARY", json.dumps(metrics, ensure_ascii=False, indent=2, default=str), flush=True)
    assert metrics["success"], "Sanity check failed; inspect metrics.json, trainer_history.json and train.log"


if __name__ == "__main__":
    main()
