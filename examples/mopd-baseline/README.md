# MOPD baseline: full fine-tuning with teacher offload

Standalone extraction of `exp3_routing_stage_b`. A BF16 full-parameter student
samples its own responses and learns from one of three frozen teachers using
Unsloth's chunked reverse-KL loss (`lmbda=beta=1`). This directory contains all
example code; it does not depend on the original experiment workspace.

The student, gradients and AdamW states stay on the GPU. Inactive teachers stay
on the CPU. After rollout, the router synchronously unloads the previous teacher
and loads the selected teacher; consecutive selections reuse the resident
teacher. At most one teacher resides on the GPU. There is no optimizer offload,
prefetch or persistent CPU mirror of the active teacher.

## Environment

Use Linux with a BF16-capable CUDA GPU and enough host RAM. The original 3B
configuration used approximately 45 GiB of allocated GPU memory and 28 GiB of
process RAM; leave headroom for loading and compilation. Exact requirements
depend on the model, sequence lengths and software versions.

The validation environment uses Python 3.11, PyTorch 2.14.1+cu130,
Transformers 5.15.1, TRL 0.25.1 and Unsloth Zoo commit
`6d194fb81a8e138878227d819c19b358a43e0a2c`.
`requirements-tested.txt` records the reference environment, including transitive
dependencies. It is not a portable lockfile; use a separate environment and the
appropriate CUDA/PyTorch installation for your machine. With those dependencies
available, install this fork from its repository root:

```bash
python -m pip install --no-deps -e .
```

The example uses Unsloth's internal GKD hidden-state and chunked-loss hooks.
Other library versions need revalidation. The runtime explicitly checks that the
chunked forward/backward paths execute without dense fallback.

## Run

For the configured local experiment, use the repository-root launcher:

```bash
bash r2.sh       # GPU 0
bash r2.sh 1     # GPU 1
```

`r2.sh` holds the local Student/Teacher/Dataset paths and fixed experiment
parameters. It defaults to the local `unsloth_opd` Python environment, imports
Unsloth from this checkout and enables offline mode. Edit its configuration or
override `MODEL_ROOT`, `STUDENT`, `TEACHER_MATH`, `TEACHER_CODE`, `TEACHER_IF`,
`DATASET` and `PYTHON_BIN` through environment variables. Set `HF_HUB_OFFLINE=0`
and `HF_DATASETS_OFFLINE=0` when online access is needed. It delegates GPU binding,
Python execution, output management and performance logging to `run.sh`.

Prepare a student and Math/Code/IF teachers with identical token vocabularies
and compatible causal-LM architectures. The original models are SmolLM3-3B
MixSFT, Math, Code and IF. Supply your own paths or model identifiers; weights
and datasets are not included.

From the repository root:

```bash
MODEL_ROOT=/path/to/OPD
bash examples/mopd-baseline/run.sh 0 \
  --student "$MODEL_ROOT/MixSFT" \
  --teacher "$MODEL_ROOT/Math" \
  --teacher-code "$MODEL_ROOT/Code" \
  --teacher-if "$MODEL_ROOT/IF" \
  --dataset "$MODEL_ROOT/data/rl_prompt_mix/train.parquet"
```

`PYTHON_BIN` overrides the active environment's `python`. `RESULTS_DIR` overrides
the output parent. The launcher accepts a physical GPU index, selects exactly
one GPU and creates a unique run directory. It can be called from any working
directory. For local-only assets, set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
Cache locations can be overridden with `HF_DATASETS_CACHE`,
`UNSLOTH_COMPILE_LOCATION`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`.
To invoke Python directly, select one GPU and pass `--output-dir` as well.

Defaults match Stage B: 6 optimizer steps, batch size 1, gradient accumulation 1,
prompt limit 512, generation limit 1024, total length 1536, seed 42, learning rate
1e-6, constant schedule, weight decay 0.01 and gradient clipping 1. Student
weights and gradients use BF16; AdamW moments and update arithmetic use FP32,
with stochastic BF16 writeback and no persistent master weights.

The parquet must contain:

- `prompt`: a chat-message list, e.g. `[{"role": "user", "content": "Compute 2+3."}]`.
- `domain`: `math`, `code` or `if`, mapped to teacher IDs 0, 1 and 2.
- Optional `teacher_id`: the corresponding integer or domain name; mismatches fail.

Prompts are rendered with the student's chat template. Rows exceeding the prompt
limit are filtered out. As in the original Stage B experiment, this is a
**coverage-validation baseline**: it selects `ceil(steps / 3)` distinct examples
per domain and uses the Trainer's seeded sampler. It does not train on the whole
parquet. Use the default 6 steps to cover all three teachers and exercise cache
reuse; arbitrary step counts/seeds can fail the required route-coverage check.
It does not save trained checkpoints (`save_strategy="no"`).

## Validation and outputs

A successful run exits with code 0 and writes `metrics.json` with
`"success": true`. Assertions must remain enabled (do not use `python -O`).
Checks include finite losses and nonzero finite gradients, actual student
parameter changes, optimizer ownership and FP32 states, frozen teachers, all
three routes, selected-teacher/head correctness, and chunked forward/backward
counts with no fallback. Before training, each teacher's logits must be bitwise
identical across a GPU → CPU → GPU round trip on fixed input.

| File | Evidence |
| --- | --- |
| `metrics.json`, `trainer_history.json` | Overall result, losses and gradients |
| `routing_evidence.json`, `migration_logits.json` | Route coverage, teacher execution and migration equality |
| `offload_metrics.json`, `teacher_transfers.json` | Residency, transfer times, switches and cache hits |
| `step_times.csv`, `step_metrics.jsonl` | Rollout, distillation, optimizer and transfer timing |
| `memory_audit.json`, `parameter_updates.json` | Actual storage/dtypes, optimizer ownership and full-tensor update comparisons |
| `chunked_jsd_evidence.json` | Actual chunked-loss calls, backward calls and fallback checks |
| `environment.json`, `*_config.json`, source snapshots | Runtime environment and reproduction information |
| `gpu.csv`, `cpu_memory_samples.json`, `exit_status.txt` | Device sampling, process RSS and launcher exit status |

Transfer timing includes CUDA synchronization. Switch counts exclude the initial
cold load. GPU peaks are PyTorch allocated memory; process RSS includes teachers,
temporary allocations and a full CPU copy of the student's initial parameters
used for update validation. Compiler/setup time is outside training time.
Generated outputs and caches are ignored by Git.

## Implementation

- `train.py`: Stage B configuration, prompt preparation, full fine-tuning and checks.
- `opd_routing.py`: domain routing and selected-teacher/head instrumentation.
- `opd_offload.py`: synchronous migration, residency assertions and process RSS.
- `opd_bf16_optimizer.py`: Open-MOPD BF16 stochastic AdamW implementation,
  retaining its original Apache-2.0 copyright notice.
- `opd_memory.py`, `opd_telemetry.py`: tensor storage audits and execution evidence.
- `run.sh`: launcher, source snapshots and GPU sampling.

## Local verification (2026-10-20)

Validated with this fork at `e1d63546017782cc107e30ab996b7026b97c1728`,
loaded via `PYTHONPATH` pointing to the checkout, and the reference environment
above. GPU: RTX PRO 5000 72GB; original MixSFT/Math/Code/IF models and parquet;
default 6-step / 512-prompt / 1024-response configuration.

- Exit code 0; `metrics.json` reports `success: true`.
- All three routes covered (0, 0, 2, 1, 1, 2); 3 switches and 2 cache hits.
- All three migration-logit comparisons are bitwise equal.
- Chunked JSD: 6 forward and 6 backward calls, zero dense fallback.
- All 326 student tensors participated in six optimizer updates; 325 had
  changed final BF16 storage values. Teachers remained frozen and gradient-free.
- Mean loss 0.1429783801; GPU allocated peak 45.061 GiB;
  training time 148.031 s.

Local evidence is under `output/20261020_150830_gpu0_856327/` (ignored by Git).
These measurements are a smoke/coverage validation, not a convergence result.
