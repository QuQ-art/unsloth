"""Synchronized single-GPU OPD stage measurements and chunked-JSD evidence."""
import csv
import functools
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


def completion_lengths(tokens, eos_token_id, pad_token_id):
    """Count newly generated tokens through the first EOS (inclusive), never pads."""
    eos_ids = [] if eos_token_id is None else (
        [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
    eos = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in eos_ids:
        eos |= tokens == token_id
    before_or_first_eos = (eos.cumsum(-1) - eos.long()) == 0
    valid = before_or_first_eos
    if pad_token_id is not None and pad_token_id not in eos_ids:
        valid = valid & (tokens != pad_token_id)
    return valid.sum(-1).tolist()


class OPDTelemetry(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.rows = []
        self.current = None
        self.trainer = None
        self.restores = []
        self.chunk_shapes = []
        self.counts = dict(compute_loss_calls=0, chunked_jsd_calls=0, chunked_jsd_backward_calls=0,
                           dense_loss_calls=0)

    def sync_time(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def add(self, key, value):
        if self.current is not None:
            self.current[key] += value

    def replace(self, obj, name, value):
        original = getattr(obj, name)
        self.restores.append(lambda: setattr(obj, name, original))
        setattr(obj, name, value)

    def install(self, trainer):
        self.trainer = trainer
        compute = trainer.compute_loss
        namespace = compute.__func__.__globals__
        chunked = namespace.get('distillation_chunked_jsd')
        sources = {}
        for name, fn in [('compute_loss', compute),
                         ('chunked_dispatch', namespace.get('_unsloth_gkd_chunked_loss')),
                         ('distillation_chunked_jsd', chunked)]:
            if fn is None:
                sources[name] = None
                continue
            file = Path(inspect.getsourcefile(fn)).resolve()
            sources[name] = {'file': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                             'line': inspect.getsourcelines(fn)[1]}
            snapshot = self.output_dir / ('source_' + file.name)
            snapshot.write_bytes(file.read_bytes())
        self.sources = sources
        original_generate = trainer.generate_on_policy_outputs

        @functools.wraps(original_generate)
        def generate(model, inputs, generation_config, pad_token_id=None):
            start = self.sync_time()
            out = original_generate(model, inputs, generation_config, pad_token_id)
            elapsed = self.sync_time() - start
            self.add('rollout_seconds', elapsed)
            self.add('rollout_calls', 1)
            lengths = completion_lengths(out[0][:, inputs['prompts'].shape[1]:],
                                         generation_config.eos_token_id, pad_token_id)
            self.add('rollout_tokens', sum(lengths))
            self.add('rollout_sequences', len(lengths))
            self.current['completion_lengths'].extend(lengths)
            if hasattr(self, "on_rollout"):
                self.on_rollout(out, inputs)
            return out
        self.replace(trainer, 'generate_on_policy_outputs', generate)

        @functools.wraps(compute)
        def compute_loss(*args, **kwargs):
            start = self.sync_time()
            out = compute(*args, **kwargs)
            self.add('distillation_forward_seconds', self.sync_time() - start)
            self.add('compute_loss_calls', 1)
            self.counts['compute_loss_calls'] += 1
            return out
        self.replace(trainer, 'compute_loss', compute_loss)
        backward = trainer.accelerator.backward

        @functools.wraps(backward)
        def timed_backward(*args, **kwargs):
            start = self.sync_time()
            out = backward(*args, **kwargs)
            self.add('student_backward_seconds', self.sync_time() - start)
            return out
        self.replace(trainer.accelerator, 'backward', timed_backward)

        # Patch the actual global resolved by the generated compute_loss helper,
        # not merely the similarly named export in unsloth_zoo.
        if chunked is not None:
            @functools.wraps(chunked)
            def observed_chunked(*args, **kwargs):
                out = chunked(*args, **kwargs)
                self.add('chunked_jsd_calls', 1)
                self.counts['chunked_jsd_calls'] += 1
                if len(self.chunk_shapes) < 8:
                    self.chunk_shapes.append({
                        'student_hidden_shape': list(args[0].shape),
                        'teacher_hidden_shape': list(args[1].shape),
                        'student_head_shape': list(args[2].shape),
                        'chunk_size': kwargs.get('chunk_size'),
                        'loss_grad_fn': type(out[0].grad_fn).__name__,
                    })
                return out
            namespace['distillation_chunked_jsd'] = observed_chunked
            self.restores.append(lambda: namespace.__setitem__('distillation_chunked_jsd', chunked))
            autograd_class = chunked.__globals__.get('_UnslothDistillationJSD')
            if autograd_class is not None:
                original_backward = autograd_class.backward

                @functools.wraps(original_backward)
                def jsd_backward(*args, **kwargs):
                    out = original_backward(*args, **kwargs)
                    self.add('chunked_jsd_backward_calls', 1)
                    self.counts['chunked_jsd_backward_calls'] += 1
                    return out
                autograd_class.backward = staticmethod(jsd_backward)
                self.restores.append(lambda: setattr(autograd_class, 'backward', staticmethod(original_backward)))

        # Instance wrapper leaves the class-level source guard unchanged.
        dense = trainer.generalized_jsd_loss
        @functools.wraps(dense)
        def observed_dense(*args, **kwargs):
            self.add('dense_loss_calls', 1)
            self.counts['dense_loss_calls'] += 1
            return dense(*args, **kwargs)
        self.replace(trainer, 'generalized_jsd_loss', observed_dense)
        self.write_evidence()

    def on_step_begin(self, args, state, control, **kwargs):
        self.current = dict(step=int(state.global_step) + 1, rollout_tokens=0, rollout_sequences=0,
                            rollout_calls=0, rollout_seconds=0.0, distillation_forward_seconds=0.0,
                            student_backward_seconds=0.0, optimizer_seconds=0.0, optimizer_calls=0,
                            compute_loss_calls=0, chunked_jsd_calls=0, chunked_jsd_backward_calls=0,
                            dense_loss_calls=0, completion_lengths=[])
        self.step_start = self.sync_time()
        if getattr(self, 'teacher_offload', False):
            self.current.update(teacher_h2d_time=0.0, teacher_d2h_time=0.0,
                                teacher_switch_time=0.0, teacher_switch_count=0,
                                teacher_cache_hit=False)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if hasattr(self, "memory_audit") and state.global_step < 2:
            self.memory_audit.capture(f"step_{state.global_step + 1}_before_optimizer")
        self.optimizer_start = self.sync_time()

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.add('optimizer_seconds', self.sync_time() - self.optimizer_start)
        self.add('optimizer_calls', 1)
        if hasattr(self, 'memory_audit') and state.global_step < 2:
            self.memory_audit.capture(f'step_{state.global_step + 1}_after_optimizer')

    def on_step_end(self, args, state, control, **kwargs):
        row = self.current
        row['step_wall_seconds'] = self.sync_time() - self.step_start
        row['training_seconds'] = row['distillation_forward_seconds'] + row['student_backward_seconds']
        if getattr(self, 'teacher_offload', False):
            row.update(rollout_time=row['rollout_seconds'], distill_time=row['training_seconds'],
                       optimizer_time=row['optimizer_seconds'],
                       peak_gpu_memory=torch.cuda.max_memory_allocated(),
                       peak_cpu_memory=self.cpu_monitor.training_peak_rss)
        row['other_seconds'] = row['step_wall_seconds'] - row['rollout_seconds'] - row['training_seconds'] - row['optimizer_seconds']
        if getattr(self, 'teacher_offload', False):
            row['other_seconds'] -= row['teacher_switch_time']
        row['generated_tokens_per_second'] = row['rollout_tokens'] / row['rollout_seconds'] if row['rollout_seconds'] else None
        row['chunked_jsd_verified'] = (row['compute_loss_calls'] > 0 and
            row['compute_loss_calls'] == row['chunked_jsd_calls'] == row['chunked_jsd_backward_calls'] and
            row['dense_loss_calls'] == 0)
        self.rows.append(row)
        with (self.output_dir / 'step_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        with (self.output_dir / 'step_times.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows({**r, 'completion_lengths': json.dumps(r['completion_lengths'])} for r in self.rows)
        print('OPD_STEP_METRICS', json.dumps(row), flush=True)
        self.write_evidence()
        self.current = None

    def evidence(self):
        builtin = int(getattr(self.trainer, '_unsloth_gkd_chunked_calls', 0))
        fallbacks = dict(getattr(self.trainer, '_unsloth_gkd_chunked_fallbacks', {}))
        return {**self.counts, 'trainer_chunked_calls': builtin, 'fallbacks': fallbacks,
                'verified': (self.counts['compute_loss_calls'] > 0 and
                    self.counts['compute_loss_calls'] == self.counts['chunked_jsd_calls'] ==
                    self.counts['chunked_jsd_backward_calls'] == builtin and
                    self.counts['dense_loss_calls'] == 0 and not fallbacks),
                'sources': self.sources, 'observed_chunk_shapes': self.chunk_shapes,
                'environment': {k: os.getenv(k) for k in ('UNSLOTH_GKD_CHUNKED', 'UNSLOTH_GKD_CHUNK_SIZE')}}

    def write_evidence(self):
        (self.output_dir / 'chunked_jsd_evidence.json').write_text(json.dumps(self.evidence(), indent=2))

    def restore(self):
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()
