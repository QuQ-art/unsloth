"""Batch-one teacher dispatch and selected-teacher/head validation."""
import inspect
import json
from pathlib import Path

import torch


def resolve_teacher_id(example):
    mapping = {'math': 0, 'code': 1, 'if': 2}
    expected = mapping[example['domain']]
    actual = example.get('teacher_id', expected)
    if isinstance(actual, str) and actual in mapping:
        actual = mapping[actual]
    if actual not in (0, 1, 2) or int(actual) != expected:
        raise ValueError(f'Invalid teacher_id/domain: {actual}/{example["domain"]}')
    return int(actual)


class SerialTeacherRouter:
    def __init__(self, trainer, teachers, timer, output_dir):
        self.trainer, self.teachers, self.timer = trainer, teachers, timer
        self.path = Path(output_dir) / 'routing_evidence.json'
        self.rows, self.handles = [], []
        self.active = None
        self.calls = {i: 0 for i in teachers}
        self.versions = {i: [p._version for p in t.parameters()] for i, t in teachers.items()}

    def select(self, teacher_id):
        raise NotImplementedError("Use OffloadedTeacherRouter")

    def install(self):
        # GKDTrainer installs this only for its initial teacher. Dynamically
        # routed teachers need the same Unsloth hidden-state adapter as Math.
        from unsloth.models.rl import _install_grpo_hidden_states_forward_wrapper
        for teacher in self.teachers.values():
            _install_grpo_hidden_states_forward_wrapper(self.trainer.accelerator.unwrap_model(teacher))
        self.compute = self.trainer.compute_loss
        original = inspect.unwrap(self.compute)
        self.namespace = original.__func__.__globals__
        self.chunk = self.namespace['distillation_chunked_jsd']
        for i, teacher in self.teachers.items():
            def pre(module, inputs, i=i):
                assert self.active == i, 'An unselected teacher was executed'
                assert not module.training and not torch.is_grad_enabled()
                assert all(not p.requires_grad and p.grad is None and p.device.type == 'cuda' for p in module.parameters())
                self.calls[i] += 1
            self.handles.append(teacher.register_forward_pre_hook(pre))

        def chunk(*args, **kwargs):
            head = self.trainer.accelerator.unwrap_model(self.teachers[self.active]).get_output_embeddings()
            assert args[3].data_ptr() == head.weight.data_ptr(), 'Wrong teacher head in chunked JSD'
            return self.chunk(*args, **kwargs)
        self.namespace['distillation_chunked_jsd'] = chunk

        def routed(model, inputs, *args, **kwargs):
            clean = dict(inputs)
            ids = clean.pop('teacher_id')
            assert ids.numel() == 1
            i = int(ids.item())
            if i not in self.teachers:
                raise ValueError(f'Unknown teacher_id {i}')
            self.select(i)
            self.active = i
            self.trainer.teacher_model = self.teachers[i]
            before = dict(self.calls)
            before_chunked = self.timer.counts['chunked_jsd_calls']
            loss = self.compute(model, clean, *args, **kwargs)
            delta = {k: self.calls[k] - before[k] for k in self.calls}
            assert delta == {k: int(k == i) for k in delta}, delta
            assert self.timer.counts['chunked_jsd_calls'] == before_chunked + 1, 'Selected teacher did not use Chunked JSD'
            self.rows.append(dict(step=self.trainer.state.global_step + 1,
                                  teacher_id=i, teacher_forward_calls=delta,
                                  loss=float(loss.detach()), selected_head_verified=True))
            self.path.write_text(json.dumps({'steps': self.rows}, indent=2))
            self.active = None
            return loss
        self.trainer.compute_loss = routed

    def restore(self):
        self.trainer.compute_loss = self.compute
        self.namespace['distillation_chunked_jsd'] = self.chunk
        for handle in self.handles:
            handle.remove()
