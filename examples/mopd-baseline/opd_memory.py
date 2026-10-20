"""Actual tensor-storage inventories and CUDA allocator checkpoints (no estimates)."""
import json
from pathlib import Path

import torch


def tensor_inventory(named_tensors):
    seen = set()
    groups = {}
    count = 0
    elements = 0
    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        count += 1
        elements += tensor.numel()
        group = f'{name}|{tensor.device}|{tensor.dtype}'
        row = groups.setdefault(group, {'tensor_count': 0, 'storage_bytes': 0, 'numel': 0})
        row['tensor_count'] += 1
        row['storage_bytes'] += storage.nbytes()
        row['numel'] += tensor.numel()
    cuda_bytes = sum(v['storage_bytes'] for k, v in groups.items() if '|cuda:' in k)
    total_bytes = sum(v['storage_bytes'] for v in groups.values())
    return {'tensor_count': count, 'numel': elements, 'storage_bytes': total_bytes,
            'cuda_bytes': cuda_bytes, 'cuda_gib': cuda_bytes / 2**30, 'groups': groups}


def model_inventory(model):
    params = list(model.parameters())
    return {'parameters': tensor_inventory(('parameter', p) for p in params),
            'buffers': tensor_inventory(('buffer', b) for b in model.buffers()),
            'trainable_numel': sum(p.numel() for p in params if p.requires_grad),
            'total_numel': sum(p.numel() for p in params),
            'trainable_tensor_count': sum(p.requires_grad for p in params),
            'tensor_count': len(params),
            'has_peft_config': bool(getattr(model, 'peft_config', None)),
            'adapter_parameter_names': [n for n, _ in model.named_parameters() if 'lora_' in n],
            'quantized': bool(getattr(model, 'is_loaded_in_4bit', False) or
                              getattr(model, 'is_loaded_in_8bit', False))}


def optimizer_inventory(optimizer):
    if optimizer is None:
        return {'state': tensor_inventory([]), 'parameter_numel': 0}
    tensors = []
    def walk(name, value):
        if isinstance(value, torch.Tensor):
            tensors.append((name, value))
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(f'{name}.{key}' if name else str(key), child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(name, child)
    for state in optimizer.state.values():
        walk('', state)
    parameters = {id(p): p for group in optimizer.param_groups for p in group['params']}
    base = getattr(optimizer, 'optimizer', optimizer)
    return {'class': f'{type(base).__module__}.{type(base).__name__}',
            'state': tensor_inventory(tensors),
            'parameter_numel': sum(p.numel() for p in parameters.values()),
            'parameter_tensor_count': len(parameters)}


class MemoryAudit:
    def __init__(self, output_dir):
        self.path = Path(output_dir) / 'memory_audit.json'
        self.rows = []
        self.student = None
        self.teacher = None
        self.teachers = None
        self.trainer = None

    def capture(self, stage):
        torch.cuda.synchronize()
        row = {'stage': stage, 'allocated_bytes': torch.cuda.memory_allocated(),
               'reserved_bytes': torch.cuda.memory_reserved(),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        if self.student is not None:
            row['student'] = model_inventory(self.student)
            row['student_gradients'] = tensor_inventory(('gradient', p.grad) for p in self.student.parameters())
        if self.teacher is not None:
            row['teacher'] = model_inventory(self.teacher)
            row['teacher_gradients'] = tensor_inventory(('gradient', p.grad) for p in self.teacher.parameters())
        if self.teachers:
            row['teachers'] = {str(i): model_inventory(t) for i, t in self.teachers.items()}
            row['all_teachers_frozen_eval'] = all(not t.training and all(not p.requires_grad and p.grad is None for p in t.parameters()) for t in self.teachers.values())
        if self.trainer is not None:
            row['optimizer'] = optimizer_inventory(self.trainer.optimizer)
            if self.trainer.optimizer is not None:
                actual = {id(p) for g in self.trainer.optimizer.param_groups for p in g['params']}
                expected = {id(p) for p in self.student.parameters() if p.requires_grad}
                row['optimizer_matches_trainable_student'] = actual == expected
                all_teachers = self.teachers.values() if self.teachers else [self.teacher]
                row['optimizer_contains_teacher'] = bool(actual & {id(p) for t in all_teachers for p in t.parameters()})
        self.rows.append(row)
        self.path.write_text(json.dumps(self.rows, indent=2))
        print('OPD_MEMORY', stage, 'allocated_gib=', round(row['allocated_bytes']/2**30, 4),
              'optimizer_state_gib=', row.get('optimizer', {}).get('state', {}).get('cuda_gib'), flush=True)
        return row
