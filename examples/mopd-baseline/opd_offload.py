"""Synchronous one-resident-teacher offload, memory monitoring and validation."""
import json
import os
import resource
import threading
import time
from pathlib import Path

import torch
from opd_routing import SerialTeacherRouter


class ProcessMemoryMonitor:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.rows = []
        self.training = False
        self.training_peak_rss = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def sample(self):
        status = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, unit = line.split()
                status[key[:-1]] = int(value) * 1024
        row = dict(time=time.perf_counter(), training=self.training,
                   rss_bytes=status['VmRSS'], process_high_water_bytes=status['VmHWM'])
        self.rows.append(row)
        if self.training:
            self.training_peak_rss = max(self.training_peak_rss, row['rss_bytes'])
        return row

    def poll(self):
        while not self.done.wait(0.02):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def begin_training(self):
        self.training = True
        self.sample()

    def stop(self):
        self.sample()
        self.done.set()
        self.thread.join()
        self.training = False
        self.resource_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.process_peak_rss = max(self.resource_peak_rss,
                                   max(r['process_high_water_bytes'] for r in self.rows),
                                   max(r['rss_bytes'] for r in self.rows))
        (self.output_dir / 'cpu_memory_samples.json').write_text(json.dumps(self.rows))


class OffloadedTeacherRouter(SerialTeacherRouter):
    offload = True

    def __init__(self, trainer, teachers, timer, output_dir, monitor):
        super().__init__(trainer, teachers, timer, output_dir)
        self.resident = None
        self.events = []
        self.monitor = monitor
        self.output_dir = Path(output_dir)
        self.phase = 'migration_validation'
        timer.teacher_offload = True
        timer.cpu_monitor = monitor

    def residency(self):
        placements = {}
        cpu_bytes = gpu_bytes = 0
        for i, teacher in self.teachers.items():
            tensors = list(teacher.parameters()) + list(teacher.buffers())
            devices = {x.device.type for x in tensors}
            assert len(devices) == 1, (i, devices)
            placements[i] = devices.pop()
            for x in tensors:
                if x.device.type == 'cpu': cpu_bytes += x.numel() * x.element_size()
                else: gpu_bytes += x.numel() * x.element_size()
        assert sum(d == 'cuda' for d in placements.values()) <= 1, placements
        assert all(p.device.type == 'cuda' for p in self.trainer.model.parameters())
        optimizer = self.trainer.optimizer
        for state in optimizer.state.values():
            assert all(not isinstance(v, torch.Tensor) or v.device.type == 'cuda' for v in state.values())
        return dict(placements=placements, teacher_cpu_bytes=cpu_bytes,
                    teacher_gpu_bytes=gpu_bytes, gpu_allocated_bytes=torch.cuda.memory_allocated(),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cpu_rss_bytes=self.monitor.sample()['rss_bytes'])

    def move(self, teacher_id, device):
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.teachers[teacher_id].to(device)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def select(self, teacher_id):
        if teacher_id == self.resident:
            if self.phase == 'training':
                self.timer.current['teacher_cache_hit'] = True
            self.events.append(dict(phase=self.phase, teacher_id=teacher_id, cache_hit=True,
                teacher_h2d_time=0., teacher_d2h_time=0., teacher_switch_time=0.,
                teacher_switch_count=0, residency=self.residency()))
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        previous = self.resident
        before = self.residency()
        d2h = 0.
        if previous is not None:
            d2h = self.move(previous, 'cpu')
        unloaded = self.residency()
        h2d = self.move(teacher_id, 'cuda:0')
        self.resident = teacher_id
        after = self.residency()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = dict(phase=self.phase, previous_teacher=previous, teacher_id=teacher_id,
                   cache_hit=False, teacher_h2d_time=h2d, teacher_d2h_time=d2h,
                   teacher_switch_time=elapsed, teacher_switch_count=int(previous is not None),
                   cold_load=previous is None, before=before, unloaded=unloaded, residency=after)
        self.events.append(row)
        (self.output_dir/'teacher_transfers.json').write_text(json.dumps(self.events, indent=2))
        if self.phase == 'training':
            for key in ('teacher_h2d_time','teacher_d2h_time','teacher_switch_time','teacher_switch_count'):
                self.timer.add(key, row[key])

    def check_migration_logits(self, tokenizer):
        ids = tokenizer('A triangle has three sides. Compute 2 + 3.', return_tensors='pt').input_ids.to('cuda')
        prior = os.environ.pop('UNSLOTH_RETURN_HIDDEN_STATES', None)
        self.logit_checks = []
        try:
            for i in self.teachers:
                self.select(i)
                model = self.teachers[i]
                def logits():
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        return model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     use_cache=False).logits.detach().float().cpu()
                first = logits()
                self.move(i, 'cpu')
                self.resident = None
                self.residency()
                self.select(i)
                second = logits()
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                self.logit_checks.append(dict(teacher_id=i, shape=list(first.shape),
                    max_abs_error=float((first-second).abs().max()), bitwise_equal=torch.equal(first,second)))
            self.move(self.resident, 'cpu')
            self.resident = None
            self.residency()
        finally:
            if prior is not None: os.environ['UNSLOTH_RETURN_HIDDEN_STATES'] = prior
        (self.output_dir/'migration_logits.json').write_text(json.dumps(dict(
            input_ids=ids.cpu().tolist(), use_cache=False, checks=self.logit_checks,
            scope='GPU logits before and after GPU->CPU->GPU, fixed input, eval, no_grad'),indent=2))
        self.phase = 'training'

    def validate(self):
        frozen = all(not t.training and all(not p.requires_grad and p.grad is None for p in t.parameters()) for t in self.teachers.values())
        unchanged = all(self.versions[i] == [p._version for p in t.parameters()] for i,t in self.teachers.items())
        covered = {r['teacher_id'] for r in self.rows} == set(self.teachers)
        ok = frozen and unchanged and covered and all(x['bitwise_equal'] for x in self.logit_checks)
        self.path.write_text(json.dumps(dict(steps=self.rows, all_teachers_frozen_eval=frozen,
            teacher_parameter_versions_unchanged=unchanged, all_routes_covered=covered,
            migration_logits_verified=all(x['bitwise_equal'] for x in self.logit_checks), verified=ok),indent=2))
        events = [r for r in self.events if r['phase']=='training']
        summary = {k:sum(r[k] for r in events) for k in ('teacher_h2d_time','teacher_d2h_time','teacher_switch_time','teacher_switch_count')}
        summary.update(teacher_load_count=sum(not r['cache_hit'] for r in events),
            teacher_cache_hits=sum(r['cache_hit'] for r in events),
            peak_gpu_memory=torch.cuda.max_memory_allocated(),
            peak_cpu_memory=self.monitor.training_peak_rss,
            process_peak_cpu_memory=self.monitor.process_peak_rss,
            resource_peak_cpu_memory=self.monitor.resource_peak_rss,
            rollout_time=sum(r['rollout_seconds'] for r in self.timer.rows),
            distill_time=sum(r['training_seconds'] for r in self.timer.rows),
            optimizer_time=sum(r['optimizer_seconds'] for r in self.timer.rows),
            memory_units='bytes', time_units='seconds',
            switch_count_definition='replacement of a resident teacher; initial cold load excluded',
            peak_cpu_scope='process RSS, includes CPU teachers and temporary allocations, sampled every 20ms during training',
            final_residency=self.residency(), verified=ok)
        (self.output_dir/'offload_metrics.json').write_text(json.dumps(summary,indent=2))
        (self.output_dir/'teacher_transfers.json').write_text(json.dumps(self.events,indent=2))
        return ok
