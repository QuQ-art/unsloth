# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Iterable
from typing import Optional

import torch


def _stochastic_copy_bfloat16_(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy FP32 values to BF16 using unbiased stochastic rounding."""
    if destination.dtype != torch.bfloat16 or source.dtype != torch.float32:
        raise TypeError("stochastic BF16 copy requires a BF16 destination and FP32 source")

    # A BF16 value is the upper 16 bits of an FP32 value. Adding a uniformly
    # distributed 16-bit integer before truncation selects the adjacent BF16
    # values with probability proportional to their distance from `source`.
    rounding = torch.randint(0, 1 << 16, source.shape, dtype=torch.int32, device=source.device)
    rounded_bits = (source.view(torch.int32) + rounding).bitwise_and_(-1 << 16)
    destination.copy_(rounded_bits.view(torch.float32))


class BF16StochasticAdamW(torch.optim.Optimizer):
    """AdamW for BF16 parameters with FP32 moments and stochastic writeback.

    Keeping the parameters themselves in BF16 allows the actor and vLLM to
    alias one storage. FP32 moments preserve optimizer accuracy, while
    stochastic rounding prevents small updates (for example at lr=1e-6) from
    deterministically disappearing when written back to BF16.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if foreach not in (None, False):
            raise ValueError("BF16StochasticAdamW only supports foreach=False")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "foreach": False,
        }
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict casts floating-point state to the parameter
        # dtype. Undo that for the moments, which are deliberately FP32.
        for param, state in self.state.items():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = state[name].to(device=param.device, dtype=torch.float32)
            if isinstance(state.get("step"), torch.Tensor):
                state["step"] = int(state["step"].item())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("BF16StochasticAdamW does not support sparse gradients")
                if param.dtype != torch.bfloat16:
                    raise RuntimeError(f"BF16StochasticAdamW requires BF16 parameters, got {param.dtype}")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_fp32 = grad.float()
                if group["maximize"]:
                    grad_fp32.neg_()

                exp_avg.lerp_(grad_fp32, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

                updated_param = param.float()
                if group["weight_decay"] != 0:
                    updated_param.mul_(1 - group["lr"] * group["weight_decay"])

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                updated_param.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias_correction1)
                _stochastic_copy_bfloat16_(param, updated_param)

        return loss
