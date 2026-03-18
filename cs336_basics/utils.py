import math
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional
from collections.abc import Callable, Iterable
from jaxtyping import Float, Int
from typing import BinaryIO, IO

# %% SGD and Adam optimizer implementation

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, 
                 weight_decay=0.0001, betas=(0.9, 0.999)):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr, "eps": eps, "weight_decay": weight_decay,
            "beta1": betas[0], "beta2": betas[1]
        }
        super(AdamW, self).__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lambda_ = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                t += 1  # Increment iteration number
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data)) 
                
                # Get the gradient of loss with respect to p and update weight tensor in-place.
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * lambda_ * p.data
                state["t"] = t
                state["m"] = m
                state["v"] = v
            return loss

  
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super(SGD, self).__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # Increment iteration number.
            return loss

# %% loss function and scheduler

def cross_entropy(inputs: Float[torch.Tensor, " batch_size vocab_size"],
                  targets: Int[torch.Tensor, " batch_size"]) -> torch.Tensor:
    """
    Compute the average cross-entropy loss between logits and target indices.

    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size)
        targets: Target indices of shape (batch_size,)

    Returns:
        Average cross-entropy loss
    """
    log_softmax = torch.log_softmax(inputs, dim=-1)
    
    batch_size = inputs.size(0)
    targets = targets.long()
    log_probs = log_softmax[torch.arange(batch_size, device=inputs.device), targets]
    loss = -log_probs.mean()
    return loss

def get_lr_cosine_schedule(
    it: int,
    maximum_lr: float,
    minimum_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return maximum_lr * it / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return minimum_lr + 0.5 * (maximum_lr - minimum_lr) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        )
    else:
        return minimum_lr
    

# %% laod and save transformerLM checkpoint
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                    iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    Save the model and optimizer state to a file.
    """
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Load the model and optimizer state from a file.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

def get_batch(dataset: np.typing.NDArray, batch_size: int, 
              context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of data from the dataset.
    
    Args:
        dataset: 输入数据集，形状为 (seq_len,)
        batch_size: 批次大小
        context_length: 上下文长度
        device: 设备类型 ('cpu' 或 'cuda')
        
    Returns:
        tuple: (x, y) 其中：
            - x: 输入序列，形状为 (batch_size, context_length)
            - y: 目标序列，形状为 (batch_size, context_length)
    """
    if len(dataset) < context_length + 1:
        raise ValueError(f"Dataset length must be greater than context_length + 1. Now \
            dataset length: {len(dataset)}, context_length: {context_length}")
    max_start_index = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start_index + 1, size=batch_size)
    
    x = np.zeros((batch_size, context_length), dtype=np.int64)
    y = np.zeros((batch_size, context_length), dtype=np.int64)
    for i, start_index in enumerate(start_indices):
        x[i] = dataset[start_index : start_index + context_length]
        y[i] = dataset[start_index + 1 : start_index + context_length + 1]
    
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    return x_tensor, y_tensor

    

# %% main

def test_optim_method(optimizer=AdamW, lr=1e-2, **kwargs):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = optimizer([weights], lr=lr, **kwargs)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
        

# %%
if __name__ == "__main__":
    print("Testing AdamW optimizer:")
    test_optim_method(optimizer=AdamW, lr=1e-1, weight_decay=1e-2)
    print("\nTesting SGD optimizer:")
    test_optim_method(optimizer=SGD, lr=1e-1)
# %%
