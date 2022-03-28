import torch
import numpy as np

def compute_grad_norm(model, norm_type=2):
    parameters = [p for p in model.parameters() if p.grad is not None]
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.item()