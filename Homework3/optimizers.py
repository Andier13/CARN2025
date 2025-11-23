import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, Muon


# https://github.com/davda54/sam/blob/main/sam.py
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def build_optimizer(cfg: dict, model: nn.Module):

    name = cfg.get("name", "SGD")
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)

    params = model.parameters()

    if name == "SGD":
        momentum = cfg.get("momentum", 0.9)
        nesterov = cfg.get("nesterov", False)
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    elif name == "Adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "AdamW":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "Muon":
        return Muon(params, lr=lr, weight_decay=weight_decay)
    elif name == "SAM":
        base = cfg.get("base", "SGD")
        rho = cfg.get("rho", 0.05)

        base_opt_map = {
            "SGD": SGD,
            "Adam": Adam,
            "AdamW": AdamW,
        }

        base_opt = base_opt_map.get(base, SGD)
        momentum = cfg.get("momentum", 0.9)
        nesterov = cfg.get("nesterov", False)

        return SAM(
            params,
            base_optimizer=base_opt,
            rho=rho,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum if base == "SGD" else None,
            nesterov=nesterov if base == "SGD" else None
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")
