import yaml
import torch
import torch.nn as nn
from torch import optim

import clip
from clip.model import CLIP
from config import config_root
from peft_clip.model import ZeroShotCLIP, PeftModelFromCLIP



def load_clip_to_cpu(name:str):
    with open(config_root / "model.yaml", "r") as f:
        config = yaml.safe_load(f)
    download_root = config["download_root_clip"]
    model, _ = clip.load(name, 'cpu', download_root=download_root)
    return model



def build_model(args, clip_model: CLIP, num_classes:int=None):
    if args.model == "zero-shot":
        model = ZeroShotCLIP(clip_model)
        tuner = None
        head = None
    elif args.model == "peft":
        model = PeftModelFromCLIP(args, clip_model, num_classes)
        if args.ckpt is not None:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            model.load_state_dict(ckpt, strict=False)
        tuner = model.tuner
        head = model.head
    else:
        raise ValueError(f"Unknown peft_clip name: {args.model}")
    return model, tuner, head


def build_optimizer(lr: float, module_dic: dict[str, nn.Module], optim_name="sgd") -> tuple[optim.Optimizer, int]:
    params: list[dict] = []
    def process(name, module: nn.Module, requires_grad: bool):
        if module is None:
            print(f"Module {name} is None, skipping.")
            return
        for _, param in module.named_parameters():
            param.requires_grad_(requires_grad)
        num_params = sum(p.numel() for p in module.parameters())
        if requires_grad:
            params.append({"params": module.parameters()})
        print(f"Module {name} has {num_params/1e6:.2f}M parameters, requires_grad={requires_grad}.")

    process("model", module_dic['model'], False)
    ## the rest modules need gradient
    total_learnable_params = 0
    for name, module in module_dic.items():
        if name != 'model':
            if module is not None:
                process(name, module, True)
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total_learnable_params += num_params
            else:
                print(f"Module {name} is None, skipping.")
    if optim_name == 'sgd':
        optimizer = optim.SGD(params, lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
    return optimizer, total_learnable_params




