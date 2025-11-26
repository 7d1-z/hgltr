import random
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from pathlib import Path
from torch.nn.parallel.distributed import DistributedDataParallel

from dataset.cifar import CIFAR100LT, CIFAR10LT
from dataset.imagenet import ImageNetLT
from dataset.inaturalist import iNaturalist
from dataset.places import PlacesLT


def set_seed(seed=42):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        seed += rank
    elif torch.cuda.device_count() > 1 and torch.distributed.is_available():
        try:
            rank = torch.distributed.get_rank()
            seed += rank
        except:
            pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_labels(dataset: str):
    if dataset == "imagenet":
        return ImageNetLT.read_classname()
    elif dataset == "inaturalist":
        return iNaturalist.read_classname()
    elif dataset == "places":
        return PlacesLT.read_classname()
    elif dataset == "cifar10":
        return CIFAR10LT().classname
    elif dataset == "cifar100":
        return CIFAR100LT().classname
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_softlabel(name: str, tau=1.0) -> torch.Tensor:
    file_path = f"dataset/{name}.npy"
    file_path = Path(file_path)
    assert file_path.exists(), f"Soft label file {file_path} does not exist."
    lca_height = np.load(file_path)
    lca_height = torch.from_numpy(lca_height)
    return torch.softmax(-lca_height * tau, dim=-1)


def test_time_ensemble(x: torch.Tensor, model: nn.Module | DistributedDataParallel):
    assert len(x.shape) == 5, "Input tensor must have 5 dimensions."
    bsz, ncrops, c, h, w = x.size()
    x = x.view(bsz * ncrops, c, h, w)
    outputs = model(x)
    if isinstance(outputs, dict):
        outputs["logit"] = outputs["logit"].view(bsz, ncrops, -1).mean(1)
        outputs["image_features"] = (
            outputs["image_features"].view(bsz, ncrops, -1).mean(1)
        )
    else:
        outputs = outputs.view(bsz, ncrops, -1).mean(1)
    return outputs


def process_eval_outputs(outputs: dict | torch.Tensor):
    if isinstance(outputs, torch.Tensor):
        return outputs

    logits = outputs.get("logit")
    return logits


def get_hierarchy_text(dataset: str) -> list[str]:
    file_path = f"./doc/hierarchy/{dataset}.json"
    if dataset == "imagenet":
        return ImageNetLT.get_hierarchy_text(file_path)
    elif dataset == "inaturalist":
        return iNaturalist.get_hierarchy_text(file_path)
    elif dataset == "places":
        return PlacesLT.get_hierarchy_text(file_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
