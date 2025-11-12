import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset.imagenet import ImageNetLT
from dataset.inaturalist import iNaturalist
from dataset.places import PlacesLT
from dataset.cifar import CIFAR10LT, CIFAR100LT
from util.sampler import ClassAwareSampler, BalancedDatasetSampler

def build_transforms(name: str, train: bool, resolution: int, tte=False):
    stats = {
        "places": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "imagenet":([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        "inaturalist": ([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
        "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        "cifar100": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    }

    mean, std = stats[name]
    if train:
        compose = [
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    else:
        if tte:
            expand = 24
            compose =[
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
            ]
        else:
            compose = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]

    return transforms.Compose(compose)

def build_dataset(name: str, transform=None, train=True, r_imb=1):
    datasets = {
        "places": PlacesLT,
        "inaturalist": iNaturalist,
        "imagenet": ImageNetLT,
        "cifar10": CIFAR10LT,
        "cifar100": CIFAR100LT,
    }
    return datasets[name](train, transform, r_imb) if 'cifar' in name else datasets[name](train, transform)

def build_dataloader(dataset, batch_size, num_workers, train: bool, sampler=None):
    shuffle = True if train and sampler is None else False
    return DataLoader(dataset, batch_size, shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )


class DataloaderBuilder:
    def __init__(self, name: str, batch_size=1, num_workers=0, resolution=224, r_imb=1, tte=False):
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.r_imb = r_imb

        transform = build_transforms(name, True, resolution)
        train_dataset = build_dataset(name, transform, True, r_imb)
        transform = build_transforms(name, False, resolution, tte=tte)
        test_dataset = build_dataset(name, transform, False)

        self.num_classes = train_dataset.num_classes
        self.many_shot = train_dataset.many_shot
        self.med_shot = train_dataset.med_shot
        self.few_shot = train_dataset.few_shot
        self.classname = train_dataset.classname
        self.cls_num_ls = train_dataset.cls_num_ls
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.resolution = resolution
        self.train_dataloader = None
        self.test_dataloader = None

    def train(self, sampler:str=None):
        dataset = self.train_dataset
        if sampler == "balanced":
            # sampler = ClassAwareSampler(dataset.labels)
            sampler = BalancedDatasetSampler(dataset.labels)
        else:
            sampler = None
        self.train_dataloader = build_dataloader(dataset, self.batch_size, self.num_workers, True, sampler)
        return self.train_dataloader

    def test(self):
        dataset = self.test_dataset
        self.test_dataloader = build_dataloader(dataset, self.batch_size, self.num_workers, False)
        return self.test_dataloader
