import torchvision
import yaml
from pathlib import Path

from config import config_root
from dataset.imbalance import get_imb_data, get_cls_num, get_imb_cls_idx


class CIFAR10LT(torchvision.datasets.CIFAR10):
    num_classes = 10

    def __init__(self, train=True, transform=None, r_imb=1):

        with open(Path(config_root) / "dataset.yaml") as f:
            config = yaml.safe_load(f)
        root = config['data_root']
        super().__init__(root, train=train, transform=transform, download=True)

        print(f'CIFAR{self.num_classes} loaded...{'Train' if train else 'Test'}')
        if train and r_imb > 1:
            self.data, self.targets = get_imb_data(self.data, self.targets,
                                                   self.num_classes, r_imb)
            print(f'Built imbalanced dataset with imbalance ratio {r_imb}')
        self.labels = self.targets
        self.classname = self.classes
        self.cls_num_ls = get_cls_num(self.targets)
        self.many_shot, self.med_shot, self.few_shot = get_imb_cls_idx(self.cls_num_ls)

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.classes[label]


class CIFAR100LT(CIFAR10LT):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
    meta = {
        "filename": "meta", "key": "fine_label_names", "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    num_classes = 100

