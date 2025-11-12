from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

from config import config_root

def get_imb_num_per_cls(data, num_classes: int, ratio: int) -> list:
    """
    imbalance type: 'exp'
    $n_k=n_kr^k$, r denotes the imbalance ratio, and k is the class index.
    """
    max_num = len(data) / num_classes
    num_per_cls = []
    ratio = 1 / ratio
    for cls_idx in range(num_classes):
        num = max_num * (ratio ** (cls_idx / (num_classes - 1.0)))
        num_per_cls.append(int(num))
    return num_per_cls


def gen_imb_data(data: np.ndarray, targets, num_per_cls):

    new_data, new_targets = [], []
    targets = np.array(targets, dtype=np.int32)
    classes = np.unique(targets)
    for cls, img_num in zip(classes, num_per_cls):
        idx = np.where(targets == cls)[0]
        np.random.shuffle(idx)
        selected_idx = idx[:img_num]
        new_data.append(data[selected_idx])
        new_targets.extend([cls] * img_num)

    new_data = np.vstack(new_data)
    return new_data, new_targets


def get_imb_data(data, targets, num_classes: int, ratio: int):
    num_per_cls = get_imb_num_per_cls(data, num_classes, ratio)
    return gen_imb_data(data, targets, num_per_cls)


def get_cls_num(targets) -> np.ndarray:
    targets = np.array(targets, dtype=int)
    cls_num = np.bincount(targets)
    return cls_num

def get_imb_cls_idx(cls_num_ls) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    many_shot = (np.array(cls_num_ls) > 100).nonzero()[0]
    med_shot = ((np.array(cls_num_ls) <= 100) & (np.array(cls_num_ls) > 20)).nonzero()[0]
    few_shot = (np.array(cls_num_ls) <= 20).nonzero()[0]
    return many_shot, med_shot, few_shot

class ImbalanceData(Dataset):
    def __init__(self, train:bool, transform, name:str):

        self.img_path = []
        self.labels = []
        self.train = train
        self.transform = transform

        with open(Path(config_root) / "dataset.yaml") as f:
            config = yaml.safe_load(f)
        doc_root = config['doc_root']
        root = config['data_root']

        train_txt = config[name]['train_txt']
        test_txt = config[name]['test_txt']

        self.train_txt = Path(doc_root) / train_txt
        self.test_txt = Path(doc_root) / test_txt
        self.root = Path(root) / config[name]['folder']


        if train:
            self.txt = self.train_txt
        else:
            self.txt = self.test_txt

        with open(self.txt) as f:
            for line in f:
                self.img_path.append(Path(self.root / line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.cls_num_ls = get_cls_num(self.labels)
        self.many_shot, self.med_shot, self.few_shot = get_imb_cls_idx(self.cls_num_ls)
        self.num_classes = len(self.cls_num_ls)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label
