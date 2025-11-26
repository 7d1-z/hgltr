from pathlib import Path
import yaml
import json

from dataset.imbalance import ImbalanceData
from config import config_root

class ImageNetLT(ImbalanceData):
    name = "imagenet"
    def __init__(self, train=True, transform=None):

        super().__init__(train, transform, self.name)
        self.classname = self.read_classname()

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        name = self.classname[label]
        return image, label, name

    @staticmethod
    def read_classname():
        classnames = []
        with open(Path(config_root) / "dataset.yaml") as f:
            config = yaml.safe_load(f)
        doc_root = config['doc_root']
        classname_txt = config[ImageNetLT.name]['classname_txt']
        with open(Path(doc_root) / classname_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames
    
    @staticmethod
    def get_hierarchy_text(hierarchy_file: str) -> list[str]:
        with open(hierarchy_file, "r") as f:
            hierarchy_structure: dict[str,list[str]] = json.load(f)
        text_list = []
        for index, class_hierarchy in hierarchy_structure.items():
            # 去掉root
            class_hierarchy.reverse()
            class_hierarchy = class_hierarchy[:-1]
            prompt = ""
            for ele in class_hierarchy:
                prompt += ele.split(",")[-1].strip() + " "
            text = prompt.strip()
            text_list.append(text)
        return text_list
