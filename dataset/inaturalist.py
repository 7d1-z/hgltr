import json
import yaml
from pathlib import Path

from dataset.imbalance import ImbalanceData
from config import config_root


class iNaturalist(ImbalanceData):
    name = "inaturalist"
    def __init__(self, train=True, transform=None):
        super().__init__(train, transform, self.name)
        self.classname = self.read_classname()


    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        name = self.classname[label]
        return image, label, name


    @staticmethod
    def read_category():
        with open(Path(config_root) / "dataset.yaml") as f:
            config = yaml.safe_load(f)
        doc_root = config['doc_root']
        classname_txt = config[iNaturalist.name]['classname_json']

        with open(Path(doc_root) / classname_txt, "r") as f:
            category = json.load(f)
        return category
    @staticmethod
    def read_classname():
        classname = []
        category = iNaturalist.read_category()
        for i in category:
            classname.append(i['name'])
        return classname

    @staticmethod
    def generate_hierarchy():
        tree = {}
        level_key = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'name']
        category = iNaturalist.read_category()
        for i in category:
            paths = ['root']
            for level in level_key:
                paths.append(i[level])
            tree[i['id']] = paths
        with open("./doc/hierarchy/inaturalist.json", "w") as f:
            json.dump(tree, f, ensure_ascii=False, indent=4)
    
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


if __name__ == "__main__":
    iNaturalist.generate_hierarchy()