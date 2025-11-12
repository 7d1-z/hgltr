from pathlib import Path
import yaml
import json

from dataset.imbalance import ImbalanceData
from config import config_root

class PlacesLT(ImbalanceData):
    name = "places"

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
        classname_txt = config[PlacesLT.name]['classname_txt']
        with open(Path(doc_root) / classname_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                classname = line.strip().split(" ")[0]
                classnames.append(classname)
        return classnames

    @staticmethod
    def get_hierarchy_text(hierarchy_file: str) -> list[str]:
        with open(hierarchy_file, "r") as f:
            hierarchy_structure: dict[str, list|str] = json.load(f)
        text_list = []
        for index, class_hierarchy in hierarchy_structure.items():
            # 去掉root
            
            class_hierarchy.reverse()
            class_hierarchy = class_hierarchy[:-1]
            class_text = class_hierarchy[0].split("/")[-1] + " "
            for i in range(1, len(class_hierarchy)-1):
                ele = class_hierarchy[i]
                
                if isinstance(ele, list):
                    mid_text = " and ".join(ele)
                else:
                    mid_text = ele

                class_text += mid_text + " "
            text_list.append(class_text.strip())
        return text_list

    
    @staticmethod
    def generate_hierarchy():
        level1 = [
            "indoor", 
            "outdoor, natural", 
            "outdoor, man-made"
        ]
        level2_indoor = [
            "outdoor, man-made", 
            "workplace (office building, factory, lab, etc.)", 
            "home or hotel", 
            "transportation (vehicle interiors, stations, etc.)",
            "sports and leisure", 
            "cultural (art, education, religion, millitary, law, politics, etc.)"
        ]
        level2_outdoor_natural = [
            "water, ice, snow", 
            "mountains, hills, desert, sky", 
            "forest, field, jungle", 
            "man-made elements"
        ]
        level2_outdoor_manmade = [
            "transportation (roads, parking, bridges, boats, airports, etc.)",
            "cultural or historical building/place (millitary, religious)",
            "sports fields, parks, leisure spaces", 
            "industrial and construction",
            "houses, cabins, gardens, and farms", 
            "commercial buildings, shops, markets, cities, and towns"
        ]
        level2 = level2_indoor + level2_outdoor_natural + level2_outdoor_manmade
        file = './doc/places365-Scene hierarchy.xlsx'
        import pandas as pd
        df = pd.read_excel(file)
        
        tree = dict()
        for row in range(1, len(df)):
            key = row-1
            leaf = df.iloc[row].to_list()[0].strip("'")

            level_idx = df.iloc[row].to_list()[1:]
            level1_idx = level_idx[:len(level1)]
            level2_idx = level_idx[len(level1):]

            path = ["root"]
            # 处理一级分类
            level1_list = []
            for i, idx in enumerate(level1_idx):
                if idx == 1:
                    level1_list.append(level1[i])
   
            # 处理二级分类
            level2_list = []
            for i, idx in enumerate(level2_idx):
                if idx == 1:
                    level2_list.append(level2[i])

            assert len(level1_list) > 0 and len(level2_list) > 0, \
                f"Invalid hierarchy for {leaf}: {level1_list}, {level2_list}"
            if len(level1_list) == 1:
                path.append(level1_list[0])
            else:
                path.append(level1_list)

            if len(level2_list) == 1:
                path.append(level2_list[0])
            else:
                path.append(level2_list)
            path.append(leaf)
            tree[key] = path
        with open('./doc/hierarchy/places.json', 'w') as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    PlacesLT.generate_hierarchy()
        