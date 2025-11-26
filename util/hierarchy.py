import json
import numpy as np
import argparse
from pathlib import Path


def load_hierarchy(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    hierarchy = {}
    for key in data:
        path = data[key]
        hierarchy[int(key)] = path  # transfer index to int
    return hierarchy


def process_list_node(node1: str | list, node2: str | list):
    """
    if node1 ^ node2 is not None, we consider they have intersection
    """
    unique_category = set()
    total_size = 0

    def process_input(node: str | list, collector: set):
        cnt = 0
        if isinstance(node, str):
            collector.add(node)
            cnt += 1
        elif isinstance(node, list):
            for i in node:
                collector.add(i)
                cnt += 1
        return cnt

    total_size += process_input(node1, unique_category)
    total_size += process_input(node2, unique_category)
    return len(unique_category) != total_size  # they have intersection


def find_lca_height(index1: int, index2: int, hierarchy: dict):
    path1 = hierarchy[index1]
    path2 = hierarchy[index2]
    min_len = min(len(path1), len(path2))

    lca_index = -1
    for i in range(min_len):
        if isinstance(path1[i], str) and isinstance(path2[i], str):
            if path1[i] == path2[i]:
                lca_index = i
            else:
                break
        else:
            if process_list_node(path1[i], path2[i]):
                lca_index = i
            else:
                break

    # we put the root node at the beginning of the path
    if lca_index == 0:
        return float("inf")

    height1 = len(path1) - lca_index - 1
    height2 = len(path2) - lca_index - 1
    return max(height1, height2)


def build_lca_height(hierarchy: dict):
    n = len(hierarchy)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):  # the matrix is symmetric
            if i == j:
                dist_matrix[i][j] = 0
            else:
                height = find_lca_height(i, j, hierarchy)
                dist_matrix[i][j] = height
                dist_matrix[j][i] = height
    return dist_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filename = args.json
    assert filename is not None, "Please provide a JSON file containing the hierarchy."
    filename = Path(filename)
    hierarchy = load_hierarchy(filename)
    ## 获取文件名，并将文件名后缀改为.npy并保存到 ./目录
    save_filename = f"dataset/{filename.stem}.npy"
    dist_file = Path(save_filename)

    print(find_lca_height(0, len(hierarchy) - 1, hierarchy))  # Example usage
    if dist_file.exists():
        print(
            f"Distance matrix file {dist_file} already exists. Skipping distance matrix computation."
        )
    else:
        print(f"Building distance matrix for {filename}...")
        dist_matrix = build_lca_height(hierarchy)
        np.save(dist_file, dist_matrix)
