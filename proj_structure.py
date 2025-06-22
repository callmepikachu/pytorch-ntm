import os

def print_tree(root_path, prefix=""):
    items = sorted(os.listdir(root_path))
    for i, item in enumerate(items):
        path = os.path.join(root_path, item)
        connector = "├── " if i < len(items) - 1 else "└── "
        print(prefix + connector + item)
        if os.path.isdir(path):
            extension = "│   " if i < len(items) - 1 else "    "
            print_tree(path, prefix + extension)

print_tree("/Users/keqinpeng/PycharmProjects/pytorch-ntm")  # 换成你的项目路径