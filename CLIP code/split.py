import os
import shutil
import random
from collections import defaultdict

# 数据集根目录
dataset_root = r'C:\Users\DELL\Desktop\Zero-ShotCLIP\dataset\retina_dataset-master\train'  # 替换成你的数据集根目录
test_ratio = 0.2

# 收集数据集中的文件
category_files = defaultdict(list)
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.endswith('.png'):  # 只考虑jpg文件
            category = os.path.basename(root)
            category_files[category].append(os.path.join(root, file))

# 创建存放测试数据的根文件夹（如果不存在的话）
test_root = r'C:\Users\DELL\Desktop\Zero-ShotCLIP\dataset\retina_dataset-master\test'
if not os.path.exists(test_root):
    os.makedirs(test_root)

# 遍历每个分类，创建分类文件夹并复制文件
for category, files in category_files.items():
    test_category_folder = os.path.join(test_root, category)
    if not os.path.exists(test_category_folder):
        os.makedirs(test_category_folder)

    # 计算当前分类的测试集大小
    test_size = int(len(files) * test_ratio)
    test_set = random.sample(files, test_size)

    # 将测试集文件复制到对应的测试分类文件夹中
    for file in test_set:
        shutil.copy(file, os.path.join(test_category_folder, os.path.basename(file)))
        # 删除训练集中的文件
        os.remove(file)

print(f"已将测试数据复制到'{test_root}'文件夹中，并从训练集中删除相应文件，保持了原有的分类结构。")