import os
import pandas as pd

base_dir = r'C:\Users\DELL\Desktop\Zero-ShotCLIP\Results\NoisyDA-gaussian_noise1\lr=1e-5,SGD'

# 初始化存储每个文件夹平均值的字典
average_values = {}

# 循环遍历权重衰减值文件夹（0~20，间隔为0.05）
for i in range(1, 21):
    decay_value = i * 0.05
    folder_name = f'{decay_value:.2f}'
    folder_path = os.path.join(base_dir, folder_name)
    # if not os.path.exists(folder_path):
    #     continue

    # 初始化四个指标的总和
    sum_results = {
        'TestingACC': 0,
        'TestingF1': 0,
        'TestingPRE': 0,
        'TestingREC': 0
    }
    count_files = 0
    # 读取每个CSV文件并计算总和
    for metric in sum_results:
        csv_file = f'Testing{metric.split("Testing")[1]}.csv'
        file_path = os.path.join(folder_path, csv_file)
        # if not os.path.exists(file_path):
        #    continue

        # 读取CSV文件
        df = pd.read_csv(file_path, header=None)  # 假设CSV文件没有表头

        # 计算每个指标的总和
        for index, row in df.iterrows():
            sum_results[metric] += float(row[0])

        count_files += len(df)

    # 计算四个指标的平均值
    for metric in sum_results:
        sum_results[metric] /= count_files

    # 计算四个指标的加和平均值
    sum_total = sum(sum_results.values())
    average_values[folder_name] = sum_total

# 找出平均值最大的权重衰减值文件夹
max_avg_value = -float('inf')
best_decay = None

for folder, total_avg in average_values.items():
    if total_avg > max_avg_value:
        max_avg_value = total_avg
        best_decay = folder

# 打印每个权重衰减值文件夹的平均值
for folder, total_avg in average_values.items():
    print(f"权重衰减值文件夹 '{folder}' 的加和平均值为：{total_avg}")

# 打印平均值最大的权重衰减值文件夹及其平均值
print(f"\n加和平均值最大的权重衰减值文件夹是：{best_decay}")
print(f"加和平均值为：{max_avg_value}")
