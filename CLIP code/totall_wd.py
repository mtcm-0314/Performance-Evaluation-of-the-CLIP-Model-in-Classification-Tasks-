import os
import pandas as pd

base_dir = r'C:\Users\DELL\Desktop\Zero-ShotCLIP\Results\NoisyDA-gaussian_noise1\lr=1e-5,SGD'
output_csv_path = os.path.join(base_dir, 'total_raw.csv')

# 初始化存储所有权重下的性能指标的列表
metrics_summary = []

# 循环遍历权重衰减值文件夹（0~20，间隔为0.05）
for i in range(1, 21):
    decay_value = i * 0.05
    folder_name = f'{decay_value:.2f}'
    folder_path = os.path.join(base_dir, folder_name)

    # 初始化存储每个指标的原始数值
    results = {
        'weight_decay': decay_value,
        'ACC': None,
        'AUC': None,
        'F1': None,
        'PRE': None,
        'REC': None
    }

    # 遍历每个指标文件，读取原始数值
    for metric in results.keys():
        csv_file = f'Testing{metric}.csv'
        file_path = os.path.join(folder_path, csv_file)

        # 读取CSV文件
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)  # 假设CSV文件没有表头

            # 获取数值（假设每个文件只有一个数值）
            results[metric] = df.iloc[0, 0] if not df.empty else None

    # 将本次权重衰减值的结果添加到总列表中
    metrics_summary.append(results)

# 将性能指标写入CSV文件
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(output_csv_path, index=False, float_format='%.6f')  # 指定浮点数保存格式为六位小数

# 提示完成保存
print(f"\n性能指标的原始数值已保存到文件：{output_csv_path}")
