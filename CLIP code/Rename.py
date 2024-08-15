import os

# 定义文件路径
import tarfile

txt_file_path = 'G:/Python_Project/Datasets/MultiImageNet/MultiImageNet/classes.txt'  # 替换为你的txt文档路径
# directory_path = 'G:/Python_Project/Datasets/MultiImageNet/MultiImageNet/blur/defocus_blur/1'  # 替换为包含文件夹的目录路径
directory_path = []
Loc = 'G:/Python_Project/Datasets/MultiImageNet/MultiImageNet/'

topclass = ['blur','digital','extra','noise','weather']
classes = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate',
           'gaussian_blur', 'saturate', 'spatter', 'speckle_noise', 'gaussian_noise', 'impulse_noise', 'shot_noise',
           'brightness', 'fog', 'frost', 'snow']

numbers = ['1','2','3','4','5']
epoch = [4,4,4,3,4]
m = 0
t = 0
Mappings = []
for i in topclass:
    Location = Loc + i + '/'
    for j in range(0, epoch[t]):
        LocationNew = Location + classes[m] + '/'

        for k in range(0,5):
            LocationNewNew =LocationNew + numbers[k] + '/'
            Mappings.append(classes[m] + numbers[k])
            directory_path.append(LocationNewNew)
        m += 1
    t += 1

print(directory_path)
print(Mappings)

def Rename(txt_file_path, directory_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # 创建一个从旧名称到新名称的映射字典
    name_mapping = {line.split()[ 0 ]: line.split()[ 2 ].strip() for line in lines}

    # 遍历目录下的所有文件夹
    for old_name in os.listdir(directory_path):
        old_folder_path = os.path.join(directory_path, old_name)

        # 确保是文件夹
        if os.path.isdir(old_folder_path):
            # 如果文件夹名称在映射字典中，则重命名
            if old_name in name_mapping:
                new_name = name_mapping[ old_name ]
                new_folder_path = os.path.join(directory_path, new_name)
                if os.path.exists(new_folder_path):
                    # 如果存在，可以选择删除或者抛出异常
                    print(f"目标路径 {new_folder_path} 已存在。")
                    # os.rmdir(new_folder_path)  # 如果你想要删除旧文件夹，请取消注释这一行
                else:
                    # 如果不存在，则重命名文件夹
                    os.rename(old_folder_path, new_folder_path)
                    print(f"文件夹已从 {old_folder_path} 重命名为 {new_folder_path}")
                print(f'Renamed "{old_name}" to "{new_name}"')


# for i,j in zip(directory_path, Mappings):
#     os.rename(i, j)
    # Rename(txt_file_path, i)