import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.001  # 空气的热扩散率，单位：m^2/s
T_land = 25  # 陆地温度，单位：摄氏度
T_sea = T_land - 5  # 海面温度，单位：摄氏度
L = 100  # 区域长度，单位：米
dx = 1  # 空间步长，单位：米
dt = 0.1  # 时间步长，单位：秒
nt = 100  # 时间步数

# 创建空间网格
nx = int(L / dx) + 1
x = np.linspace(0, L, nx)

# 初始化温度场
T = np.zeros(nx)
T[int(nx / 2):] = T_land
T[:int(nx / 2)] = T_sea

# 有限差分求解
for n in range(nt):
    Tn = T.copy()
    for i in range(1, nx - 1):
        T[i] = Tn[i] + alpha * dt / dx**2 * (Tn[i+1] - 2 * Tn[i] + Tn[i-1])

# 绘制温度与距离的关系图
plt.plot(x, T)
plt.xlabel('Distance (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution')
plt.grid(True)
plt.show()
