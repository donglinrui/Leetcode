import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 创建画布
fig = plt.figure(figsize=(12, 8),
                 facecolor='lightyellow'
                 )

# 创建 3D 坐标系
ax = fig.gca(fc='whitesmoke',
             projection='3d'
             )
# 二元函数定义域
x = np.linspace(0, 900, 10)
y = np.linspace(0, 900, 10)
X, Y = np.meshgrid(x, y)

# -------------------------------- 绘制 3D 图形 --------------------------------
# 平面 z=3 的部分
# ax.plot_surface(X,
#                 Y,
#                 Z=X * 0 + 3,
#                 color='g'
#                 )
# 平面 z=2y 的部分
# ax.plot_surface(X,
#                 Y=Y,
#                 Z=Y * 2,
#                 color='y',
#                 alpha=0.6
#                 )
# 平面 z=-2y + 10 部分
ax.plot_surface(X=X,
                Y=Y,
                Z=600 - 2*X +2*Y,
                color='r',
                alpha=0.7
                )
# --------------------------------  --------------------------------

# 设置坐标轴标题和刻度
ax.set(xlabel='X',
       ylabel='Y',
       zlabel='Z',
       xlim=(0, 900),
       ylim=(0, 900),
       zlim=(0, 900),
       xticks=np.arange(0, 9, 2),
       yticks=np.arange(0, 9, 1),
       zticks=np.arange(0, 9, 1)
       )

# 调整视角
ax.view_init(elev=15,  # 仰角
             azim=10  # 方位角
             )

# 显示图形
plt.show()