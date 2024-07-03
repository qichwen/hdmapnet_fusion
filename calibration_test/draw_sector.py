import math
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  

def draw_sector(radius = 2, start_angle = 0, stop_angle = math.pi/4, num_points = 50):
    # 定义扇形参数  
    # radius = 2  # 半径  
    # start_angle = 0  # 起始角度（弧度）  
    # stop_angle = math.pi/4  # 终止角度（弧度），例如 45 度  
    # num_points = 50  # 用于绘制圆弧的点的数量  
    
    # 生成圆弧上的点  
    theta = np.linspace(start_angle, stop_angle, num_points)  
    x = np.cos(theta) * radius  
    y = np.sin(theta) * radius  
    z = np.zeros_like(x)  # 假设扇形在 z=0 的平面上  
    
    # # 闭合扇形：添加起始点作为结束点  
    # x = np.concatenate((x, 0))  
    # y = np.concatenate((y, 0))  
    # z = np.concatenate((z, 0))  
    # 添加扇形的圆心（如果需要的话）  
    # 注意：这通常不是闭合扇形所必需的，因为圆心不是扇形边界的一部分  
    center_x = np.array([0])  
    center_y = np.array([0])  
    center_z = np.array([0])

    x = np.concatenate((x, center_x))  
    y = np.concatenate((y, center_y))  
    z = np.concatenate((z, center_z))  
    return x, y, z
    # # 创建 Poly3DCollection 对象  
    # verts = [list(zip(x, y, z))]  # 将 x, y, z 坐标组合成顶点列表的列表  
    # collection = Poly3DCollection(verts, alpha=0.2, color=(0, 1, 0))  # 红色  
    
    # # 创建一个 3D 图形并添加集合  
    # fig = plt.figure()  
    # ax = fig.add_subplot(111, projection='3d')  
    # ax.add_collection3d(collection)  
    
    # # 设置视角等（可选）  
    # ax.set_xlim([-radius, radius])  
    # ax.set_ylim([-radius, radius])  
    # ax.set_zlim([-radius, radius])  
    plt.show()
if __name__ == "__main__":
    draw_sector()