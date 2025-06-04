import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2
import os

# 载入数据
data_path = r"C:\Users\user\Desktop\Dropbox\SoftwareEngineeringGroup\Project\Python\smpl_simu\source\libs\STAR_local\1033_gom_optimized_points_3d\1033_gom_optimized_points_3d.npz"
data = np.load(data_path)
points = data['gom_optimized_points_3d'].reshape(773, 33, 3)

# 调整每一帧的点位置
for frame in range(points.shape[0]):
    # 计算 11-23 和 12-24 的向量
    vec_11_23 = points[frame, 23] - points[frame, 11]
    vec_12_24 = points[frame, 24] - points[frame, 12]
    
    # 计算向量的长度
    len_11_23 = np.linalg.norm(vec_11_23)
    len_12_24 = np.linalg.norm(vec_12_24)
    
    # 如果长度不相等，调整 11 的位置
    if len_11_23 != len_12_24:
        # 计算调整比例
        scale_factor = len_12_24 / len_11_23
        # 调整 11 的位置
        points[frame, 11] = points[frame, 23] - vec_11_23 * scale_factor

# 确保输出目录存在
output_dir = r"C:\Users\user\Desktop\113Camera\output"
os.makedirs(output_dir, exist_ok=True)

# 保存调整后的数据
output_path = os.path.join(output_dir, "adjusted1033_gom_optimized_points_3d.npz")
np.savez(output_path, gom_optimized_points_3d=points)
print(f"数据已保存到: {output_path}")

# 相机内参数（假设值，需要根据实际情况调整）
# 格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
camera_matrix = np.array([
    [1000, 0, 640],
    [0, 1000, 480],
    [0, 0, 1]
], dtype=np.float32)

# 畸变系数（假设无畸变）
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 定义图像大小
img_width, img_height = 1280, 960

# 定义 MediaPipe 骨架连接
POSE_CONNECTIONS = [
    # 臉部
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # 身體
    (9, 10),
    (11, 12),  # 肩膀連接
    (11, 23),  # 左肩到左髖
    (12, 24),  # 右肩到右髖
    (23, 24),  # 髖部連接
    # 左手
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 21), (15, 19),
    # 右手
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (16, 20),
    # 左腳
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # 右腳
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    # 臉部到身體的連接
    (0, 9),    # 鼻子到頸部中心
    (9, 11),   # 頸部中心到左肩
    (9, 12)    # 頸部中心到右肩
]
 
# 创建主图形
fig = plt.figure(figsize=(16, 10))

# 创建3D和2D子图
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)
plt.subplots_adjust(bottom=0.35)

# 初始化散点图和线条 (3D)
scatter3d = ax3d.scatter(points[0,:,0], points[0,:,1], points[0,:,2], c='b', marker='o', s=50)
lines3d = [ax3d.plot([], [], [], 'r-')[0] for _ in POSE_CONNECTIONS]

# 设置3D坐标轴范围和标签
ax3d.set_xlim([points[:,:,0].min(), points[:,:,0].max()])
ax3d.set_ylim([points[:,:,1].min(), points[:,:,1].max()])
ax3d.set_zlim([points[:,:,2].min(), points[:,:,2].max()])
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('3D Visualization')

# 初始化2D图像
ax2d.set_xlim([0, img_width])
ax2d.set_ylim([img_height, 0])  # 注意y轴反转以匹配图像坐标
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')
ax2d.set_title('2D Projection')
ax2d.set_aspect('equal')

# 初始化2D点和线
scatter2d = ax2d.scatter([], [], c='b', marker='o', s=50)
lines2d = [ax2d.plot([], [], 'r-')[0] for _ in POSE_CONNECTIONS]

# 使用solvePnP计算相机外部参数并投影3D点到2D平面
def project_3d_to_2d(points_3d, camera_config=None):
    global current_camera
    if camera_config is None:
        camera_config = camera_configs[current_camera]
    
    # 将3D点转换为齐次坐标
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # 构建相机外参矩阵
    rotation_matrix, _ = cv2.Rodrigues(camera_config['rotation'])
    extrinsic_matrix = np.hstack((rotation_matrix, camera_config['translation'].reshape(3, 1)))
    
    # 投影矩阵
    projection_matrix = camera_config['matrix'] @ extrinsic_matrix
    
    # 投影到2D
    points_2d_homogeneous = points_3d_homogeneous @ projection_matrix.T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
    
    return points_2d

# 更新函数
def update(frame):
    # 更新3D点
    scatter3d._offsets3d = (points[frame,:,0], points[frame,:,1], points[frame,:,2])
    
    # 更新3D骨架线条
    for line, connection in zip(lines3d, POSE_CONNECTIONS):
        start_idx, end_idx = connection
        line.set_data_3d([points[frame,start_idx,0], points[frame,end_idx,0]],
                        [points[frame,start_idx,1], points[frame,end_idx,1]],
                        [points[frame,start_idx,2], points[frame,end_idx,2]])
    
    # 计算当前帧的2D投影
    points_2d = project_3d_to_2d(points[frame])
    
    # 更新2D点
    scatter2d.set_offsets(points_2d)
    
    # 更新2D骨架线条
    for line, connection in zip(lines2d, POSE_CONNECTIONS):
        start_idx, end_idx = connection
        line.set_data([points_2d[start_idx,0], points_2d[end_idx,0]],
                      [points_2d[start_idx,1], points_2d[end_idx,1]])
    
    frame_slider.set_val(frame)
    fig.canvas.draw_idle()

# 添加滑块
ax_slider = plt.axes([0.1, 0.2, 0.65, 0.03])
frame_slider = Slider(ax_slider, '帧数', 0, points.shape[0]-1, valinit=0, valstep=1)

# 添加播放/暂停按钮
ax_play = plt.axes([0.8, 0.2, 0.1, 0.03])
play_button = Button(ax_play, '播放/暂停')

# 添加视角控制按钮
ax_front = plt.axes([0.1, 0.1, 0.1, 0.03])
ax_side = plt.axes([0.25, 0.1, 0.1, 0.03])
ax_top = plt.axes([0.4, 0.1, 0.1, 0.03])
ax_iso = plt.axes([0.55, 0.1, 0.1, 0.03])
ax_reset = plt.axes([0.7, 0.1, 0.1, 0.03])

front_button = Button(ax_front, 'Front')
side_button = Button(ax_side, 'Side')
top_button = Button(ax_top, 'Top')
iso_button = Button(ax_iso, 'Iso')
reset_button = Button(ax_reset, 'Reset')

# 定义全局变量
current_camera = 'front'
is_playing = False
current_frame = 0

# 视角控制函数
def view_front(event):
    ax3d.view_init(elev=0, azim=0)
    fig.canvas.draw_idle()

def view_side(event):
    ax3d.view_init(elev=0, azim=90)
    fig.canvas.draw_idle()

def view_top(event):
    ax3d.view_init(elev=90, azim=0)
    fig.canvas.draw_idle()

def view_iso(event):
    ax3d.view_init(elev=45, azim=45)
    fig.canvas.draw_idle()

def view_reset(event):
    ax3d.view_init(elev=30, azim=-60)
    fig.canvas.draw_idle()

# 滑块更新函数
def slider_update(val):
    global current_frame
    current_frame = int(val)
    update(current_frame)

# 播放/暂停控制
def play_pause(event):
    global is_playing
    is_playing = not is_playing
    if is_playing:
        animate()

def animate():
    global current_frame
    if is_playing:
        current_frame = (current_frame + 1) % points.shape[0]
        update(current_frame)
        plt.pause(0.05)
        fig.canvas.draw_idle()
        if is_playing:
            fig.canvas.start_event_loop(0.001)
            animate()

# 连接事件处理函数
frame_slider.on_changed(slider_update)
play_button.on_clicked(play_pause)
front_button.on_clicked(view_front)
side_button.on_clicked(view_side)
top_button.on_clicked(view_top)
iso_button.on_clicked(view_iso)
reset_button.on_clicked(view_reset)

# 设置初始视角并更新第一帧
ax3d.view_init(elev=30, azim=-60)
update(0)

camera_configs = {
    'front': {
        'rotation': np.array([0, 0, 0], dtype=np.float32),
        'translation': np.array([0, 0, 1000], dtype=np.float32),
        'matrix': np.array([
            [1000, 0, 640],
            [0, 1000, 480],
            [0, 0, 1]
        ], dtype=np.float32)
    },
    'side': {...},
    'top': {...}
}

# 视角控制
ax_radio = plt.axes([0.8, 0.3, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('front', 'side', 'top'))

# 缩放控制
ax_scale = plt.axes([0.1, 0.1, 0.3, 0.03])
scale_slider = Slider(ax_scale, '缩放', 0.1, 2.0, valinit=1.0)

plt.show()