import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

# 載入數據
data = np.load(r"C:\Users\user\Desktop\113Camera\1033_gom_optimized_points_3d.npz")#
points = data['gom_optimized_points_3d'].reshape(773, 33, 3)

# 調整每一幀的點位置
for frame in range(points.shape[0]):
    # 計算 11-23 和 12-24 的向量
    vec_11_23 = points[frame, 23] - points[frame, 11]
    vec_12_24 = points[frame, 24] - points[frame, 12]
    
    # 計算向量的長度
    len_11_23 = np.linalg.norm(vec_11_23)
    len_12_24 = np.linalg.norm(vec_12_24)
    
    # 如果長度不相等，調整 11 的位置
    if len_11_23 != len_12_24:
        # 計算調整比例
        scale_factor = len_12_24 / len_11_23
        # 調整 11 的位置
        points[frame, 11] = points[frame, 23] - vec_11_23 * scale_factor

# 保存調整後的數據
np.savez(r'C:\Users\user\Desktop\20241027\adjusted1031_gom_optimized_points_3d.npz', gom_optimized_points_3d=points)
# 定義 MediaPipe 骨架連接
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
 
# 創建主圖形和3D軸
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.35)

# 初始化散點圖和線條
scatter = ax.scatter(points[0,:,0], points[0,:,1], points[0,:,2], c='b', marker='o', s=50)
lines = [ax.plot([], [], [], 'r-')[0] for _ in POSE_CONNECTIONS]

# 設置坐標軸範圍和標籤
ax.set_xlim([points[:,:,0].min(), points[:,:,0].max()])
ax.set_ylim([points[:,:,1].min(), points[:,:,1].max()])
ax.set_zlim([points[:,:,2].min(), points[:,:,2].max()])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D visualization')

# 更新函數
def update(frame):
    # 更新點
    scatter._offsets3d = (points[frame,:,0], points[frame,:,1], points[frame,:,2])
    
    # 更新骨架線條
    for line, connection in zip(lines, POSE_CONNECTIONS):
        start_idx, end_idx = connection
        line.set_data_3d([points[frame,start_idx,0], points[frame,end_idx,0]],
                        [points[frame,start_idx,1], points[frame,end_idx,1]],
                        [points[frame,start_idx,2], points[frame,end_idx,2]])
    
    frame_slider.set_val(frame)
    fig.canvas.draw_idle()

# 添加滑塊
ax_slider = plt.axes([0.1, 0.2, 0.65, 0.03])
frame_slider = Slider(ax_slider, '幀數', 0, points.shape[0]-1, valinit=0, valstep=1)

# 添加播放/暫停按鈕
ax_play = plt.axes([0.8, 0.2, 0.1, 0.03])
play_button = Button(ax_play, '播放/暫停')

# 添加視角控制按鈕
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

# 播放控制變量
is_playing = False
current_frame = 0

# 視角控制函數
def view_front(event):
    ax.view_init(elev=0, azim=0)
    fig.canvas.draw_idle()

def view_side(event):
    ax.view_init(elev=0, azim=90)
    fig.canvas.draw_idle()

def view_top(event):
    ax.view_init(elev=90, azim=0)
    fig.canvas.draw_idle()

def view_iso(event):
    ax.view_init(elev=45, azim=45)
    fig.canvas.draw_idle()

def view_reset(event):
    ax.view_init(elev=30, azim=-60)
    fig.canvas.draw_idle()

# 滑塊更新函數
def slider_update(val):
    global current_frame
    current_frame = int(val)
    update(current_frame)

# 播放/暫停控制
def play_pause(event):
    global is_playing
    is_playing = not is_playing
    if is_playing:
        animate()

def animate():
    global current_frame
    if is_playing:
        current_frame = (current_frame + 1) % points.shape[0]  # 修改为使用points.shape[0]
        update(current_frame)
        plt.pause(0.05)
        fig.canvas.draw_idle()
        if is_playing:
            fig.canvas.start_event_loop(0.001)
            animate()

# 連接事件處理函數
frame_slider.on_changed(slider_update)
play_button.on_clicked(play_pause)
front_button.on_clicked(view_front)
side_button.on_clicked(view_side)
top_button.on_clicked(view_top)
iso_button.on_clicked(view_iso)
reset_button.on_clicked(view_reset)

# 設置初始視角並更新第一幀v nvcfs1Q ``CXVFD5ER4 1
ax.view_init(elev=30, azim=-60)
update(0)

plt.show()

