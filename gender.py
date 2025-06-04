import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import json
from pathlib import Path

# 定義骨架的雙側測量部分
BILATERAL_MEASURES = {
    "arm_length": ["shoulder", "elbow"],  # 手臂長度
    "forearm_length": ["elbow", "wrist"], # 前臂長度
    "leg_length": ["hip", "knee"],        # 大腿長度
    "shin_length": ["knee", "ankle"]      # 小腿長度
}

# 定義骨架的對稱測量部分
SYMMETRICAL_MEASURES = {
    "shoulder_width": ["left_shoulder", "right_shoulder"],   # 肩寬
    "hip_width": ["left_hip", "right_hip"],                 # 髖寬
    "face_width": ["left_ear", "right_ear"]                 # 臉寬
}

# 關鍵點映射
POINT_MAPPING = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_ear": 7,
    "right_ear": 8
}

# 定義骨架連接關係
LIMB_CONNECTIONS = [
    # 臉部連接
    (0, 1), (1, 2), (2, 3), (3, 7),  # 右臉輪廓
    (0, 4), (4, 5), (5, 6), (6, 8),  # 左臉輪廓
    (9, 10),  # 嘴巴
    (0, 9), (0, 10),  # 連接臉部到嘴巴
    
    # 身體骨架
    (11, 12),  # 肩膀
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (11, 23), (12, 24),  # 肩膀到臀部
    (23, 24),  # 臀部
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # 左腿
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # 右腿
]

def calculate_distance(points_3d_sequence, point1_idx, point2_idx):
    """
    計算兩個關鍵點之間的平均距離,排除離群值
    """
    distances = []
    for frame in points_3d_sequence:
        dist = np.sqrt(np.sum((frame[point1_idx] - frame[point2_idx])**2))
        distances.append(dist)
    
    distances = np.array(distances)
    
    # 計算四分位數用於離群值檢測
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    
    # 過濾離群值
    filtered_distances = distances[(distances >= Q1 - 1.5*IQR) & 
                                 (distances <= Q3 + 1.5*IQR)]
    
    # 計算平均長度
    avg_length = np.mean(filtered_distances)
    return avg_length

def generate_skeleton_config(points_3d_sequence, output_path):
    """
    生成骨架配置文件
    """
    config = {}
    
    # 計算雙側測量
    for measure, points in BILATERAL_MEASURES.items():
        print(f"計算 {measure} 的平均距離")
        
        mean_distance = 0
        for side in ["left", "right"]:
            point1 = f"{side}_{points[0]}"
            point2 = f"{side}_{points[1]}"
            
            point1_idx = POINT_MAPPING[point1]
            point2_idx = POINT_MAPPING[point2]
            
            distance = calculate_distance(points_3d_sequence, point1_idx, point2_idx)
            print(f"從 {point1} 到 {point2} 的平均距離是 {distance:.4f}")
            mean_distance += distance / 2
            
        config[measure] = round(mean_distance, 4)
    
    # 計算對稱測量
    for measure, points in SYMMETRICAL_MEASURES.items():
        point1_idx = POINT_MAPPING[points[0]]
        point2_idx = POINT_MAPPING[points[1]]
        
        distance = calculate_distance(points_3d_sequence, point1_idx, point2_idx)
        config[measure] = round(distance, 4)
    
    # 保存配置文件
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return config

def visualize_skeleton(points_3d, limb_connections):
    """可視化骨架結構"""
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製骨架連接
    for start, end in limb_connections:
        x = [points_3d[start,0], points_3d[end,0]]
        y = [points_3d[start,1], points_3d[end,1]] 
        z = [points_3d[start,2], points_3d[end,2]]
        
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.8)
        
    # 繪製關鍵點
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2],
              c='r', marker='o', s=50)
              
    # 設置坐標軸標籤和標題
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('3D Skeleton Visualization')
    
    plt.show()

def animate_skeleton(points_3d_sequence, limb_connections):
    """動畫展示骨架運動"""
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.cla()
        points = points_3d_sequence[frame]
        
        # 繪製骨架
        for start, end in limb_connections:
            x = [points[start,0], points[end,0]]
            y = [points[start,1], points[end,1]]
            z = [points[start,2], points[end,2]]
            ax.plot(x, y, z, 'b-', linewidth=2)
            
        ax.scatter(points[:,0], points[:,1], points[:,2],
                  c='r', marker='o', s=50)
                  
        # 設置視角和範圍
        ax.view_init(elev=10, azim=frame)
        ax.set_xlim([np.min(points_3d_sequence[:,:,0]),
                    np.max(points_3d_sequence[:,:,0])])
        ax.set_ylim([np.min(points_3d_sequence[:,:,1]),
                    np.max(points_3d_sequence[:,:,1])])
        ax.set_zlim([np.min(points_3d_sequence[:,:,2]),
                    np.max(points_3d_sequence[:,:,2])])
                    
    anim = FuncAnimation(fig, update,
                        frames=len(points_3d_sequence),
                        interval=50, blit=False)
                        
    plt.show()

def main():
    # 載入數據
    file_path = r"C:\Users\user\Desktop\113Camera\1033_raw_3d_points.npz"
    points_3d_sequence = np.load(file_path)['all_points_3d_original']
    
    # 生成骨架配置
    output_path = Path(file_path).parent / "skeleton_config.json"
    config = generate_skeleton_config(points_3d_sequence, output_path)
    
    # 打印結果
    print("\n骨架測量結果:")
    for measure, value in config.items():
        print(f"{measure}: {value:.4f} 單位")
    
    # 可視化和動畫展示部分保持不變
    visualize_skeleton(points_3d_sequence[0], LIMB_CONNECTIONS)
    animate_skeleton(points_3d_sequence, LIMB_CONNECTIONS)

if __name__ == "__main__":
    main()
