import cv2
import numpy as np
from scipy.optimize import least_squares

def correct_multi_camera_bundle_adjustment(
    objpoints_all,  # 所有帧的3D点 [frame_num, point_num, 3]
    imgpoints_all,  # 所有相機的2D点 {cam_idx: [frame_num, point_num, 2]}
    camera_matrices,  # 各相机内参
    dist_coeffs,     # 各相机畸变参数
    num_cameras,     # 相机数量
    reference_cam=0  # 参考相机（世界坐标系原点）
):
    """
    正确的多相机Bundle Adjustment实现
    
    核心思想：
    1. 以reference_cam为世界坐标系原点
    2. 优化其他相机相对于参考相机的外参
    3. 同时优化每一帧棋盘格在世界坐标系中的位姿
    """
    
    num_frames = len(objpoints_all)
    
    def objective_function(x):
        """
        目标函数：最小化所有相机所有帧的重投影误差
        
        参数布局：
        - x[0:6*(num_cameras-1)]: 非参考相机的外参 (rvec, tvec)
        - x[6*(num_cameras-1):]: 每帧棋盘格的位姿 (rvec, tvec)
        """
        total_errors = []
        
        # 解包相机外参 (相对于参考相机)
        camera_poses = {}
        camera_poses[reference_cam] = (np.zeros(3), np.zeros(3))  # 参考相机外参为0
        
        param_idx = 0
        for cam_idx in range(num_cameras):
            if cam_idx != reference_cam:
                rvec_cam = x[param_idx:param_idx+3]
                tvec_cam = x[param_idx+3:param_idx+6]
                camera_poses[cam_idx] = (rvec_cam, tvec_cam)
                param_idx += 6
        
        # 解包棋盘格位姿 (在世界坐标系中)
        board_poses = []
        for frame_idx in range(num_frames):
            rvec_board = x[param_idx:param_idx+3]
            tvec_board = x[param_idx+3:param_idx+6]
            board_poses.append((rvec_board, tvec_board))
            param_idx += 6
        
        # 计算每个相机每一帧的重投影误差
        for cam_idx in range(num_cameras):
            if cam_idx not in imgpoints_all:
                continue
                
            rvec_cam, tvec_cam = camera_poses[cam_idx]
            R_cam, _ = cv2.Rodrigues(rvec_cam)
            
            for frame_idx in range(num_frames):
                if frame_idx >= len(imgpoints_all[cam_idx]):
                    continue
                    
                # 获取棋盘格在世界坐标系中的位姿
                rvec_board, tvec_board = board_poses[frame_idx]
                R_board, _ = cv2.Rodrigues(rvec_board)
                
                # 将棋盘格点从棋盘格坐标系转换到世界坐标系
                points_3d_world = []
                for point_3d in objpoints_all[frame_idx]:
                    point_world = R_board @ point_3d.reshape(3, 1) + tvec_board.reshape(3, 1)
                    points_3d_world.append(point_world.flatten())
                points_3d_world = np.array(points_3d_world)
                
                # 投影到当前相机
                projected, _ = cv2.projectPoints(
                    points_3d_world,
                    rvec_cam,
                    tvec_cam,
                    camera_matrices[cam_idx],
                    dist_coeffs[cam_idx]
                )
                projected = projected.reshape(-1, 2)
                
                # 计算重投影误差
                observed = imgpoints_all[cam_idx][frame_idx].reshape(-1, 2)
                error = (observed - projected).ravel()
                total_errors.extend(error)
        
        return np.array(total_errors)
    
    # 初始化参数
    # 1. 初始化相机外参（通过双目标定或其他方法）
    # 2. 初始化棋盘格位姿（通过solvePnP）
    
    # 这里简化初始化过程，实际应用中需要更好的初始化
    x0 = []
    
    # 初始化非参考相机的外参
    for cam_idx in range(num_cameras):
        if cam_idx != reference_cam:
            # 简单初始化，实际应该通过更好的方法获得初值
            x0.extend([0, 0, 0, 0, 0, 0])  # rvec, tvec
    
    # 初始化每帧棋盘格位姿
    for frame_idx in range(num_frames):
        if reference_cam in imgpoints_all and frame_idx < len(imgpoints_all[reference_cam]):
            # 使用参考相机来初始化棋盘格位姿
            ret, rvec, tvec = cv2.solvePnP(
                objpoints_all[frame_idx],
                imgpoints_all[reference_cam][frame_idx],
                camera_matrices[reference_cam],
                dist_coeffs[reference_cam]
            )
            x0.extend(rvec.flatten())
            x0.extend(tvec.flatten())
        else:
            x0.extend([0, 0, 0, 0, 0, 0])  # 默认初始化
    
    x0 = np.array(x0)
    
    # 运行优化
    print("开始Bundle Adjustment优化...")
    result = least_squares(
        objective_function,
        x0,
        method='lm',
        verbose=2,
        max_nfev=1000,
        ftol=1e-8,
        xtol=1e-8
    )
    
    print(f"优化完成，最终残差: {result.cost}")
    
    # 解析优化结果
    optimized_camera_poses = {}
    optimized_board_poses = []
    
    param_idx = 0
    optimized_camera_poses[reference_cam] = (np.zeros(3), np.zeros(3))
    
    for cam_idx in range(num_cameras):
        if cam_idx != reference_cam:
            rvec_cam = result.x[param_idx:param_idx+3]
            tvec_cam = result.x[param_idx+3:param_idx+6]
            optimized_camera_poses[cam_idx] = (rvec_cam, tvec_cam)
            param_idx += 6
    
    for frame_idx in range(num_frames):
        rvec_board = result.x[param_idx:param_idx+3]
        tvec_board = result.x[param_idx+3:param_idx+6]
        optimized_board_poses.append((rvec_board, tvec_board))
        param_idx += 6
    
    return optimized_camera_poses, optimized_board_poses

# 使用示例和关键改进点说明
"""
关键改进点：

1. **正确的参数化**：
   - 优化相机间的相对外参，而不是相机相对于棋盘格的外参
   - 同时优化棋盘格在世界坐标系中的位姿

2. **统一坐标系**：
   - 以一个相机为参考建立世界坐标系
   - 所有其他相机的外参都相对于这个参考相机

3. **全局优化**：
   - 所有相机、所有帧同时参与优化
   - 最小化全局重投影误差

4. **正确的投影过程**：
   棋盘格点 -> 世界坐标系 -> 相机坐标系 -> 图像平面
   
5. **更好的初始化**：
   - 可以先用双目标定或其他方法获得相机间的初始外参
   - 用参考相机的solvePnP结果初始化棋盘格位姿
"""

def print_algorithm_comparison():
    """
    打印算法对比说明
    """
    print("=" * 60)
    print("Bundle Adjustment 算法对比")
    print("=" * 60)
    
    print("\n❌ 原算法的问题:")
    print("1. 优化相机相对于棋盘格的外参，而非相机间外参")
    print("2. 分组处理，缺乏全局优化")  
    print("3. 通过双目标定连接子系统，误差累积")
    print("4. 初始化概念错误")
    
    print("\n✅ 正确算法的特点:")
    print("1. 优化相机间的相对外参")
    print("2. 建立统一的世界坐标系")
    print("3. 全局同时优化所有参数")
    print("4. 最小化全局重投影误差")
    
    print("\n🔧 建议的修改步骤:")
    print("1. 重新设计参数化方式")
    print("2. 实现正确的目标函数")  
    print("3. 改进初始化方法")
    print("4. 使用全局优化策略")

if __name__ == "__main__":
    print_algorithm_comparison() 