"""
多相机标定与Bundle Adjustment优化程序

修改日期：2024年11月
主要修改内容：
1. 修正了Bundle Adjustment算法的根本性错误
2. 从"优化相机相对于棋盘格的外参"改为"优化相机间的相对外参"
3. 实现了正确的多相机BA算法：
   - 建立统一的世界坐标系（以相机0和相机7为参考）
   - 同时优化相机外参和棋盘格位姿
   - 最小化全局重投影误差
4. 改进了参数初始化方法
5. 增加了详细的算法说明和错误处理

核心改进：
- 参数化方式：相机间相对外参 + 棋盘格世界位姿
- 投影流程：棋盘格坐标 -> 世界坐标 -> 相机坐标 -> 图像坐标
- 优化目标：全局重投影误差最小化
"""

import cv2
import numpy as np
import glob
import os
from scipy.optimize import least_squares

# ---------------------------- 1. 相机 0-3 的校准 ---------------------------- #

# 棋盘格参数
nx = 9  # 水平方向的角点数
ny = 6  # 垂直方向的角点数
square_size = 3.0  # 方格尺寸，单位：mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# 准备对象点，例如 (0,0,0), (3,0,0), ..., (24,15,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_size

# 创建存储对象点和图像点的列表
objpoints_0_3 = []  # 3D 点
imgpoints_0_3 = {i: [] for i in range(4)}  # 每个相机的 2D 点

# 加载相机 0-3 的图像
image_paths_0_3 = {}
for cam_idx in range(4):
    image_dir = f"D:/CalibrationImages/Cam{cam_idx}"
    if not os.path.exists(image_dir):
        print(f"相机 {cam_idx} 的图像目录不存在：{image_dir}")
        exit(1)
    image_paths_0_3[cam_idx] = glob.glob(os.path.join(image_dir, "*.jpeg"))
    image_paths_0_3[cam_idx].sort()
    if len(image_paths_0_3[cam_idx]) == 0:
        print(f"相机 {cam_idx} 的图像目录中没有找到 jpeg 图像")
        exit(1)

# 确保所有相机的图像数量一致
num_images_0_3 = len(image_paths_0_3[0])
for cam_idx in range(1, 4):
    if len(image_paths_0_3[cam_idx]) != num_images_0_3:
        print(f"相机 {cam_idx} 的图像数量与相机 0 不一致")
        exit(1)

# 遍历每一组图像
for i in range(num_images_0_3):
    imgs = {}
    ret_corners = {}
    corners = {}

    # 读取所有相机的图像
    for cam_idx in range(4):
        img_path = image_paths_0_3[cam_idx][i]
        imgs[cam_idx] = cv2.imread(img_path)
        if imgs[cam_idx] is None:
            print(f"无法读取相机 {cam_idx} 的图像：{img_path}")
            continue
        gray = cv2.cvtColor(imgs[cam_idx], cv2.COLOR_BGR2GRAY)
        ret_corners[cam_idx], corners[cam_idx] = cv2.findChessboardCorners(
            gray, (nx, ny), None
        )

    # 检查所有相机是否都找到了角点
    if all(ret_corners.values()):
        # 精细化角点并收集数据
        objpoints_0_3.append(objp)
        for cam_idx in range(4):
            gray = cv2.cvtColor(imgs[cam_idx], cv2.COLOR_BGR2GRAY)
            corners2 = cv2.cornerSubPix(
                gray, corners[cam_idx], (11, 11), (-1, -1), criteria
            )
            imgpoints_0_3[cam_idx].append(corners2)
    else:
        print(f"在图像索引 {i} 处并非所有相机都检测到角点，跳过此组")

# 对每个相机进行单独校准
calibration_results_0_3 = {}
image_size = (imgs[0].shape[1], imgs[0].shape[0])  # 假设所有图像大小相同

for cam_idx in range(4):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_0_3, imgpoints_0_3[cam_idx], image_size, None, None
    )
    calibration_results_0_3[cam_idx] = {
        "ret": ret,
        "mtx": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
    }
    print(f"相机 {cam_idx} 校准完成。RMS 误差：{ret}")

# 保存校准结果
np.savez(
    "single_camera_calibration_0_3.npz",
    calibration_results=calibration_results_0_3,
    image_size=image_size,
)
print("相机 0-3 的校准结果已保存到 single_camera_calibration_0_3.npz")
# ---------------------------- 2. 相机 0-3 的外参估计和优化 ---------------------------- #

# 加载单相机校准结果
calib_data_0_3 = np.load("single_camera_calibration_0_3.npz", allow_pickle=True)
calibration_results_0_3 = calib_data_0_3["calibration_results"].item()
image_size = calib_data_0_3["image_size"]

# 初始化相机参数列表
camera_matrix_list_0_3 = [calibration_results_0_3[i]["mtx"] for i in range(4)]
dist_coeff_list_0_3 = [calibration_results_0_3[i]["dist"] for i in range(4)]
R_list_0_3 = [np.eye(3) for _ in range(4)]  # 初始旋转矩阵
T_list_0_3 = [np.zeros((3, 1)) for _ in range(4)]  # 初始平移向量
rvec_list_0_3 = [np.zeros((3, 1)) for _ in range(4)]  # 初始旋转向量
tvec_list_0_3 = [np.zeros((3, 1)) for _ in range(4)]  # 初始平移向量

# 检测角点数据已在前面收集到：objpoints_0_3, imgpoints_0_3

# 计算每个相机的初始外参（相对于相机 0）
for cam_idx in range(1, 4):
    ret, rvec, tvec = cv2.solvePnP(
        objpoints_0_3[0],
        imgpoints_0_3[cam_idx][0],
        camera_matrix_list_0_3[cam_idx],
        dist_coeff_list_0_3[cam_idx]
    )
    R, _ = cv2.Rodrigues(rvec)
    rvec_list_0_3[cam_idx] = rvec
    tvec_list_0_3[cam_idx] = tvec
    R_list_0_3[cam_idx] = R
    T_list_0_3[cam_idx] = tvec

# 对相机 0-3 进行外参优化（Bundle Adjustment）
def bundle_adjustment_0_3():
    """
    正确的多相机Bundle Adjustment实现（相机0-3）
    
    核心改进：
    1. 优化相机间的相对外参，而非相机相对于棋盘格的外参
    2. 同时优化棋盘格在世界坐标系中的位姿
    3. 建立以相机0为参考的统一世界坐标系
    """
    def objective_function(x):
        """
        目标函数：最小化所有相机所有帧的重投影误差
        
        参数布局：
        - x[0:9]: 相机1-3相对于相机0的旋转向量 (3x3)
        - x[9:18]: 相机1-3相对于相机0的平移向量 (3x3) 
        - x[18:]: 每帧棋盘格在世界坐标系中的位姿 (6*num_frames)
        """
        total_errors = []
        
        # 解包相机外参 (相对于相机0)
        camera_rvecs = {}
        camera_tvecs = {}
        camera_rvecs[0] = np.zeros(3)  # 相机0为参考，外参为0
        camera_tvecs[0] = np.zeros(3)
        
        # 相机1-3的外参
        for i in range(3):
            cam_idx = i + 1
            camera_rvecs[cam_idx] = x[i*3:(i+1)*3]
            camera_tvecs[cam_idx] = x[9+i*3:9+(i+1)*3]
        
        # 解包棋盘格位姿 (在世界坐标系中)
        board_rvecs = []
        board_tvecs = []
        for frame_idx in range(len(objpoints_0_3)):
            start_idx = 18 + frame_idx * 6
            board_rvecs.append(x[start_idx:start_idx+3])
            board_tvecs.append(x[start_idx+3:start_idx+6])
        
        # 计算每个相机每一帧的重投影误差
        for cam_idx in range(4):
            if cam_idx not in imgpoints_0_3 or len(imgpoints_0_3[cam_idx]) == 0:
                continue
                
            cam_rvec = camera_rvecs[cam_idx]
            cam_tvec = camera_tvecs[cam_idx]
            
            for frame_idx in range(len(objpoints_0_3)):
                if frame_idx >= len(imgpoints_0_3[cam_idx]):
                    continue
                
                # 获取棋盘格在世界坐标系中的位姿
                board_rvec = board_rvecs[frame_idx]
                board_tvec = board_tvecs[frame_idx]
                R_board, _ = cv2.Rodrigues(board_rvec)
                
                # 将棋盘格点从棋盘格坐标系转换到世界坐标系
                points_3d_world = []
                for point_3d in objpoints_0_3[frame_idx]:
                    point_world = R_board @ point_3d.reshape(3, 1) + board_tvec.reshape(3, 1)
                    points_3d_world.append(point_world.flatten())
                points_3d_world = np.array(points_3d_world)
                
                # 投影到当前相机
                projected, _ = cv2.projectPoints(
                    points_3d_world,
                    cam_rvec,
                    cam_tvec,
                    camera_matrix_list_0_3[cam_idx],
                    dist_coeff_list_0_3[cam_idx]
                )
                projected = projected.reshape(-1, 2)
                
                # 计算重投影误差
                observed = imgpoints_0_3[cam_idx][frame_idx].reshape(-1, 2)
                error = (observed - projected).ravel()
                total_errors.extend(error)
        
        return np.array(total_errors)

    # 改进的初始化方法
    num_frames = len(objpoints_0_3)
    x0 = []
    
    # 初始化相机1-3相对于相机0的外参
    print("初始化相机间外参...")
    for cam_idx in range(1, 4):
        if len(imgpoints_0_3[0]) > 0 and len(imgpoints_0_3[cam_idx]) > 0:
            # 使用第一帧进行初始外参估计
            try:
                # 计算相机0和当前相机的棋盘格位姿
                ret0, rvec0, tvec0 = cv2.solvePnP(
                    objpoints_0_3[0], imgpoints_0_3[0][0],
                    camera_matrix_list_0_3[0], dist_coeff_list_0_3[0]
                )
                ret_cam, rvec_cam, tvec_cam = cv2.solvePnP(
                    objpoints_0_3[0], imgpoints_0_3[cam_idx][0],
                    camera_matrix_list_0_3[cam_idx], dist_coeff_list_0_3[cam_idx]
                )
                
                if ret0 and ret_cam:
                    # 计算相机间的相对外参
                    R0, _ = cv2.Rodrigues(rvec0)
                    R_cam, _ = cv2.Rodrigues(rvec_cam)
                    
                    # 相机cam相对于相机0的外参
                    R_rel = R_cam @ R0.T
                    t_rel = tvec_cam - R_rel @ tvec0
                    rvec_rel, _ = cv2.Rodrigues(R_rel)
                    
                    x0.extend(rvec_rel.flatten())
                else:
                    x0.extend([0, 0, 0])  # 默认初始化
            except:
                x0.extend([0, 0, 0])  # 默认初始化
        else:
            x0.extend([0, 0, 0])  # 默认初始化
    
    # 初始化相机1-3相对于相机0的平移向量
    for cam_idx in range(1, 4):
        if len(imgpoints_0_3[0]) > 0 and len(imgpoints_0_3[cam_idx]) > 0:
            try:
                ret0, rvec0, tvec0 = cv2.solvePnP(
                    objpoints_0_3[0], imgpoints_0_3[0][0],
                    camera_matrix_list_0_3[0], dist_coeff_list_0_3[0]
                )
                ret_cam, rvec_cam, tvec_cam = cv2.solvePnP(
                    objpoints_0_3[0], imgpoints_0_3[cam_idx][0],
                    camera_matrix_list_0_3[cam_idx], dist_coeff_list_0_3[cam_idx]
                )
                
                if ret0 and ret_cam:
                    R0, _ = cv2.Rodrigues(rvec0)
                    R_cam, _ = cv2.Rodrigues(rvec_cam)
                    R_rel = R_cam @ R0.T
                    t_rel = tvec_cam - R_rel @ tvec0
                    x0.extend(t_rel.flatten())
                else:
                    x0.extend([0, 0, 100])  # 默认初始化，假设相机间距100mm
            except:
                x0.extend([0, 0, 100])  # 默认初始化
        else:
            x0.extend([0, 0, 100])  # 默认初始化
    
    # 初始化每帧棋盘格位姿（使用相机0作为参考）
    print("初始化棋盘格位姿...")
    for frame_idx in range(num_frames):
        if frame_idx < len(imgpoints_0_3[0]):
            try:
                ret, rvec, tvec = cv2.solvePnP(
                    objpoints_0_3[frame_idx],
                    imgpoints_0_3[0][frame_idx],
                    camera_matrix_list_0_3[0],
                    dist_coeff_list_0_3[0]
                )
                if ret:
                    x0.extend(rvec.flatten())
                    x0.extend(tvec.flatten())  
                else:
                    x0.extend([0, 0, 0, 0, 0, 500])  # 默认初始化
            except:
                x0.extend([0, 0, 0, 0, 0, 500])  # 默认初始化
        else:
            x0.extend([0, 0, 0, 0, 0, 500])  # 默认初始化

    x0 = np.array(x0)
    print(f"初始化完成，参数数量: {len(x0)}")

    # 运行优化
    print("开始Bundle Adjustment优化...")
    result = least_squares(
        objective_function,
        x0,
        method='lm',  # Levenberg-Marquardt
        verbose=2,
        max_nfev=1000,
        ftol=1e-8,
        xtol=1e-8
    )

    # 更新相机外参
    for cam_idx in range(1, 4):
        i = cam_idx - 1
        rvec_list_0_3[cam_idx] = result.x[i*3:(i+1)*3].reshape(3, 1)
        tvec_list_0_3[cam_idx] = result.x[9+i*3:9+(i+1)*3].reshape(3, 1)
        R_list_0_3[cam_idx], _ = cv2.Rodrigues(rvec_list_0_3[cam_idx])
        T_list_0_3[cam_idx] = tvec_list_0_3[cam_idx]

    # 计算并打印最终的重投影误差
    print(f"\n优化完成，最终残差: {result.cost}")
    print("相机 0-3 的重投影误差:")
    total_error = 0
    for cam_idx in range(4):
        if cam_idx not in imgpoints_0_3 or len(imgpoints_0_3[cam_idx]) == 0:
            continue
            
        errors = []
        for frame_idx in range(len(objpoints_0_3)):
            if frame_idx >= len(imgpoints_0_3[cam_idx]):
                continue
                
            # 使用优化后的参数计算重投影误差
            if cam_idx == 0:
                rvec = np.zeros((3, 1))
                tvec = np.zeros((3, 1))
            else:
                rvec = rvec_list_0_3[cam_idx]
                tvec = tvec_list_0_3[cam_idx]
                
            # 获取优化后的棋盘格位姿
            start_idx = 18 + frame_idx * 6
            board_rvec = result.x[start_idx:start_idx+3]
            board_tvec = result.x[start_idx+3:start_idx+6]
            R_board, _ = cv2.Rodrigues(board_rvec)
            
            # 转换到世界坐标系
            points_3d_world = []
            for point_3d in objpoints_0_3[frame_idx]:
                point_world = R_board @ point_3d.reshape(3, 1) + board_tvec.reshape(3, 1)
                points_3d_world.append(point_world.flatten())
            points_3d_world = np.array(points_3d_world)
            
            projected, _ = cv2.projectPoints(
                points_3d_world, rvec, tvec,
                camera_matrix_list_0_3[cam_idx],
                dist_coeff_list_0_3[cam_idx]
            )
            projected = projected.reshape(-1, 2)
            points_2d = imgpoints_0_3[cam_idx][frame_idx].reshape(-1, 2)
            error = np.linalg.norm(points_2d - projected, axis=1)
            errors.append(np.mean(error))
            
        if errors:
            mean_error = np.mean(errors)
            total_error += mean_error
            print(f"相机 {cam_idx} 的平均重投影误差: {mean_error:.3f} 像素")

    active_cameras = len([i for i in range(4) if i in imgpoints_0_3 and len(imgpoints_0_3[i]) > 0])
    if active_cameras > 0:
        print(f"相机 0-3 的平均重投影误差: {total_error/active_cameras:.3f} 像素")

bundle_adjustment_0_3()

# 保存结果
np.savez(
    "multi_camera_calibration_0_3.npz",
    camera_matrix_list=camera_matrix_list_0_3,
    dist_coeff_list=dist_coeff_list_0_3,
    R_list=R_list_0_3,
    T_list=T_list_0_3,
    rvec_list=rvec_list_0_3,
    tvec_list=tvec_list_0_3
)
print("相机 0-3 的外参优化完成并保存")


#-------
# ---------------------------- 3. 相机 4-7 的校准 ---------------------------- #

# 创建存储对象点和图像点的列表
objpoints_4_7 = []  # 3D 点
imgpoints_4_7 = {i: [] for i in range(4, 8)}  # 每个相机的 2D 点

# 加载相机 4-7 的图像
image_paths_4_7 = {}
for cam_idx in range(4, 8):
    image_dir = f"D:/CalibrationImages/Cam{cam_idx}"
    if not os.path.exists(image_dir):
        print(f"相机 {cam_idx} 的图像目录不存在：{image_dir}")
        exit(1)
    image_paths_4_7[cam_idx] = glob.glob(os.path.join(image_dir, "*.jpeg"))
    image_paths_4_7[cam_idx].sort()
    if len(image_paths_4_7[cam_idx]) == 0:
        print(f"相机 {cam_idx} 的图像目录中没有找到 jpeg 图像")
        exit(1)

# 确保所有相机的图像数量一致
num_images_4_7 = len(image_paths_4_7[4])
for cam_idx in range(5, 8):
    if len(image_paths_4_7[cam_idx]) != num_images_4_7:
        print(f"相机 {cam_idx} 的图像数量与相机 4 不一致")
        exit(1)

# 遍历每一组图像
for i in range(num_images_4_7):
    imgs = {}
    ret_corners = {}
    corners = {}

    # 读取所有相机的图像
    for cam_idx in range(4, 8):
        img_path = image_paths_4_7[cam_idx][i]
        imgs[cam_idx] = cv2.imread(img_path)
        if imgs[cam_idx] is None:
            print(f"无法读取相机 {cam_idx} 的图像：{img_path}")
            continue
        gray = cv2.cvtColor(imgs[cam_idx], cv2.COLOR_BGR2GRAY)
        ret_corners[cam_idx], corners[cam_idx] = cv2.findChessboardCorners(
            gray, (nx, ny), None
        )

    # 检查所有相机是否都找到了角点
    if all(ret_corners.values()):
        # 精细化角点并收集数据
        objpoints_4_7.append(objp)
        for cam_idx in range(4, 8):
            gray = cv2.cvtColor(imgs[cam_idx], cv2.COLOR_BGR2GRAY)
            corners2 = cv2.cornerSubPix(
                gray, corners[cam_idx], (11, 11), (-1, -1), criteria
            )
            imgpoints_4_7[cam_idx].append(corners2)
    else:
        print(f"在图像索引 {i} 处并非所有相机都检测到角点，跳过此组")

# 对每个相机进行单独校准
calibration_results_4_7 = {}
for cam_idx in range(4, 8):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_4_7, imgpoints_4_7[cam_idx], image_size, None, None
    )
    calibration_results_4_7[cam_idx] = {
        "ret": ret,
        "mtx": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
    }
    print(f"相机 {cam_idx} 校准完成。RMS 误差：{ret}")

# 保存校准结果
np.savez(
    "single_camera_calibration_4_7.npz",
    calibration_results=calibration_results_4_7,
    image_size=image_size,
)
print("相机 4-7 的校准结果已保存到 single_camera_calibration_4_7.npz")
# ---------------------------- 4. 相机 4-7 的外参估计和优化 ---------------------------- #

# 加载单相机校准结果
calib_data_4_7 = np.load("single_camera_calibration_4_7.npz", allow_pickle=True)
calibration_results_4_7 = calib_data_4_7["calibration_results"].item()

# 初始化相机参数列表
camera_matrix_list_4_7 = [calibration_results_4_7[i]["mtx"] for i in range(4, 8)]
dist_coeff_list_4_7 = [calibration_results_4_7[i]["dist"] for i in range(4, 8)]
R_list_4_7 = [np.eye(3) for _ in range(4)]  # 初始旋转矩阵
T_list_4_7 = [np.zeros((3, 1)) for _ in range(4)]  # 初始平移向量
rvec_list_4_7 = [np.zeros((3, 1)) for _ in range(4)]  # 初始旋转向量
tvec_list_4_7 = [np.zeros((3, 1)) for _ in range(4)]  # 初始平移向量

# 计算每个相机的初始外参（相对于相机 7）
for idx, cam_idx in enumerate(range(4, 7)):
    ret, rvec, tvec = cv2.solvePnP(
        objpoints_4_7[0],
        imgpoints_4_7[cam_idx][0],
        camera_matrix_list_4_7[idx],
        dist_coeff_list_4_7[idx]
    )
    R, _ = cv2.Rodrigues(rvec)
    rvec_list_4_7[idx] = rvec
    tvec_list_4_7[idx] = tvec
    R_list_4_7[idx] = R
    T_list_4_7[idx] = tvec

# 对相机 4-7 进行外参优化（Bundle Adjustment），以相机 7 为参考
def bundle_adjustment_4_7():
    """
    正确的多相机Bundle Adjustment实现（相机4-7）
    
    核心改进：
    1. 优化相机间的相对外参，而非相机相对于棋盘格的外参
    2. 同时优化棋盘格在世界坐标系中的位姿
    3. 建立以相机7为参考的统一世界坐标系
    """
    def objective_function(x):
        """
        目标函数：最小化所有相机所有帧的重投影误差
        
        参数布局：
        - x[0:9]: 相机4-6相对于相机7的旋转向量 (3x3)
        - x[9:18]: 相机4-6相对于相机7的平移向量 (3x3)
        - x[18:]: 每帧棋盘格在世界坐标系中的位姿 (6*num_frames)
        """
        total_errors = []
        
        # 解包相机外参 (相对于相机7)
        camera_rvecs = {}
        camera_tvecs = {}
        camera_rvecs[7] = np.zeros(3)  # 相机7为参考，外参为0
        camera_tvecs[7] = np.zeros(3)
        
        # 相机4-6的外参
        for i in range(3):
            cam_idx = i + 4
            camera_rvecs[cam_idx] = x[i*3:(i+1)*3]
            camera_tvecs[cam_idx] = x[9+i*3:9+(i+1)*3]
        
        # 解包棋盘格位姿 (在世界坐标系中)
        board_rvecs = []
        board_tvecs = []
        for frame_idx in range(len(objpoints_4_7)):
            start_idx = 18 + frame_idx * 6
            board_rvecs.append(x[start_idx:start_idx+3])
            board_tvecs.append(x[start_idx+3:start_idx+6])
        
        # 计算每个相机每一帧的重投影误差
        for i, cam_idx in enumerate(range(4, 7)):
            if cam_idx not in imgpoints_4_7 or len(imgpoints_4_7[cam_idx]) == 0:
                continue
                
            cam_rvec = camera_rvecs[cam_idx]
            cam_tvec = camera_tvecs[cam_idx]
            
            for frame_idx in range(len(objpoints_4_7)):
                if frame_idx >= len(imgpoints_4_7[cam_idx]):
                    continue
                
                # 获取棋盘格在世界坐标系中的位姿
                board_rvec = board_rvecs[frame_idx]
                board_tvec = board_tvecs[frame_idx]
                R_board, _ = cv2.Rodrigues(board_rvec)
                
                # 将棋盘格点从棋盘格坐标系转换到世界坐标系
                points_3d_world = []
                for point_3d in objpoints_4_7[frame_idx]:
                    point_world = R_board @ point_3d.reshape(3, 1) + board_tvec.reshape(3, 1)
                    points_3d_world.append(point_world.flatten())
                points_3d_world = np.array(points_3d_world)
                
                # 投影到当前相机
                projected, _ = cv2.projectPoints(
                    points_3d_world,
                    cam_rvec,
                    cam_tvec,
                    camera_matrix_list_4_7[i],
                    dist_coeff_list_4_7[i]
                )
                projected = projected.reshape(-1, 2)
                
                # 计算重投影误差
                observed = imgpoints_4_7[cam_idx][frame_idx].reshape(-1, 2)
                error = (observed - projected).ravel()
                total_errors.extend(error)
        
        return np.array(total_errors)

    # 改进的初始化方法
    num_frames = len(objpoints_4_7)
    x0 = []
    
    # 初始化相机4-6相对于相机7的外参
    print("初始化相机间外参...")
    for cam_idx in range(4, 7):
        if len(imgpoints_4_7[7]) > 0 and len(imgpoints_4_7[cam_idx]) > 0:
            # 使用第一帧进行初始外参估计
            try:
                # 计算相机7和当前相机的棋盘格位姿
                ret7, rvec7, tvec7 = cv2.solvePnP(
                    objpoints_4_7[0], imgpoints_4_7[7][0],
                    camera_matrix_list_4_7[3], dist_coeff_list_4_7[3]  # 相机7在索引3
                )
                ret_cam, rvec_cam, tvec_cam = cv2.solvePnP(
                    objpoints_4_7[0], imgpoints_4_7[cam_idx][0],
                    camera_matrix_list_4_7[cam_idx-4], dist_coeff_list_4_7[cam_idx-4]
                )
                
                if ret7 and ret_cam:
                    # 计算相机间的相对外参
                    R7, _ = cv2.Rodrigues(rvec7)
                    R_cam, _ = cv2.Rodrigues(rvec_cam)
                    
                    # 相机cam相对于相机7的外参
                    R_rel = R_cam @ R7.T
                    t_rel = tvec_cam - R_rel @ tvec7
                    rvec_rel, _ = cv2.Rodrigues(R_rel)
                    
                    x0.extend(rvec_rel.flatten())
                else:
                    x0.extend([0, 0, 0])  # 默认初始化
            except:
                x0.extend([0, 0, 0])  # 默认初始化
        else:
            x0.extend([0, 0, 0])  # 默认初始化
    
    # 初始化相机4-6相对于相机7的平移向量
    for cam_idx in range(4, 7):
        if len(imgpoints_4_7[7]) > 0 and len(imgpoints_4_7[cam_idx]) > 0:
            try:
                ret7, rvec7, tvec7 = cv2.solvePnP(
                    objpoints_4_7[0], imgpoints_4_7[7][0],
                    camera_matrix_list_4_7[3], dist_coeff_list_4_7[3]
                )
                ret_cam, rvec_cam, tvec_cam = cv2.solvePnP(
                    objpoints_4_7[0], imgpoints_4_7[cam_idx][0],
                    camera_matrix_list_4_7[cam_idx-4], dist_coeff_list_4_7[cam_idx-4]
                )
                
                if ret7 and ret_cam:
                    R7, _ = cv2.Rodrigues(rvec7)
                    R_cam, _ = cv2.Rodrigues(rvec_cam)
                    R_rel = R_cam @ R7.T
                    t_rel = tvec_cam - R_rel @ tvec7
                    x0.extend(t_rel.flatten())
                else:
                    x0.extend([0, 0, 100])  # 默认初始化，假设相机间距100mm
            except:
                x0.extend([0, 0, 100])  # 默认初始化
        else:
            x0.extend([0, 0, 100])  # 默认初始化
    
    # 初始化每帧棋盘格位姿（使用相机7作为参考）
    print("初始化棋盘格位姿...")
    for frame_idx in range(num_frames):
        if frame_idx < len(imgpoints_4_7[7]):
            try:
                ret, rvec, tvec = cv2.solvePnP(
                    objpoints_4_7[frame_idx],
                    imgpoints_4_7[7][frame_idx],
                    camera_matrix_list_4_7[3],  # 相机7在索引3
                    dist_coeff_list_4_7[3]
                )
                if ret:
                    x0.extend(rvec.flatten())
                    x0.extend(tvec.flatten())
                else:
                    x0.extend([0, 0, 0, 0, 0, 500])  # 默认初始化
            except:
                x0.extend([0, 0, 0, 0, 0, 500])  # 默认初始化
        else:
            x0.extend([0, 0, 0, 0, 0, 500])  # 默认初始化

    x0 = np.array(x0)
    print(f"初始化完成，参数数量: {len(x0)}")

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

    # 更新相机外参
    for idx, cam_idx in enumerate(range(4, 7)):
        rvec_list_4_7[idx] = result.x[idx*3:(idx+1)*3].reshape(3, 1)
        tvec_list_4_7[idx] = result.x[9+idx*3:9+(idx+1)*3].reshape(3, 1)
        R_list_4_7[idx], _ = cv2.Rodrigues(rvec_list_4_7[idx])
        T_list_4_7[idx] = tvec_list_4_7[idx]

    # 相机 7（索引 3）的外参保持为初始值，即 R = I, T = 0

    # 计算并打印最终的重投影误差
    print(f"\n优化完成，最终残差: {result.cost}")
    print("相机 4-7 的重投影误差:")
    total_error = 0
    for i, cam_idx in enumerate(range(4, 8)):
        if cam_idx not in imgpoints_4_7 or len(imgpoints_4_7[cam_idx]) == 0:
            continue
            
        errors = []
        for frame_idx in range(len(objpoints_4_7)):
            if frame_idx >= len(imgpoints_4_7[cam_idx]):
                continue
                
            # 使用优化后的参数计算重投影误差
            if cam_idx == 7:
                rvec = np.zeros((3, 1))
                tvec = np.zeros((3, 1))
            else:
                rvec = rvec_list_4_7[cam_idx-4]  # 相机4-6对应索引0-2
                tvec = tvec_list_4_7[cam_idx-4]
                
            # 获取优化后的棋盘格位姿
            start_idx = 18 + frame_idx * 6
            board_rvec = result.x[start_idx:start_idx+3]
            board_tvec = result.x[start_idx+3:start_idx+6]
            R_board, _ = cv2.Rodrigues(board_rvec)
            
            # 转换到世界坐标系
            points_3d_world = []
            for point_3d in objpoints_4_7[frame_idx]:
                point_world = R_board @ point_3d.reshape(3, 1) + board_tvec.reshape(3, 1)
                points_3d_world.append(point_world.flatten())
            points_3d_world = np.array(points_3d_world)
            
            projected, _ = cv2.projectPoints(
                points_3d_world, rvec, tvec,
                camera_matrix_list_4_7[i],
                dist_coeff_list_4_7[i]
            )
            projected = projected.reshape(-1, 2)
            points_2d = imgpoints_4_7[cam_idx][frame_idx].reshape(-1, 2)
            error = np.linalg.norm(points_2d - projected, axis=1)
            errors.append(np.mean(error))
            
        if errors:
            mean_error = np.mean(errors)
            total_error += mean_error
            print(f"相机 {cam_idx} 的平均重投影误差: {mean_error:.3f} 像素")

    active_cameras = len([i for i in range(4, 8) if i in imgpoints_4_7 and len(imgpoints_4_7[i]) > 0])
    if active_cameras > 0:
        print(f"相机 4-7 的平均重投影误差: {total_error/active_cameras:.3f} 像素")

bundle_adjustment_4_7()

# 保存结果
np.savez(
    "multi_camera_calibration_4_7.npz",
    camera_matrix_list=camera_matrix_list_4_7,
    dist_coeff_list=dist_coeff_list_4_7,
    R_list=R_list_4_7,
    T_list=T_list_4_7,
    rvec_list=rvec_list_4_7,
    tvec_list=tvec_list_4_7
)
print("相机 4-7 的外参优化完成并保存")

# ---------------------------- 5. 相机 0 和相机 7 的双目标定 ---------------------------- #

# 准备相机 0 和相机 7 的共同的棋盘格角点数据
objpoints_0_7 = []
imgpoints_0 = []
imgpoints_7 = []

# 加载相机 0 和相机 7 的图像
image_paths_0 = glob.glob(os.path.join(f"D:/CalibrationImages/Cam0", "*.jpeg"))
image_paths_7 = glob.glob(os.path.join(f"D:/CalibrationImages/Cam7", "*.jpeg"))
image_paths_0.sort()
image_paths_7.sort()

# 假设有一些图像，棋盘格同时被相机 0 和相机 7 看到
num_images_0_7 = min(len(image_paths_0), len(image_paths_7))

for i in range(num_images_0_7):
    img0 = cv2.imread(image_paths_0[i])
    img7 = cv2.imread(image_paths_7[i])
    if img0 is None or img7 is None:
        continue
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
    ret0, corners0 = cv2.findChessboardCorners(gray0, (nx, ny), None)
    ret7, corners7 = cv2.findChessboardCorners(gray7, (nx, ny), None)
    if ret0 and ret7:
        objpoints_0_7.append(objp)
        corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
        corners7 = cv2.cornerSubPix(gray7, corners7, (11, 11), (-1, -1), criteria)
        imgpoints_0.append(corners0)
        imgpoints_7.append(corners7)
    else:
        print(f"在索引 {i} 处未能同时检测到角点，跳过")

# 进行双目相机校准
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints_0_7,
    imgpoints_0,
    imgpoints_7,
    calibration_results_0_3[0]["mtx"],
    calibration_results_0_3[0]["dist"],
    calibration_results_4_7[7]["mtx"],
    calibration_results_4_7[7]["dist"],
    image_size,
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("相机 0 和相机 7 的双目标定完成")

# 将相机 7 的外参转换到相机 0 的坐标系下
R_0_7 = R
T_0_7 = T

# 更新相机 7 的外参
R_cam7_in_cam0 = R_0_7
T_cam7_in_cam0 = T_0_7

# 将相机 4-7 的外参转换到相机 0 的坐标系
for idx in range(4):  # idx 0-3 对应相机 4-7
    R_cam_in_cam7 = R_list_4_7[idx]
    T_cam_in_cam7 = T_list_4_7[idx]
    # 相机在相机 0 坐标系下的外参
    R_cam_in_cam0 = R_cam7_in_cam0 @ R_cam_in_cam7
    T_cam_in_cam0 = R_cam7_in_cam0 @ T_cam_in_cam7 + T_cam7_in_cam0
    R_list_4_7[idx] = R_cam_in_cam0
    T_list_4_7[idx] = T_cam_in_cam0

# 合并所有相机的内参和外参
camera_matrix_list = camera_matrix_list_0_3 + camera_matrix_list_4_7
dist_coeff_list = dist_coeff_list_0_3 + dist_coeff_list_4_7
R_list = R_list_0_3 + R_list_4_7
T_list = T_list_0_3 + T_list_4_7

# 保存最终结果
np.savez(
    "multi_camera_calibration_all.npz",
    camera_matrix_list=camera_matrix_list,
    dist_coeff_list=dist_coeff_list,
    R_list=R_list,
    T_list=T_list
)

print("所有相机的外参已转换到相机 0 的坐标系并保存")

