# ---------------------------- 1. 单独相机校正 ---------------------------- #

import cv2
import numpy as np
import glob
import os

# 棋盘格参数
nx = 9  # 水平方向的角点数
ny = 6  # 垂直方向的角点数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# 准备对象点，例如 (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# 创建存储对象点和图像点的列表
objpoints = []  # 3D点
imgpoints = {i: [] for i in range(8)}  # 每个相机的2D点

# 加载8个相机的图像
image_paths = {}
for cam_idx in range(8):
    # 假设每个相机的像存放在 D:/CalibrationImages/Cam0, Cam1, ..., Cam7 文件夹中
    image_dir = f"D:/CalibrationImages/Cam{cam_idx}"  # 请根据您的实际路径修改
    if not os.path.exists(image_dir):
        print(f"相机 {cam_idx} 的图像目录不存在：{image_dir}")
        exit(1)
    image_paths[cam_idx] = glob.glob(os.path.join(image_dir, "*.jpeg"))
    image_paths[cam_idx].sort()  # 确保顺序一致
    if len(image_paths[cam_idx]) == 0:
        print(f"相机 {cam_idx} 的图像目录中没有找到jpeg图像")
        exit(1)

# 对每个相机进行单独校准
calibration_results = {}
for cam_idx in range(8):
    objpoints = []
    imgpoints_single = []
    for img_path in image_paths[cam_idx]:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取相机 {cam_idx} 的图像：{img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints_single.append(corners2)
        else:
            print(f"相机 {cam_idx } 的图像 {img_path} 未检测到角点，跳过此张图片")
    if len(objpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints_single, gray.shape[::-1], None, None
        )
        calibration_results[cam_idx] = {
            "ret": ret,
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
        }
        print(f"相机 {cam_idx} 校准完成。RMS 误差：{ret}")
    else:
        print(f"相机 {cam_idx} 没有足够的有效图像进行校准")

# 保存校准结果
np.savez(
    "single_camera_calibration_8cams.npz",
    calibration_results=calibration_results,
)
print("校准结果已保存到 single_camera_calibration_8cams.npz")
