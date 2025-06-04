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
    """优化相机 0-3 的外部参数"""
    def objective_function(x):
        total_errors = []
        # 解包参数
        rvecs = x[:9].reshape(3, 3)  # 3 个相机的旋转向量（相机 1-3）
        tvecs = x[9:].reshape(3, 3)  # 3 个相机的平移向量（相机 1-3）
        for cam_idx in range(1, 4):
            rvec = rvecs[cam_idx - 1].reshape(3, 1)
            tvec = tvecs[cam_idx - 1].reshape(3, 1)
            for frame_idx in range(len(objpoints_0_3)):
                points_2d = imgpoints_0_3[cam_idx][frame_idx].reshape(-1, 2)
                projected, _ = cv2.projectPoints(
                    objpoints_0_3[frame_idx],
                    rvec,
                    tvec,
                    camera_matrix_list_0_3[cam_idx],
                    dist_coeff_list_0_3[cam_idx]
                )
                projected = projected.reshape(-1, 2)
                error = (points_2d - projected).ravel()
                total_errors.extend(error)
        return np.array(total_errors)

    # 初始参数
    x0 = np.zeros(18)  # 3 * (3 + 3)
    for cam_idx in range(1, 4):
        idx = (cam_idx - 1) * 3
        x0[idx:idx+3] = rvec_list_0_3[cam_idx].ravel()
        x0[9+idx:9+idx+3] = tvec_list_0_3[cam_idx].ravel()

    # 运行优化
    result = least_squares(
        objective_function,
        x0,
        method='lm',  # Levenberg-Marquardt
        verbose=2,
        max_nfev=1000
    )

    # 更新参数
    for cam_idx in range(1, 4):
        idx = (cam_idx - 1) * 3
        rvec_list_0_3[cam_idx] = result.x[idx:idx+3].reshape(3, 1)
        tvec_list_0_3[cam_idx] = result.x[9+idx:9+idx+3].reshape(3, 1)
        R_list_0_3[cam_idx], _ = cv2.Rodrigues(rvec_list_0_3[cam_idx])
        T_list_0_3[cam_idx] = tvec_list_0_3[cam_idx]

    # 计算并打印最终的重投影误差
    print("\n相机 0-3 的重投影误差:")
    total_error = 0
    for cam_idx in range(1, 4):
        errors = []
        for frame_idx in range(len(objpoints_0_3)):
            projected, _ = cv2.projectPoints(
                objpoints_0_3[frame_idx],
                rvec_list_0_3[cam_idx],
                tvec_list_0_3[cam_idx],
                camera_matrix_list_0_3[cam_idx],
                dist_coeff_list_0_3[cam_idx]
            )
            projected = projected.reshape(-1, 2)
            points_2d = imgpoints_0_3[cam_idx][frame_idx].reshape(-1, 2)
            error = np.linalg.norm(points_2d - projected, axis=1)
            errors.append(np.mean(error))
        mean_error = np.mean(errors)
        total_error += mean_error
        print(f"相机 {cam_idx} 的平均重投影误差: {mean_error:.3f} 像素")

    print(f"相机 0-3 的平均重投影误差: {total_error/3:.3f} 像素")

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
    """优化相机 4-7 的外部参数"""
    def objective_function(x):
        total_errors = []
        # 解包参数
        rvecs = x[:9].reshape(3, 3)  # 3 个相机的旋转向量（相机 4-6）
        tvecs = x[9:].reshape(3, 3)  # 3 个相机的平移向量（相机 4-6）
        for idx, cam_idx in enumerate(range(4, 7)):
            rvec = rvecs[idx].reshape(3, 1)
            tvec = tvecs[idx].reshape(3, 1)
            for frame_idx in range(len(objpoints_4_7)):
                points_2d = imgpoints_4_7[cam_idx][frame_idx].reshape(-1, 2)
                projected, _ = cv2.projectPoints(
                    objpoints_4_7[frame_idx],
                    rvec,
                    tvec,
                    camera_matrix_list_4_7[idx],
                    dist_coeff_list_4_7[idx]
                )
                projected = projected.reshape(-1, 2)
                error = (points_2d - projected).ravel()
                total_errors.extend(error)
        return np.array(total_errors)

    # 初始参数
    x0 = np.zeros(18)  # 3 * (3 + 3)
    for idx in range(3):
        x0[idx*3:idx*3+3] = rvec_list_4_7[idx].ravel()
        x0[9+idx*3:9+idx*3+3] = tvec_list_4_7[idx].ravel()

    # 运行优化
    result = least_squares(
        objective_function,
        x0,
        method='lm',
        verbose=2,
        max_nfev=1000
    )

    # 更新参数
    for idx in range(3):
        rvec_list_4_7[idx] = result.x[idx*3:idx*3+3].reshape(3, 1)
        tvec_list_4_7[idx] = result.x[9+idx*3:9+idx*3+3].reshape(3, 1)
        R_list_4_7[idx], _ = cv2.Rodrigues(rvec_list_4_7[idx])
        T_list_4_7[idx] = tvec_list_4_7[idx]

    # 相机 7（索引 3）的外参保持为初始值，即 R = I, T = 0

    # 计算并打印最终的重投影误差
    print("\n相机 4-7 的重投影误差:")
    total_error = 0
    for idx, cam_idx in enumerate(range(4, 7)):
        errors = []
        for frame_idx in range(len(objpoints_4_7)):
            projected, _ = cv2.projectPoints(
                objpoints_4_7[frame_idx],
                rvec_list_4_7[idx],
                tvec_list_4_7[idx],
                camera_matrix_list_4_7[idx],
                dist_coeff_list_4_7[idx]
            )
            projected = projected.reshape(-1, 2)
            points_2d = imgpoints_4_7[cam_idx][frame_idx].reshape(-1, 2)
            error = np.linalg.norm(points_2d - projected, axis=1)
            errors.append(np.mean(error))
        mean_error = np.mean(errors)
        total_error += mean_error
        print(f"相机 {cam_idx} 的平均重投影误差: {mean_error:.3f} 像素")

    print(f"相机 4-7 的平均重投影误差: {total_error/3:.3f} 像素")

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

