import cv2
import numpy as np
import glob
from scipy.optimize import least_squares

# Number of inner corners in the chessboard
nx = 9  # Number of inner corners in x
ny = 6  # Number of inner corners in y

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D point in real world space
imgpoints_dict = {}  # Dictionary to hold image points for each camera

# List of camera names
cameras = ['Left', 'Left1', 'Left2', 'Center', 'Right', 'Right1', 'Right2', 'Right3']

# Initialize image points dictionary
for cam in cameras:
    imgpoints_dict[cam] = []

# Load images for each camera
images_dict = {}
for cam in cameras:
    if cam == 'Left':
        images = glob.glob(r"D:\20241112\CAM_LEFT\Cam2-*.jpeg")
    elif cam == 'Left1':
        images = glob.glob(r"D:\20241112\CAM_LEFT1\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Left2':
        images = glob.glob(r"D:\20241112\CAM_LEFT2\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Center':
        images = glob.glob(r"D:\20241112\CAM_MIDDLE\Cam1-*.jpeg")
    elif cam == 'Right':
        images = glob.glob(r"D:\20241112\CAM_RIGHT\Cam0-*.jpeg")
    elif cam == 'Right1':
        images = glob.glob(r"D:\20241112\CAM_RIGHT1\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Right2':
        images = glob.glob(r"D:\20241112\CAM_RIGHT2\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Right3':
        images = glob.glob(r"D:\20241112\CAM_RIGHT3\CamX-*.jpeg")  # Update with actual path
    else:
        continue  # Unknown camera

    images.sort()
    images_dict[cam] = images

# Ensure that all cameras have the same number of images
num_images = len(images_dict['Left'])
for cam in cameras:
    if len(images_dict[cam]) != num_images:
        print(f"Camera {cam} has a different number of images.")
        # Handle this case appropriately (e.g., skip images or raise an error)

# Find chessboard corners for each image
for i in range(num_images):
    corners_found = True
    current_imgpoints = {}
    for cam in cameras:
        img_path = images_dict[cam][i]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}")
            corners_found = False
            break

        # Find corners in the chessboard pattern
        ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
        if not ret:
            print(f"Chessboard corners not found in camera {cam}, image {i}")
            corners_found = False
            break

        # Refine corner locations
        corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                   corners, (11, 11), (-1, -1), criteria)
        current_imgpoints[cam] = corners

    if corners_found:
        # Add object points and image points for all cameras
        objpoints.append(objp)
        for cam in cameras:
            imgpoints_dict[cam].append(current_imgpoints[cam])

# Perform camera calibration for each camera individually
mtx_dict = {}
dist_dict = {}
rvecs_dict = {}
tvecs_dict = {}
for cam in cameras:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints_dict[cam], (1920, 1200), None, None)
    mtx_dict[cam] = mtx
    dist_dict[cam] = dist
    rvecs_dict[cam] = rvecs
    tvecs_dict[cam] = tvecs
    print(f"Camera {cam} calibration completed.")

# Save single camera calibration results
np.savez('single_camera_calibration.npz', mtx_dict=mtx_dict, dist_dict=dist_dict,
         rvecs_dict=rvecs_dict, tvecs_dict=tvecs_dict, image_size=(1920, 1200))

print("Single camera calibration results saved to single_camera_calibration.npz")

# Calculate and display the field of view (FOV) for each camera
def calculate_fov(mtx, image_size):
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    fov_x = 2 * np.arctan(image_size[0] / (2 * fx)) * (180 / np.pi)
    fov_y = 2 * np.arctan(image_size[1] / (2 * fy)) * (180 / np.pi)
    return fov_x, fov_y

for cam in cameras:
    fov_x, fov_y = calculate_fov(mtx_dict[cam], (1920, 1200))
    print(f"Camera {cam} horizontal FOV (FOV_x): {fov_x} degrees")
    print(f"Camera {cam} vertical FOV (FOV_y): {fov_y} degrees")

# ---------------------------- External Parameters Calibration (Fixed Intrinsics) ---------------------------- #

# Chessboard parameters
pattern_size = (9, 6)
square_size = 3.0  # Adjust according to actual square size in mm

# Prepare object points
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0],
                       0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Image size
image_size = (1920, 1200)

# Re-initialize the image points dictionary
imgpoints_dict = {}
for cam in cameras:
    imgpoints_dict[cam] = []

# Reload images for each camera
images_dict = {}
for cam in cameras:
    if cam == 'Left':
        images = glob.glob(r"D:\20241112\CAM_LEFT\Cam2-*.jpeg")
    elif cam == 'Left1':
        images = glob.glob(r"D:\20241112\CAM_LEFT1\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Left2':
        images = glob.glob(r"D:\20241112\CAM_LEFT2\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Center':
        images = glob.glob(r"D:\20241112\CAM_MIDDLE\Cam1-*.jpeg")
    elif cam == 'Right':
        images = glob.glob(r"D:\20241112\CAM_RIGHT\Cam0-*.jpeg")
    elif cam == 'Right1':
        images = glob.glob(r"D:\20241112\CAM_RIGHT1\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Right2':
        images = glob.glob(r"D:\20241112\CAM_RIGHT2\CamX-*.jpeg")  # Update with actual path
    elif cam == 'Right3':
        images = glob.glob(r"D:\20241112\CAM_RIGHT3\CamX-*.jpeg")  # Update with actual path
    else:
        continue

    images.sort()
    images_dict[cam] = images

# Ensure all cameras have the same number of images
num_images = len(images_dict['Left'])
for cam in cameras:
    if len(images_dict[cam]) != num_images:
        print(f"Camera {cam} has a different number of images.")
        # Handle this appropriately

# Detect chessboard corners and collect data
objpoints = []
for i in range(num_images):
    corners_found = True
    current_imgpoints = {}
    for cam in cameras:
        img_path = images_dict[cam][i]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image {img_path}")
            corners_found = False
            break

        # Convert to grayscale and detect corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f"Chessboard corners not found in camera {cam}, image {i}")
            corners_found = False
            break

        # Refine corners
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints_dict[cam].append(corners)

    if corners_found:
        objpoints.append(objp)
    else:
        print(f"Skipping image set {i} due to missing corners.")

# Load single camera calibration results
calib_data = np.load('single_camera_calibration.npz', allow_pickle=True)
mtx_dict = calib_data['mtx_dict'].item()
dist_dict = calib_data['dist_dict'].item()

# Prepare fixed intrinsic and distortion parameters list
mtx_list = []
dist_list = []
for cam in cameras:
    mtx_list.append(mtx_dict[cam])
    dist_list.append(dist_dict[cam])

# Initialize external parameters list
rvecs_dict = {}
tvecs_dict = {}
for cam in cameras:
    rvecs_dict[cam] = []
    tvecs_dict[cam] = []

n_images = len(objpoints)

# Compute initial external parameters using solvePnP
for i in range(n_images):
    for idx, cam in enumerate(cameras):
        ret, rvec, tvec = cv2.solvePnP(
            objpoints[i], imgpoints_dict[cam][i], mtx_dict[cam], dist_dict[cam])
        rvecs_dict[cam].append(rvec)
        tvecs_dict[cam].append(tvec)

# ---------------------------- Global Optimization ---------------------------- #

def pack_params(rvecs_dict, tvecs_dict, cameras, n_images):
    params = []
    for cam in cameras:
        for i in range(n_images):
            rvec = rvecs_dict[cam][i].ravel()
            tvec = tvecs_dict[cam][i].ravel()
            params.extend(rvec)
            params.extend(tvec)
    return np.array(params)

def reprojection_error(params, n_cameras, n_points, n_images,
                       camera_indices, image_indices, point_indices,
                       points_2d, objpoints,
                       mtx_list, dist_list):
    error = []
    idx = 0
    rvecs = []
    tvecs = []

    total_poses = n_cameras * n_images
    for _ in range(total_poses):
        rvec = params[idx:idx+3].reshape(3, 1)
        tvec = params[idx+3:idx+6].reshape(3, 1)
        rvecs.append(rvec)
        tvecs.append(tvec)
        idx += 6

    for i in range(len(points_2d)):
        camera_idx = camera_indices[i]
        image_idx = image_indices[i]
        point_idx = point_indices[i]

        mtx = mtx_list[camera_idx]
        dist_coeffs = dist_list[camera_idx]
        pose_idx = camera_idx * n_images + image_idx
        rvec = rvecs[pose_idx]
        tvec = tvecs[pose_idx]

        objp = objpoints[image_idx][point_idx]
        imgp = points_2d[i]

        imgp_proj, _ = cv2.projectPoints(
            objp.reshape(1, 3), rvec, tvec, mtx, dist_coeffs)

        error.append(imgp.ravel() - imgp_proj.ravel())

    return np.concatenate(error)

# Prepare data for optimization
n_cameras = len(cameras)
n_points = len(objp)
camera_indices = []
image_indices = []
point_indices = []
points_2d = []

for i in range(n_images):
    for j in range(n_points):
        for idx, cam in enumerate(cameras):
            camera_indices.append(idx)
            image_indices.append(i)
            point_indices.append(j)
            points_2d.append(imgpoints_dict[cam][i][j])

# Initialize optimization parameters
x0 = pack_params(rvecs_dict, tvecs_dict, cameras, n_images)

# Perform global optimization
res = least_squares(
    reprojection_error,
    x0,
    verbose=2,
    method='trf',
    loss='huber',
    args=(n_cameras, n_points, n_images,
          camera_indices, image_indices, point_indices,
          points_2d, objpoints,
          mtx_list, dist_list)
)

# Extract optimized parameters
def unpack_params(params, n_cameras, n_images):
    idx = 0
    rvecs_dict = {}
    tvecs_dict = {}
    for cam_idx in range(n_cameras):
        cam_rvecs = []
        cam_tvecs = []
        for _ in range(n_images):
            rvec = params[idx:idx+3].reshape(3, 1)
            idx += 3
            tvec = params[idx:idx+3].reshape(3, 1)
            idx += 3
            cam_rvecs.append(rvec)
            cam_tvecs.append(tvec)
        cam_name = cameras[cam_idx]
        rvecs_dict[cam_name] = cam_rvecs
        tvecs_dict[cam_name] = cam_tvecs
    return rvecs_dict, tvecs_dict

rvecs_opt_dict, tvecs_opt_dict = unpack_params(res.x, n_cameras, n_images)

# The internal parameters and distortion coefficients remain unchanged
mtx_opt_dict = mtx_dict
dist_opt_dict = dist_dict

# Build external parameters for each camera (using the first image)
R_dict = {}
T_dict = {}
for cam in cameras:
    rvec = rvecs_opt_dict[cam][0]
    tvec = tvecs_opt_dict[cam][0]
    R, _ = cv2.Rodrigues(rvec)
    T = tvec
    R_dict[cam] = R
    T_dict[cam] = T

# Choose a reference camera, for example 'Center'
reference_cam = 'Center'
R_ref = R_dict[reference_cam]
T_ref = T_dict[reference_cam]

# Compute relative rotations and translations to the reference camera
relative_R_dict = {}
relative_T_dict = {}
for cam in cameras:
    if cam == reference_cam:
        relative_R_dict[cam] = np.eye(3)
        relative_T_dict[cam] = np.zeros((3, 1))
        continue
    R_cam = R_dict[cam]
    T_cam = T_dict[cam]
    # Relative rotation and translation
    relative_R = R_ref @ R_cam.T
    relative_T = -relative_R @ T_cam + T_ref
    relative_R_dict[cam] = relative_R
    relative_T_dict[cam] = relative_T

# Save results
np.savez('multi_camera_calibration_global_fixed_intrinsics.npz',
         mtx_dict=mtx_opt_dict, dist_dict=dist_opt_dict,
         relative_R_dict=relative_R_dict, relative_T_dict=relative_T_dict)

print("Global multi-camera calibration completed (fixed intrinsics). Results saved to multi_camera_calibration_global_fixed_intrinsics.npz")
#--------------------------------------TEST--------------------------------------#

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import sys
import glob

# ---------------------------- 1. 加载8相机的校准参数 ---------------------------- #

# 加载校准参数
calibration_data = np.load('multi_camera_calibration_global_fixed_intrinsics.npz', allow_pickle=True)
mtx_dict = calibration_data['mtx_dict'].item()
dist_dict = calibration_data['dist_dict'].item()
relative_R_dict = calibration_data['relative_R_dict'].item()
relative_T_dict = calibration_data['relative_T_dict'].item()

# 构建各相机的外部参数
cameras = ['Left', 'Left1', 'Left2', 'Center', 'Right', 'Right1', 'Right2', 'Right3']

# 选择参考相机，例如 'Center'
reference_cam = 'Center'
R_ref = np.eye(3)
T_ref = np.zeros((3, 1))

# 构建每个相机的旋转矩阵和平移向量
R_dict = {}
T_dict = {}

for cam in cameras:
    if cam == reference_cam:
        R_dict[cam] = R_ref
        T_dict[cam] = T_ref
    else:
        R_rel = relative_R_dict[cam]
        T_rel = relative_T_dict[cam]
        # 计算相机的全局旋转和平移
        R_cam = R_rel.T @ R_ref
        T_cam = -R_cam @ T_rel
        R_dict[cam] = R_cam
        T_dict[cam] = T_cam

# 构建投影矩阵
P_dict = {}
for cam in cameras:
    R = R_dict[cam]
    T = T_dict[cam]
    P = np.hstack((R, T))
    P_dict[cam] = P

# ---------------------------- 2. 读取8张图像并检测关键点 ---------------------------- #

# Paths to images
image_paths = {}
image_paths['Left'] = r"D:\20241112\SUBCALI_Left\subcaliCam_Left-3.jpeg"
image_paths['Left1'] = r"D:\20241112\SUBCALI_Left1\subcaliCam_Left1-3.jpeg"
image_paths['Left2'] = r"D:\20241112\SUBCALI_Left2\subcaliCam_Left2-3.jpeg"
image_paths['Center'] = r"D:\20241112\SUBCALI_Center\subcaliCam_Center-3.jpeg"
image_paths['Right'] = r"D:\20241112\SUBCALI_Right\subcaliCam_Right-3.jpeg"
image_paths['Right1'] = r"D:\20241112\SUBCALI_Right1\subcaliCam_Right1-3.jpeg"
image_paths['Right2'] = r"D:\20241112\SUBCALI_Right2\subcaliCam_Right2-3.jpeg"
image_paths['Right3'] = r"D:\20241112\SUBCALI_Right3\subcaliCam_Right3-3.jpeg"

# 读取图像
images = {}
for cam in cameras:
    img = cv2.imread(image_paths[cam])
    if img is None:
        print(f"无法读取相机 {cam} 的图像，请检查路径")
        sys.exit(1)
    images[cam] = img

# 初始化 MediaPipe PoseLandmarker
base_options = python.BaseOptions(model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# 检测关键点
pose_landmarks_dict = {}
for cam in cameras:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(images[cam], cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    pose_landmarks_list = detection_result.pose_landmarks
    if not pose_landmarks_list:
        print(f"未能在相机 {cam} 的图像中检测到关键点")
        sys.exit(1)
    # 提取关键点列表
    pose_landmarks_dict[cam] = pose_landmarks_list[0]

# 提取关键点坐标
pose_2d_points = {}
for cam in cameras:
    img = images[cam]
    landmarks = pose_landmarks_dict[cam]
    pose = np.zeros((len(landmarks), 2))
    for idx, landmark in enumerate(landmarks):
        a = landmark.x
        b = landmark.y
        pose[idx, 0] = img.shape[1] * a
        pose[idx, 1] = img.shape[0] * b
    pose_2d_points[cam] = pose

# ---------------------------- 3. 多视角三角测量 ---------------------------- #

def triangulate_points_nviews(proj_matrices, points_2d):
    """
    使用多个视角的投影矩阵和2D点，计算3D点坐标
    """
    num_views = len(proj_matrices)
    A = np.zeros((num_views * 2, 4))
    for i in range(num_views):
        P = proj_matrices[i]
        x, y = points_2d[i][0], points_2d[i][1]
        A[2 * i] = x * P[2, :] - P[0, :]
        A[2 * i + 1] = y * P[2, :] - P[1, :]
    # 通过 SVD 求解
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]
    return X[:3]

# 对每个关键点进行三角测量
points_3d = []
num_points = len(pose_2d_points[reference_cam])  # 以参考相机的关键点数量为准

# 检查每个相机的关键点数量是否一致
for cam in cameras:
    if len(pose_2d_points[cam]) != num_points:
        print(f"相机 {cam} 的关键点数量与参考相机不一致")
        sys.exit(1)

# 三角测量
for i in range(num_points):
    pts_2d = []
    proj_matrices = []
    for cam in cameras:
        # 获取相机内参和畸变参数
        mtx = mtx_dict[cam]
        dist = dist_dict[cam]
        # 获取2D点并去畸变
        pt = pose_2d_points[cam][i]
        pt_norm = cv2.undistortPoints(pt.reshape(1, 1, 2), mtx, dist).reshape(2)
        pts_2d.append(pt_norm)
        # 获取相机的投影矩阵
        P = P_dict[cam]
        proj_matrices.append(P)
    # 进行三角测量
    X = triangulate_points_nviews(proj_matrices, pts_2d)
    points_3d.append(X)

points_3d = np.array(points_3d)

# 输出3D点坐标
for i, point in enumerate(points_3d):
    print(f"3D point {i + 1}: {point}")

# ---------------------------- 4. 重新投影8张图像和误差计算 ---------------------------- #

# 定义重投影函数
def reproject_points(points_3d, rvec, tvec, mtx, dist):
    projected_points, _ = cv2.projectPoints(points_3d,
                                            rvec,
                                            tvec,
                                            mtx,
                                            dist)
    projected_points = projected_points.reshape(-1, 2)
    return projected_points

# 计算每个相机的重投影误差
errors_dict = {}
projected_points_dict = {}
for cam in cameras:
    # 获取相机的旋转向量和平移向量
    R = R_dict[cam]
    T = T_dict[cam]
    rvec, _ = cv2.Rodrigues(R)
    tvec = T.reshape(3)
    # 获取相机的内参和畸变参数
    mtx = mtx_dict[cam]
    dist = dist_dict[cam]
    # 重新投影
    projected_points = reproject_points(points_3d, rvec, tvec, mtx, dist)
    projected_points_dict[cam] = projected_points
    # 计算误差
    original_points = pose_2d_points[cam]
    errors = np.linalg.norm(original_points - projected_points, axis=1)
    mean_error = np.mean(errors)
    errors_dict[cam] = errors
    print(f"{cam} camera reprojection error: {mean_error}")

# ---------------------------- 5. 可视化结果 ---------------------------- #

def visualize_reprojection(image, original_points, projected_points, errors, window_name):
    desired_display_size = (900, 700)  # (宽度, 高度)
    scale_x = desired_display_size[0] / image.shape[1]
    scale_y = desired_display_size[1] / image.shape[0]
    image_resized = cv2.resize(image, desired_display_size, interpolation=cv2.INTER_AREA)
    original_points_scaled = original_points.copy()
    original_points_scaled[:, 0] *= scale_x
    original_points_scaled[:, 1] *= scale_y
    projected_points_scaled = projected_points.copy()
    projected_points_scaled[:, 0] *= scale_x
    projected_points_scaled[:, 1] *= scale_y
    image_copy = image_resized.copy()
    for idx, (orig_pt, proj_pt, error) in enumerate(zip(original_points_scaled, projected_points_scaled, errors)):
        x_orig, y_orig = int(orig_pt[0]), int(orig_pt[1])
        x_proj, y_proj = int(proj_pt[0]), int(proj_pt[1])
        # 绘制原始点（蓝色）
        cv2.circle(image_copy, (x_orig, y_orig), 5, (255, 0, 0), -1)
        # 绘制重投影点（绿色）
        cv2.circle(image_copy, (x_proj, y_proj), 5, (0, 255, 0), -1)
        # 显示误差值
        cv2.putText(image_copy, f"{error:.2f}", (x_orig, y_orig - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # 转换颜色空间以在 Matplotlib 中正确显示
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(window_name)
    plt.axis('off')
    plt.show()

# 可视化每个相机的重投影
for cam in cameras:
    image = images[cam]
    original_points = pose_2d_points[cam]
    projected_points = projected_points_dict[cam]
    errors = errors_dict[cam]
    window_name = f"{cam} Camera Reprojection"
    visualize_reprojection(image, original_points, projected_points, errors, window_name)

