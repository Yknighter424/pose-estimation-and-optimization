import cv2
import numpy as np
import sys
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess
import json
from scipy.optimize import minimize



# ---------------------------- 1. 单独相机校正 ---------------------------- #

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.io import savemat

# Number of inner corners in the chessboard
nx = 9  # 水平方向的角点数
ny = 6  # 垂直方向的角点数

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []         # 3D point in real world space
imgpoints_left = []    # 2D points in left image plane
imgpoints_center = []  # 2D points in center image plane
imgpoints_right = []   # 2D points in right image plane

# Load images
# 加载校正图像
images_left = glob.glob(r"D:\20241112\CAM_LEFT\Cam2-*.jpeg")
#"D:\20241112\CAM_LEFT\Cam2-*.jpeg"

images_center = glob.glob(r"D:\20241112\CAM_MIDDLE\Cam1-*.jpeg")
#"D:\20241112\CAM_MIDDLE\Cam1-*.jpeg"

images_right = glob.glob(r"D:\20241112\CAM_RIGHT\Cam0-*.jpeg")

# 确保所有图像列表的顺序一致
images_left.sort()
images_center.sort()
images_right.sort()

# Find chessboard corners for each image
for i in range(len(images_left)):
    img_left = cv2.imread(images_left[i])
    img_center = cv2.imread(images_center[i])
    img_right = cv2.imread(images_right[i])

    if img_left is None or img_center is None or img_right is None:
        print(f"无法读取图像组 {i}")
        print(f"左图: {images_left[i]} - {'读取成功' if img_left is not None else '读取失败'}")
        print(f"中图: {images_center[i]} - {'读取成功' if img_center is not None else '读取失败'}")
        print(f"右图: {images_right[i]} - {'读取成功' if img_right is not None else '读取失败'}")
        print("跳过这组图片\n")
        continue  # 跳过本次循环

    # Find corners in the chessboard pattern
    ret_left, corners_left = cv2.findChessboardCorners(img_left, (nx, ny), None)
    ret_center, corners_center = cv2.findChessboardCorners(img_center, (nx, ny), None)
    ret_right, corners_right = cv2.findChessboardCorners(img_right, (nx, ny), None)

    print("iter : ", i)
    print("ret_left :", ret_left)
    print("ret_center :", ret_center)
    print("ret_right :", ret_right)
    print("--------------")

    if ret_left and ret_center and ret_right:
        # Refine corner locations
        corners_left = cv2.cornerSubPix(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),
                                        corners_left, (11, 11), (-1, -1), criteria)
        corners_center = cv2.cornerSubPix(cv2.cvtColor(img_center, cv2.COLOR_BGR2GRAY),
                                          corners_center, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY),
                                         corners_right, (11, 11), (-1, -1), criteria)

        # Add object points, image points
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_center.append(corners_center)
        imgpoints_right.append(corners_right)

        # 获取图像尺寸（只需获取一次即可）
        if 'h' not in locals() or 'w' not in locals():
            h, w = img_left.shape[:2]
    else:
        print(f"无法在图片组 {i} 中找到所有角点")
        print(f"左图: {images_left[i]} - 棋盘检测结果: {ret_left}")
        print(f"中图: {images_center[i]} - 棋盘检测结果: {ret_center}")
        print(f"右图: {images_right[i]} - 棋盘检测结果: {ret_right}")
        print("跳过这组图片\n")

# 检查是否成功检测到足够的角点进行校准
if len(objpoints) == 0:
    print("没有足够的有效数据进行相机校准。请检查棋盘图像的质量和路径。")
    sys.exit()

# Perform camera calibration for each camera
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, (w, h), None, None)
ret_center, mtx_center, dist_center, rvecs_center, tvecs_center = cv2.calibrateCamera(
    objpoints, imgpoints_center, (w, h), None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, (w, h), None, None)

print("ret_left:", ret_left)
print("ret_center:", ret_center)
print("ret_right:", ret_right)
print("mtx_left:\n", mtx_left)
print("mtx_center:\n", mtx_center)
print("mtx_right:\n", mtx_right)

# 保存单相机校准结果
np.savez('single_camera_calibration.npz',
         mtx_left=mtx_left,
         dist_left=dist_left,
         rvecs_left=rvecs_left,
         tvecs_left=tvecs_left,
         mtx_center=mtx_center,
         dist_center=dist_center,
         rvecs_center=rvecs_center,
         tvecs_center=tvecs_center,
         mtx_right=mtx_right,
         dist_right=dist_right,
         rvecs_right=rvecs_right,
         tvecs_right=tvecs_right,
         image_size=(w, h))

print("校准结果已保存到 single_camera_calibration.npz")

# 计算并显示每个相机的视野 (FOV)
def calculate_fov(mtx, image_size):
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    fov_x = 2 * np.arctan(image_size[0] / (2 * fx)) * (180 / np.pi)
    fov_y = 2 * np.arctan(image_size[1] / (2 * fy)) * (180 / np.pi)
    return fov_x, fov_y

# 计算每个相机的FOV
fov_x_left, fov_y_left = calculate_fov(mtx_left, (w, h))
fov_x_center, fov_y_center = calculate_fov(mtx_center, (w, h))
fov_x_right, fov_y_right = calculate_fov(mtx_right, (w, h))

print(f"左相机水平视野 (FOV_x): {fov_x_left:.2f} degrees")

print(f"左相机垂直视野 (FOV_y): {fov_y_left:.2f} degrees")
print(f"中间相机水平视野 (FOV_x): {fov_x_center:.2f} degrees")
print(f"中间相机垂直视野 (FOV_y): {fov_y_center:.2f} degrees")
print(f"右相机水平视野 (FOV_x): {fov_x_right:.2f} degrees")
print(f"右相机垂直视野 (FOV_y): {fov_y_right:.2f} degrees")

# 显示去畸变结果
def show_undistorted_image(img, mtx, dist, title):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w_roi, h_roi = roi
    cv2.rectangle(undistorted, (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 2)
    resized = cv2.resize(undistorted, (900, 700))
    cv2.imshow(title, resized)
    return undistorted

# 读取示例图像并显示去畸变结果
img_left_sample = cv2.imread(images_left[0])
img_center_sample = cv2.imread(images_center[0])
img_right_sample = cv2.imread(images_right[0])

if img_left_sample is None or img_center_sample is None or img_right_sample is None:
    print("无法读取示例图像，用于显示去畸变结果。")
else:
    undist_left = show_undistorted_image(img_left_sample, mtx_left, dist_left, 'Undistorted Left Image')
    undist_center = show_undistorted_image(img_center_sample, mtx_center, dist_center, 'Undistorted Center Image')
    undist_right = show_undistorted_image(img_right_sample, mtx_right, dist_right, 'Undistorted Right Image')

    cv2.waitKey(0)
    cv2.destroyAllWindows()




'''def detect_chessboard_corners(image, pattern_size=(9, 6)):#
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                             cv2.CALIB_CB_FAST_CHECK)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners2
    else:
        return False, None'''



#----------------------Stereo_calibration---------------------------------------------------
def detect_chessboard_corners(image, pattern_size=(9, 6)):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用findChessboardCorners函数检测棋盘格角点
    # pattern_size: 棋盘格内角点的数量(宽x高)
    # CALIB_CB_ADAPTIVE_THRESH: 使用自适应阈值
    # CALIB_CB_NORMALIZE_IMAGE: 对图像进行归一化
    # CALIB_CB_FAST_CHECK: 快速检查模式
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                             cv2.CALIB_CB_FAST_CHECK)
    
    if ret:
        # 设置亚像素角点检测的终止条件
        # TERM_CRITERIA_EPS: 精度条件
        # TERM_CRITERIA_MAX_ITER: 最大迭代次数
        # 30: 最大迭代次数
        # 0.001: 最小精度要求
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 使用cornerSubPix进行亚像素级别的角点检测
        # (11,11): 搜索窗口大小
        # (-1,-1): 死区大小
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners2
    else:
        # 如果未检测到角点则返回False
        return False, None

# 设置棋盘格的参数
pattern_size = (9, 6)  # 根据您的棋盘格尺寸修改
square_size = 3.0  # 根据实际棋盘格方格的尺寸修改

# 準備物體點
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size
'''images_left = glob.glob(r"D:\20241112\DUAL_2\Cam2-*.jpeg")#左相機圖像路徑
#"D:\20241112\CAM_dual0\Cam0-*.jpeg"
images_center = glob.glob(r"D:\20241112\CAM_DUAL1\Cam1-*.jpeg")#中間相機圖像路徑
#"D:\20241112\CAM_DUAL1\Cam1-*.jpeg"
images_right = glob.glob(r"D:\20241112\CAM_dual0\Cam0-*.jpeg")#右相機圖像路徑'''
# 讀取影像
images_left = glob.glob(r"D:\20241112\DUAL_2\Cam2-*.jpeg")#左相機圖像路徑#cam-0_1.jpg=left
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\camL\Cam-1_1.jpg"
images_left.sort()
images_center = glob.glob(r"D:\20241112\CAM_DUAL1\Cam1-*.jpeg")#cam-1_1.jpg=right
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\camR\Cam-0_1.jpg"
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1013\dual\camR\Cam-1_1.jpg"
images_center.sort()

images_right = glob.glob(r"D:\20241112\CAM_dual0\Cam0-*.jpeg")#右相機圖像路徑#cam-2_1.jpg=right
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camR\cam2-walk.mp4"
images_right.sort()

objpoints = []  # 3D點
imgpoints_left = []  # 左相機2D點
imgpoints_center = []  # 中間相機2D點
imgpoints_right = []  # 右相機2D點
# 假設我們已經有了以下變數:
# objpoints_single, imgpoints_left_single, imgpoints_right_single
# image_width, image_height
# nx, ny (棋盤格的內角點數)
# square_size (棋盤格方格的實際尺寸,單位為mm)

# 遍历每组三相机的图像
for idx, (img_left_path, img_center_path, img_right_path) in enumerate(zip(images_left, images_center, images_right)):
    # 读取三个相机的图像
    img_left = cv2.imread(img_left_path)    # 读取左相机图像
    img_center = cv2.imread(img_center_path) # 读取中间相机图像  
    img_right = cv2.imread(img_right_path)   # 读取右相机图像
    
    # 检查图像是否成功读取
    if img_left is None or img_center is None or img_right is None:
        print(f"无法读取图像组 {idx+1}")
        print(f"左图: {img_left_path}")
        print(f"中图: {img_center_path}")
        print(f"右图: {img_right_path}")
        continue
    
    # 在三个相机图像中检测棋盘格角点
    ret_left, corners_left = detect_chessboard_corners(img_left, pattern_size)     # 检测左相机角点
    ret_center, corners_center = detect_chessboard_corners(img_center, pattern_size)# 检测中间相机角点
    ret_right, corners_right = detect_chessboard_corners(img_right, pattern_size)  # 检测右相机角点
    
    # 如果三个相机都成功检测到角点
    if ret_left and ret_center and ret_right:
        objpoints.append(objp)                  # 添加3D世界坐标点
        imgpoints_left.append(corners_left)     # 添加左相机2D角点
        imgpoints_center.append(corners_center) # 添加中间相机2D角点
        imgpoints_right.append(corners_right)   # 添加右相机2D角点
        
        # 创建用于显示的图像副本
        img_left_display = img_left.copy()
        img_center_display = img_center.copy()
        img_right_display = img_right.copy()
        
        # 在图像上绘制检测到的棋盘格角点
        cv2.drawChessboardCorners(img_left_display, pattern_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_center_display, pattern_size, corners_center, ret_center)
        cv2.drawChessboardCorners(img_right_display, pattern_size, corners_right, ret_right)
        
        # 使用matplotlib显示结果
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(cv2.cvtColor(img_left_display, cv2.COLOR_BGR2RGB))
        plt.title(f'Left Image {idx+1}'), plt.axis('off')
        plt.subplot(132), plt.imshow(cv2.cvtColor(img_center_display, cv2.COLOR_BGR2RGB))
        plt.title(f'Center Image {idx+1}'), plt.axis('off')
        plt.subplot(133), plt.imshow(cv2.cvtColor(img_right_display, cv2.COLOR_BGR2RGB))
        plt.title(f'Right Image {idx+1}'), plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        # 如果有任何相机未检测到角点,输出错误信息
        print(f"无法在图片组 {idx+1} 中找到所有角点")
        print(f"左图: {img_left_path} - 检测结果: {ret_left}")
        print(f"中图: {img_center_path} - 检测结果: {ret_center}")
        print(f"右图: {img_right_path} - 检测结果: {ret_right}")
        print("跳过这组图片\n")

# 设置立体校正参数
# 设置标定参数
flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参标志,使用已有的相机内参进行优化
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)  # 迭代终止条件:最大迭代100次或精度达到1e-5

## ------------------------ 1. 三相机立体校正部分 ------------------------ #
# 左-中相机对的校正
# 使用cv2.stereoCalibrate计算左右相机之间的变换关系
# 输入:
# - objpoints: 棋盘格角点的3D坐标(世界坐标系)
# - imgpoints_left/center: 左/中相机图像中检测到的棋盘格角点2D坐标
# - mtx_left/center, dist_left/center: 左/中相机的内参和畸变系数
# 输出:
# - retval_lc: 重投影误差
# - R_lc: 从左到中相机的旋转矩阵
# - T_lc: 从左到中相机的平移向量
# - E_lc: 本质矩阵
# - F_lc: 基础矩阵
retval_lc, mtx_left, dist_left, mtx_center, dist_center, R_lc, T_lc, E_lc, F_lc = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_center, 
    mtx_left, dist_left, mtx_center, dist_center, 
    img_left.shape[:2][::-1], criteria=criteria, flags=flags)

# 中-右相机对的校正
# 使用相同方法计算中间相机到右相机的变换关系
# 注意这里忽略了部分返回值(用_表示),因为中间相机的参数已在上一步获得
retval_cr, _, _, mtx_right, dist_right, R_cr, T_cr, E_cr, F_cr = cv2.stereoCalibrate(
    objpoints, imgpoints_center, imgpoints_right, 
    mtx_center, dist_center, mtx_right, dist_right, 
    img_center.shape[:2][::-1], criteria=criteria, flags=flags)

# ------------------------ 2. 投影矩阵计算部分 ------------------------ #
# 计算从左相机到右相机的变换
# 根据矩阵乘法规则:
# 1. 旋转矩阵的组合: R_lr = R_cr @ R_lc 
#    (先应用R_lc从左到中,再应用R_cr从中到右)
R_lr = R_cr @ R_lc  # 组合旋转矩阵

# 2. 平移向量的组合: T_lr = R_cr @ T_lc + T_cr
#    (先将T_lc通过R_cr转换到右相机坐标系,再加上T_cr)
T_lr = R_cr @ T_lc + T_cr  # 组合平移向量

# 计算三个相机的投影矩阵
# 投影矩阵P = [R|t],其中R为3x3旋转矩阵,t为3x1平移向量
P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # 左相机作为参考坐标系,因此R=I,t=0
P2 = np.hstack([R_lc, T_lc.reshape(3, 1)])     # 中间相机相对于左相机的投影矩阵
P3 = np.hstack([R_lr, T_lr.reshape(3, 1)])     # 右相机相对于左相机的投影矩阵

# ------------------------ 3. 修改保存结果部分 ------------------------ #
np.savez('triple_camera_calibration.npz',
         # 相机内部参数
         mtx_left=mtx_left, dist_left=dist_left,
         mtx_center=mtx_center, dist_center=dist_center,
         mtx_right=mtx_right, dist_right=dist_right,
         # 左-中相机对的外部参数
         R_lc=R_lc, T_lc=T_lc, E_lc=E_lc, F_lc=F_lc,
         # 中-右相机对的外部参数
         R_cr=R_cr, T_cr=T_cr, E_cr=E_cr, F_cr=F_cr,
         # 左-右相机对的组合变换
         R_lr=R_lr, T_lr=T_lr,
         # 投影矩阵
         P1=P1, P2=P2, P3=P3,
         # 其他参数
         image_size=img_left.shape[:2][::-1],
         calibration_error_lc=retval_lc,
         calibration_error_cr=retval_cr)

print("立體校正結果已保存到 triple_camera_calibration.npz")


# 从校准文件加载相机参数
calibration_data = np.load('triple_camera_calibration.npz')
mtx_left = calibration_data['mtx_left']
dist_left = calibration_data['dist_left']
mtx_center = calibration_data['mtx_center']
dist_center = calibration_data['dist_center'] 
mtx_right = calibration_data['mtx_right']
dist_right = calibration_data['dist_right']

# 加载转换矩阵
R_lc = calibration_data['R_lc']  # 左-中相机旋转矩阵
T_lc = calibration_data['T_lc']  # 左-中相机平移向量
R_cr = calibration_data['R_cr']  # 中-右相机旋转矩阵
T_cr = calibration_data['T_cr']  # 中-右相机平移向量
R_lr = calibration_data['R_lr']  # 左-右相机组合旋转矩阵
T_lr = calibration_data['T_lr']  # 左-右相机组合平移向量

# 加载投影矩阵
P1 = calibration_data['P1']  # 左相机投影矩阵
P2 = calibration_data['P2']  # 中间相机投影矩阵
P3 = calibration_data['P3']  # 右相机投影矩阵

# 加载三个相机的图像
imggg1 = cv2.imread(r"D:\20241112\SUBCALI2\subcaliCam2-0.jpeg")     # 左相机图像'path/to/left_image.jpg'
imggg2 = cv2.imread(r"D:\20241112\SUBCALI1\subcaliCam1-0.jpeg")   # 中间相机图像'path/to/center_image.jpg'
imggg3 = cv2.imread(r"D:\20241112\SUBCALI0\subcaliCam0-0.jpeg")

# 设置MediaPipe
base_options = python.BaseOptions(model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# MediaPipe检测
mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg1, cv2.COLOR_BGR2RGB))
mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg2, cv2.COLOR_BGR2RGB))
mp_image3 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg3, cv2.COLOR_BGR2RGB))

detection_result1 = detector.detect(mp_image1)
detection_result2 = detector.detect(mp_image2)
detection_result3 = detector.detect(mp_image3)

# 提取关键点
pose_landmarks_list1 = detection_result1.pose_landmarks
pose_landmarks_list2 = detection_result2.pose_landmarks
pose_landmarks_list3 = detection_result3.pose_landmarks

# 转换为numpy数组
pose1 = np.zeros((len(pose_landmarks_list1[0]), 2))
pose2 = np.zeros((len(pose_landmarks_list2[0]), 2))
pose3 = np.zeros((len(pose_landmarks_list3[0]), 2))

# 提取左相机关键点
for i, landmark in enumerate(pose_landmarks_list1[0]):
    pose1[i, 0] = int(1920 * landmark.x)
    pose1[i, 1] = int(1200 * landmark.y)

# 提取中间相机关键点
for i, landmark in enumerate(pose_landmarks_list2[0]):
    pose2[i, 0] = int(1920 * landmark.x)
    pose2[i, 1] = int(1200 * landmark.y)

# 提取右相机关键点
for i, landmark in enumerate(pose_landmarks_list3[0]):
    pose3[i, 0] = int(1920 * landmark.x)
    pose3[i, 1] = int(1200 * landmark.y)

# 三维重建
'''points_3d = []
for i in range(len(pose1)):
    # 对三个相机的点进行去畸变
    pose_temp_1 = cv2.undistortPoints(pose1[i].reshape(1,1,2), mtx_left, dist_left, None, None)
    pose_temp_2 = cv2.undistortPoints(pose2[i].reshape(1,1,2), mtx_center, dist_center, None, None)
    pose_temp_3 = cv2.undistortPoints(pose3[i].reshape(1,1,2), mtx_right, dist_right, None, None)
    
    # 使用DLT方法进行三维重建
    A = np.zeros((6, 4))
    
    # 左相机约束
    A[0:2] = pose_temp_1[0,0,0] * P1[2] - P1[0]
    A[1] = pose_temp_1[0,0,1] * P1[2] - P1[1]
    
    # 中间相机约束
    A[2:4] = pose_temp_2[0,0,0] * P2[2] - P2[0]
    A[3] = pose_temp_2[0,0,1] * P2[2] - P2[1]
    
    # 右相机约束
    A[4:6] = pose_temp_3[0,0,0] * P3[2] - P3[0]
    A[5] = pose_temp_3[0,0,1] * P3[2] - P3[1]
    
    # 求解最小二乘问题
    _, _, Vt = np.linalg.svd(A)
    point_4d = Vt[-1]
    point_3d = point_4d[:3] / point_4d[3]
    points_3d.append(point_3d)

points_3d = np.array(points_3d)'''
# 3D重建
points_3d = []
# 初始化一個空列表來存儲三維點坐標

# 三維重建可以改成使用triangulation的套件實現8相機任意兩個可見度高的進行重建的邏輯；下面是使用DLT算法進行三維重建
# 我是希望能做到使用三個可見度高的相機進行三維重建 ，這樣可以讓結果穩定且精準
for i in range(len(pose1)):
    # 對每個關鍵點進行迭代
    
    # 對三個相機的點進行去畸變並歸一化
    pose_temp_1 = cv2.undistortPoints(pose1[i].reshape(1,1,2), mtx_left, dist_left)
    pose_temp_2 = cv2.undistortPoints(pose2[i].reshape(1,1,2), mtx_center, dist_center)
    pose_temp_3 = cv2.undistortPoints(pose3[i].reshape(1,1,2), mtx_right, dist_right)
    # 使用相機內參和畸變係數對每個相機的2D點進行去畸變和歸一化
    
    # 由於我們使用的是歸一化的圖像坐標，因此投影矩陣應只包含外參
    # 獲取每個相機的旋轉矩陣和平移向量
    R1 = np.eye(3)  # 左相機作為參考坐標系，旋轉矩陣為單位矩陣
    T1 = np.zeros((3, 1))  # 左相機作為原點，平移向量為零向量

    R2 = R_lc  # 中間相機相對於左相機的旋轉矩陣
    T2 = T_lc.reshape(3, 1)  # 中間相機相對於左相機的平移向量

    R3 = R_lr  # 右相機相對於左相機的旋轉矩陣
    T3 = T_lr.reshape(3, 1)  # 右相機相對於左相機的平移向量

    # 構建每個相機的投影矩陣（僅包含外參）
    P1 = np.hstack((R1, T1))  # 左相機投影矩陣
    P2 = np.hstack((R2, T2))  # 中間相機投影矩陣
    P3 = np.hstack((R3, T3))  # 右相機投影矩陣

    # 構建DLT算法的矩陣A
    A = np.zeros((6, 4))  # 初始化6x4的矩陣A

    # 左相機約束
    x1, y1 = pose_temp_1[0, 0, 0], pose_temp_1[0, 0, 1]  # 提取左相機歸一化坐標
    A[0] = x1 * P1[2] - P1[0]  # DLT方程的第一行
    A[1] = y1 * P1[2] - P1[1]  # DLT方程的第二行

    # 中間相機約束
    x2, y2 = pose_temp_2[0, 0, 0], pose_temp_2[0, 0, 1]  # 提取中間相機歸一化坐標
    A[2] = x2 * P2[2] - P2[0]  # DLT方程的第三行
    A[3] = y2 * P2[2] - P2[1]  # DLT方程的第四行

    # 右相機約束
    x3, y3 = pose_temp_3[0, 0, 0], pose_temp_3[0, 0, 1]  # 提取右相機歸一化坐標
    A[4] = x3 * P3[2] - P3[0]  # DLT方程的第五行
    A[5] = y3 * P3[2] - P3[1]  # DLT方程的第六行

    # 求解最小二乘問題
    _, _, Vt = np.linalg.svd(A)  # 對矩陣A進行奇異值分解
    X = Vt[-1]  # 取最小奇異值對應的特徵向量作為解
    point_3d = X[:3] / X[3]  # 將齊次坐標轉換為3D坐標
    points_3d.append(point_3d)  # 將3D點添加到結果列表中

points_3d = np.array(points_3d)  # 將結果轉換為numpy數組



# 计算重投影点
# 确保旋转向量和平移向量的形状为 (3, 1)
rvec_left = np.zeros((3, 1))  # 左相机作为参考坐标系
tvec_left = np.zeros((3, 1))

# 中间相机的旋转向量和平移向量
rvec_center, _ = cv2.Rodrigues(R_lc)
tvec_center = T_lc.reshape(3, 1)

# 右相机的旋转向量和平移向量
rvec_right, _ = cv2.Rodrigues(R_lr)
tvec_right = T_lr.reshape(3, 1)

# 计算重投影点
projected_points_left, _ = cv2.projectPoints(points_3d, rvec_left, tvec_left, mtx_left, dist_left)
projected_points_center, _ = cv2.projectPoints(points_3d, rvec_center, tvec_center, mtx_center, dist_center)
projected_points_right, _ = cv2.projectPoints(points_3d, rvec_right, tvec_right, mtx_right, dist_right)

# 将投影点从形状 (N, 1, 2) 转换为 (N, 2)
projected_points_left = projected_points_left.reshape(-1, 2)
projected_points_center = projected_points_center.reshape(-1, 2)
projected_points_right = projected_points_right.reshape(-1, 2)

# 计算重投影误差
errors_left = np.linalg.norm(pose1 - projected_points_left, axis=1)
errors_center = np.linalg.norm(pose2 - projected_points_center, axis=1)
errors_right = np.linalg.norm(pose3 - projected_points_right, axis=1)

# 打印平均重投影误差
print(f"Left camera mean reprojection error: {np.mean(errors_left):.2f} pixels")
print(f"Center camera mean reprojection error: {np.mean(errors_center):.2f} pixels")
print(f"Right camera mean reprojection error: {np.mean(errors_right):.2f} pixels")

# 打印每个关键点的重投影误差
print("\nDetailed reprojection errors for each joint:")
for i in range(len(errors_left)):
    print(f"Joint {i+1}:")
    print(f"  Left camera: {errors_left[i]:.2f} pixels")
    print(f"  Center camera: {errors_center[i]:.2f} pixels")
    print(f"  Right camera: {errors_right[i]:.2f} pixels")

# 可视化函数
def visualize_reprojection(image, original_points, projected_points, errors, window_name):
    desired_display_size = (900, 700)
    scale_x = desired_display_size[0] / image.shape[1]
    scale_y = desired_display_size[1] / image.shape[0]
    
    image_resized = cv2.resize(image, desired_display_size, interpolation=cv2.INTER_AREA)
    
    original_points_scaled = original_points.copy()
    # 根據縮放比例調整原始關鍵點座標的x和y值
    original_points_scaled[:, 0] *= scale_x  # 調整x座標
    original_points_scaled[:, 1] *= scale_y  # 調整y座標
    
    projected_points_scaled = projected_points.copy()
    projected_points_scaled[:, 0] *= scale_x#第一列的所有行,即所有x座標
    projected_points_scaled[:, 1] *= scale_y#第二列的所有行,即所有y座標
    
    image_copy = image_resized.copy()
    
    # 绘制骨架连接
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),    # 左眼
        (0, 4), (4, 5), (5, 6), (6, 8),    # 右眼
        (9, 10),                           # 嘴
        (11, 13), (13, 15),                # 左臂
        (15, 17), (15, 19), (15, 21),      # 左手
        (17, 19), (12, 14), (14, 16),      # 右臂
        (16, 18), (16, 20), (16, 22),      # 右手
        (18, 20), (11, 12),                # 肩膀
        (11, 23), (12, 24),                # 躯干两侧
        (23, 24),                          # 臀部
        (23, 25), (24, 26),                # 大腿
        (25, 27), (27, 29), (29, 31),      # 左腿
        (26, 28), (28, 30), (30, 32),      # 右腿
        (27, 31), (28, 32)                 # 脚
    ]
    
    # 绘制原始点的骨架（蓝色）
    for connection in connections:
        pt1 = tuple(map(int, original_points_scaled[connection[0]]))
        pt2 = tuple(map(int, original_points_scaled[connection[1]]))
        cv2.line(image_copy, pt1, pt2, (255, 0, 0), 2)
    
    # 绘制重投影点的骨架（绿色）
    for connection in connections:
        pt1 = tuple(map(int, projected_points_scaled[connection[0]]))
        pt2 = tuple(map(int, projected_points_scaled[connection[1]]))
        cv2.line(image_copy, pt1, pt2, (0, 255, 0), 2)
    
    # 绘制关键点和误差值
    for idx, (orig_pt, proj_pt, error) in enumerate(zip(original_points_scaled, projected_points_scaled, errors)):
        x_orig, y_orig = int(orig_pt[0]), int(orig_pt[1])
        x_proj, y_proj = int(proj_pt[0]), int(proj_pt[1])
        
        cv2.circle(image_copy, (x_orig, y_orig), 5, (255, 0, 0), -1)  # 原始点（蓝色）
        cv2.circle(image_copy, (x_proj, y_proj), 5, (0, 255, 0), -1)  # 重投影点（绿色）
        cv2.putText(image_copy, f"{error:.2f}", (x_orig, y_orig - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 显示图像
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(window_name)
    plt.axis('off')
    plt.show()

# 显示三个相机的重投影结果
visualize_reprojection(imggg1, pose1, projected_points_left, errors_left, "Left Camera Reprojection")
visualize_reprojection(imggg2, pose2, projected_points_center, errors_center, "Center Camera Reprojection")
visualize_reprojection(imggg3, pose3, projected_points_right, errors_right, "Right Camera Reprojection")

# 3D可视化函数
def visualize_3d_pose(points_3d, title="3D Pose"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关键点
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    
    # 绘制骨架连接
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),    # 左眼
        (0, 4), (4, 5), (5, 6), (6, 8),    # 右眼
        (9, 10),                           # 嘴
        (11, 13), (13, 15),                # 左臂
        (15, 17), (15, 19), (15, 21),      # 左手
        (17, 19), (12, 14), (14, 16),      # 右臂
        (16, 18), (16, 20), (16, 22),      # 右手
        (18, 20), (11, 12),                # 肩膀
        (11, 23), (12, 24),                # 躯干两侧
        (23, 24),                          # 臀部
        (23, 25), (24, 26),                # 大腿
        (25, 27), (27, 29), (29, 31),      # 左腿
        (26, 28), (28, 30), (30, 32),      # 右腿
        (27, 31), (28, 32)                 # 脚
    ]
    
    for connection in connections:
        pt1 = points_3d[connection[0]]
        pt2 = points_3d[connection[1]]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'r-')
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置标题
    plt.title(title)
    
    # 设置视角
    ax.view_init(elev=10, azim=-60)
    
    plt.show()

# 显示3D重建结果
visualize_3d_pose(points_3d, "3D Reconstruction Result")
#-----------------------------------------------END------------------------------------------------------#
'''def triangulate_points_nviews_weighted(proj_matrices, points_2d, weights):
    """
    帶權重的多視角三角測量
    
    參數:
    - proj_matrices: 投影矩陣列表
    - points_2d: 2D點列表
    - weights: 每個視角對應點的權重 (來自 MediaPipe 的可見度分數)
    """
    num_views = len(proj_matrices)
    A = np.zeros((num_views * 2, 4))
    W = np.zeros((num_views * 2, num_views * 2))  # 權重矩陣
    
    for i in range(num_views):
        P = proj_matrices[i]
        x, y = points_2d[i][0], points_2d[i][1]
        w = weights[i]
        
        # 構建 A 矩陣
        A[2*i] = x * P[2, :] - P[0, :]
        A[2*i + 1] = y * P[2, :] - P[1, :]
        
        # 設置權重
        W[2*i, 2*i] = w
        W[2*i+1, 2*i+1] = w
    
    # 應用權重
    WA = W @ A
    
    # 通過 SVD 求解加權最小二乘問題
    U, S, Vt = np.linalg.svd(WA)
    X = Vt[-1]
    X = X / X[3]  # 齊次座標歸一化
    
    return X[:3]

# 使用示例：
points_3d = []
for i in range(num_points):
    # 獲取三個相機的2D點和對應的可見度分數
    pts_2d = []
    weights = []
    
    # 左相機
    pt1_norm = cv2.undistortPoints(pose1[i].reshape(1,1,2), mtx_left, dist_left).reshape(2)
    pts_2d.append(pt1_norm)
    weights.append(pose1_visibility[i])
    
    # 中間相機
    pt2_norm = cv2.undistortPoints(pose2[i].reshape(1,1,2), mtx_center, dist_center).reshape(2)
    pts_2d.append(pt2_norm)
    weights.append(pose2_visibility[i])
    
    # 右相機
    pt3_norm = cv2.undistortPoints(pose3[i].reshape(1,1,2), mtx_right, dist_right).reshape(2)
    pts_2d.append(pt3_norm)
    weights.append(pose3_visibility[i])
    
    # 加權三角測量
    X = triangulate_points_nviews_weighted(proj_matrices, pts_2d, weights)
    points_3d.append(X)'''
#下面可以不看













































import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python

import matplotlib.pyplot as plt
from mediapipe.tasks.python import vision
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D

# 1. 加载相机参数和MediaPipe模型
def load_camera_params(file_path):
    # 从NPZ文件加载相机参数
    data = np.load(file_path)
    return (data['mtx_left'], data['dist_left'], 
            data['mtx_center'], data['dist_center'],
            data['mtx_right'], data['dist_right'],
            data['R_lc'], data['T_lc'],  # 左-中相机对的外参
            data['R_cr'], data['T_cr'])  # 中-右相机对的外参

# 从之前保存的标定结果中加载 R 和 T
calibration_data = np.load('triple_camera_calibration.npz')
R = calibration_data['R_lc']  # 使用左-中相机对的旋转矩阵
T = calibration_data['T_lc']  # 使用左-中相机对的平移向量

# 右相机的旋转向量和平移向量
rvec_right, _ = cv2.Rodrigues(R)
tvec_right = T

# 设置MediaPipe模型
base_options = python.BaseOptions(model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# 2. 处理视频帧
def process_frame(frame_left, frame_right):
    # 使用MediaPipe检测人体姿势
    mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
    mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))
    
    detection_result_left = detector.detect(mp_image_left)
    detection_result_right = detector.detect(mp_image_right)
    
    # 提取关键点
    pose_left = np.array([[landmark.x * frame_left.shape[1], landmark.y * frame_left.shape[0]] 
                          for landmark in detection_result_left.pose_landmarks[0]])
    pose_right = np.array([[landmark.x * frame_right.shape[1], landmark.y * frame_right.shape[0]] 
                           for landmark in detection_result_right.pose_landmarks[0]])
    
    # 三维重建
    points_3d = triangulate_points(pose_left, pose_right, mtx_left, dist_left, mtx_right, dist_right, R, T)
    
    return points_3d

# 3. 三角测量函数
def triangulate_points(points_left, points_right, mtx_left, dist_left, mtx_right, dist_right, R, T):
    # 构建投影矩阵
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T.reshape(3, 1)))
    
    points_3d = []
    for pt_left, pt_right in zip(points_left, points_right):
        pt_left_undist = cv2.undistortPoints(pt_left.reshape(1, 1, 2), mtx_left, dist_left)
        pt_right_undist = cv2.undistortPoints(pt_right.reshape(1, 1, 2), mtx_right, dist_right)
        
        point_4d = cv2.triangulatePoints(P1, P2, pt_left_undist, pt_right_undist)
        point_3d = (point_4d / point_4d[3])[:3]
        
        points_3d.append(point_3d.ravel())
    
    return np.array(points_3d)

# 4. 主处理循环
def process_videos(video_path_left, video_path_right):
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)
    
    all_points_3d = []
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            break
        
        points_3d = process_frame(frame_left, frame_right)
        all_points_3d.append(points_3d)
    
    cap_left.release()
    cap_right.release()
    
    return np.array(all_points_3d)
print(process_frame)
print(process_videos)

# 5. 可视化函数
def visualize_3d_animation(all_points_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算所有帧的坐标范围
    all_points = np.concatenate(all_points_3d)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # 设置固定的坐标轴范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # 初始化散点图
    scatter = ax.scatter([], [], [])
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 更新函数
    def update(frame):
        points = all_points_3d[frame]
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        ax.set_title(f'3D Animation - Frame {frame}')
        return scatter,
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(all_points_3d), interval=50, blit=False, repeat=True)
    
    plt.show()

from scipy.signal import savgol_filter
def smooth_points_savgol(points, window_size=7, polyorder=3):
    smoothed_points = np.zeros_like(points)
    for i in range(points.shape[1]):
        for j in range(3):  # x, y, z
            smoothed_points[:, i, j] = savgol_filter(points[:, i, j], window_size, polyorder)
    return smoothed_points

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
def visualize_3d_animation_comparison1(original_points, smoothed_points, R, T):
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 计算所有点的范围
    all_points = np.vstack((original_points.reshape(-1, 3), smoothed_points.reshape(-1, 3)))
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    
    # 设置坐标轴范围，添加一些边距
    margin = 0.1 * range_vals
    for ax in [ax1, ax2]:
        ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
        ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
        ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # 设置视角
    for ax in [ax1, ax2]:
        ax.view_init(elev=90, azim=90)
    
    # 初始化散点图和线条
    scatter1 = ax1.scatter([], [], [], s=20, c='r', alpha=0.6)
    scatter2 = ax2.scatter([], [], [], s=20, c='b', alpha=0.6)
    lines1 = []
    lines2 = []
    
    # 绘制全局坐标系统的左相机位置（原点）
    left_camera_pos = np.array([0, 0, 0])
    ax1.scatter(*left_camera_pos, color='k', s=100, label='Left Camera')
    ax2.scatter(*left_camera_pos, color='k', s=100, label='Left Camera')
    
    # 计算右相机位置
    right_camera_pos = T.flatten()
    ax1.scatter(*right_camera_pos, color='m', s=100, label='Right Camera')
    ax2.scatter(*right_camera_pos, color='m', s=100, label='Right Camera')
    
    # 绘制相机坐标轴，根据点云范围调整长度
    axis_length = 0.2 * np.min(range_vals)
    
    # 左相机坐标轴
    ax1.quiver(*left_camera_pos, axis_length, 0, 0, color='r', label='X_axis')
    ax1.quiver(*left_camera_pos, 0, axis_length, 0, color='g', label='Y_axis')
    ax1.quiver(*left_camera_pos, 0, 0, axis_length, color='b', label='Z_axis')
    ax2.quiver(*left_camera_pos, axis_length, 0, 0, color='r')
    ax2.quiver(*left_camera_pos, 0, axis_length, 0, color='g')
    ax2.quiver(*left_camera_pos, 0, 0, axis_length, color='b')
    
    # 右相机坐标轴
    ax1.quiver(*right_camera_pos, *(R @ np.array([axis_length, 0, 0])), color='r')
    ax1.quiver(*right_camera_pos, *(R @ np.array([0, axis_length, 0])), color='g')
    ax1.quiver(*right_camera_pos, *(R @ np.array([0, 0, axis_length])), color='b')
    ax2.quiver(*right_camera_pos, *(R @ np.array([axis_length, 0, 0])), color='r')
    ax2.quiver(*right_camera_pos, *(R @ np.array([0, axis_length, 0])), color='g')
    ax2.quiver(*right_camera_pos, *(R @ np.array([0, 0, axis_length])), color='b')
    
    # 添加图例
    ax1.legend()
    ax2.legend()
    
    # 定义MediaPipe姿势骨架的连接
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),    # 左眼
        (0, 4), (4, 5), (5, 6), (6, 8),    # 右眼
        (9, 10),                           # 嘴
        (11, 13), (13, 15),                # 左臂
        (15, 17), (15, 19), (15, 21),      # 左手
        (17, 19), (12, 14), (14, 16),      # 右臂
        (16, 18), (16, 20), (16, 22),      # 右手
        (18, 20), (11, 12),                # 肩膀
        (11, 23), (12, 24),                # 躯干两侧
        (23, 24),                          # 臀部
        (23, 25), (24, 26),                # 大腿
        (25, 27), (27, 29), (29, 31),      # 左腿
        (26, 28), (28, 30), (30, 32),      # 右腿
        (27, 31), (28, 32)                 # 脚
    ]
    
    for _ in POSE_CONNECTIONS:
        line1, = ax1.plot([], [], [], 'r-', lw=2, alpha=0.7)
        line2, = ax2.plot([], [], [], 'b-', lw=2, alpha=0.7)
        lines1.append(line1)
        lines2.append(line2)
    
    # 更新函数
    def update(frame):
        original = original_points[frame]
        smoothed = smoothed_points[frame]
        
        scatter1._offsets3d = (original[:, 0], original[:, 1], original[:, 2])
        scatter2._offsets3d = (smoothed[:, 0], smoothed[:, 1], smoothed[:, 2])
        
        for i, (start, end) in enumerate(POSE_CONNECTIONS):
            lines1[i].set_data([original[start, 0], original[end, 0]], 
                               [original[start, 1], original[end, 1]])
            lines1[i].set_3d_properties([original[start, 2], original[end, 2]])
            
            lines2[i].set_data([smoothed[start, 0], smoothed[end, 0]], 
                               [smoothed[start, 1], smoothed[end, 1]])
            lines2[i].set_3d_properties([smoothed[start, 2], smoothed[end, 2]])
        
        ax1.set_title(f'original - Frame {frame}')
        ax2.set_title(f'smoothed - Frame {frame}')
        
        return scatter1, scatter2, *lines1, *lines2
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(original_points), interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()
def visualize_3d_animation_comparison(original_points, smoothed_points):
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 计算所有点的范围
    all_points = np.vstack((original_points.reshape(-1, 3), smoothed_points.reshape(-1, 3)))
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    
    # 设置坐标轴范围，添加一些边距
    margin = 0.1 * range_vals
    for ax in [ax1, ax2]:
        ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
        ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
        ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # 设置视角
    for ax in [ax1, ax2]:
        ax.view_init(elev=90, azim=90)
    
    # 初始化散点图和线条
    scatter1 = ax1.scatter([], [], [], s=20, c='r', alpha=0.6)
    scatter2 = ax2.scatter([], [], [], s=20, c='b', alpha=0.6)
    lines1 = []
    lines2 = []
    
    # 绘制全局坐标系统的相机位置
    camera_pos = np.array([0, 0, 0])
    ax1.scatter(*camera_pos, color='k', s=100, label='camera')
    ax2.scatter(*camera_pos, color='k', s=100, label='camera')
    
    # 绘制相机坐标轴，根据点云范围调整长度
    axis_length = 0.2 * np.min(range_vals)
    ax1.quiver(*camera_pos, axis_length, 0, 0, color='r', label='X_axis')
    ax1.quiver(*camera_pos, 0, axis_length, 0, color='g', label='Y_axis')
    ax1.quiver(*camera_pos, 0, 0, axis_length, color='b', label='Z_axis')
    ax2.quiver(*camera_pos, axis_length, 0, 0, color='r')
    ax2.quiver(*camera_pos, 0, axis_length, 0, color='g')
    ax2.quiver(*camera_pos, 0, 0, axis_length, color='b')
    
    # 添加图例
    ax1.legend()
    ax2.legend()
    
    # 定义MediaPipe姿势骨架的连接
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),    # 左眼
        (0, 4), (4, 5), (5, 6), (6, 8),    # 右眼
        (9, 10),                           # 嘴
        (11, 13), (13, 15),                # 左臂
        (15, 17), (15, 19), (15, 21),      # 左手
        (17, 19), (12, 14), (14, 16),      # 右臂
        (16, 18), (16, 20), (16, 22),      # 右手
        (18, 20), (11, 12),                # 肩膀
        (11, 23), (12, 24),                # 躯干两侧
        (23, 24),                          # 臀部
        (23, 25), (24, 26),                # 大腿
        (25, 27), (27, 29), (29, 31),      # 左腿
        (26, 28), (28, 30), (30, 32),      # 右腿
        (27, 31), (28, 32)                 # 脚
    ]
    
    for _ in POSE_CONNECTIONS:
        line1, = ax1.plot([], [], [], 'r-', lw=2, alpha=0.7)
        line2, = ax2.plot([], [], [], 'b-', lw=2, alpha=0.7)
        lines1.append(line1)
        lines2.append(line2)
    
    # 更新函数
    def update(frame):
        original = original_points[frame]
        smoothed = smoothed_points[frame]
        
        scatter1._offsets3d = (original[:, 0], original[:, 1], original[:, 2])
        scatter2._offsets3d = (smoothed[:, 0], smoothed[:, 1], smoothed[:, 2])
        
        for i, (start, end) in enumerate(POSE_CONNECTIONS):
            lines1[i].set_data([original[start, 0], original[end, 0]], 
                               [original[start, 1], original[end, 1]])
            lines1[i].set_3d_properties([original[start, 2], original[end, 2]])
            
            lines2[i].set_data([smoothed[start, 0], smoothed[end, 0]], 
                               [smoothed[start, 1], smoothed[end, 1]])
            lines2[i].set_3d_properties([smoothed[start, 2], smoothed[end, 2]])
        
        ax1.set_title(f'original - Frame {frame}')
        ax2.set_title(f'smoothed - Frame {frame}')
        
        return scatter1, scatter2, *lines1, *lines2
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(original_points), interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
import matplotlib.animation as animation
#elev=10, azim=-70
def save_animation_as_mp4(points, filename, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 計算所有點的範圍
    all_points = np.concatenate(points)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    
    # 設置坐標軸範圍，添加一些邊距
    margin = 0.1 * range_vals
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
    
    # 設置坐標軸標籤
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 初始化散點圖
    scatter = ax.scatter([], [], [], s=20, c='r', alpha=0.6)
    
    # 定義MediaPipe姿勢骨架的連接
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),    # 左眼
        (0, 4), (4, 5), (5, 6), (6, 8),    # 右眼
        (9, 10),                           # 嘴
        (11, 13), (13, 15),                # 左臂
        (15, 17), (15, 19), (15, 21),      # 左手
        (17, 19), (12, 14), (14, 16),      # 右臂
        (16, 18), (16, 20), (16, 22),      # 右手
        (18, 20), (11, 12),                # 肩膀
        (11, 23), (12, 24),                # 躯干兩側
        (23, 24),                          # 臀部
        (23, 25), (24, 26),                # 大腿
        (25, 27), (27, 29), (29, 31),      # 左腿
        (26, 28), (28, 30), (30, 32),      # 右腿
        (27, 31), (28, 32)                 # 腳
    ]
    
    # 初始化骨架線條
    lines = [ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0] for _ in POSE_CONNECTIONS]
    
    # 更新函數
    def update(frame):
        point = points[frame]
        scatter._offsets3d = (point[:, 0], point[:, 1], point[:, 2])
        
        for i, (start, end) in enumerate(POSE_CONNECTIONS):
            lines[i].set_data([point[start, 0], point[end, 0]], 
                              [point[start, 1], point[end, 1]])
            lines[i].set_3d_properties([point[start, 2], point[end, 2]])
        
        ax.set_title(f'{title} - Frame {frame}')
        return scatter, *lines
    
    # 創建動畫
    anim = FuncAnimation(fig, update, frames=len(points), interval=50, blit=False)
    
    # 顯示動並允許用戶調整視角
    plt.show(block=False)
    
    input("調整視角後，按 Enter 鍵保存動畫...")
    
    # 獲取當前視角
    elev, azim = ax.elev, ax.azim
    
    # 關閉互動視窗
    plt.close(fig)
    
    # 重新創建圖形和動畫，使用調整後的視角
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    
    scatter = ax.scatter([], [], [], s=20, c='r', alpha=0.6)
    lines = [ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0] for _ in POSE_CONNECTIONS]
    
    anim = FuncAnimation(fig, update, frames=len(points), interval=50, blit=False)
    
    # 保存為MP4文件
    writer = FFMpegWriter(fps=20.0, metadata=dict(artist='Unknown'), bitrate=11206)
    anim.save(filename, writer=writer)
    
    plt.close(fig)
    print(f"動畫已保存為 {filename}")


# 在主程序中调用此函数# 主程序
video_path_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camR\cam0-2.mp4"#左相機
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camR\cam0-2.mp4"
video_path_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camL\cam1-2.mp4"#右相機
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camL\cam1-walk.mp4"

all_points_3d = process_videos(video_path_left, video_path_right)
visualize_3d_animation(all_points_3d)
print(all_points_3d)


# 应用Savitzky-Golay滤���器
smoothed_points_3d = smooth_points_savgol(all_points_3d, window_size=7, polyorder=3)
# 在主程序中调用此函数
save_animation_as_mp4(all_points_3d, 'original_3d_animation.mp4', 'Original 3D Points')
save_animation_as_mp4(smoothed_points_3d, 'smoothed_3d_animation.mp4', 'Smoothed 3D Points')
## 指定不同的视角
#save_animation_as_mp4(smoothed_points_3d, 'smoothed_3d_animation.mp4', 'Smoothed 3D Points', elev=30, azim=-45)
print("动画已保存为 MP4 文件")
# 可视化比较
visualize_3d_animation_comparison(all_points_3d, smoothed_points_3d)
visualize_3d_animation_comparison1(all_points_3d, smoothed_points_3d, R, T)


import subprocess
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def get_video_info(video_path):
    command = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
    
    fps = eval(video_stream.get('avg_frame_rate', '0/0'))
    fps = fps if isinstance(fps, (int, float)) else 0
    
    bitrate = int(data['format'].get('bit_rate', 0)) // 1000
    duration = float(data['format'].get('duration', 0))
    
    return {
        'fps': fps,
        'bitrate': bitrate,
        'duration': duration,
        'frame_count': int(fps * duration)
    }
def sync_videos(video_path_3d, video_path_original, output_path):
    info_3d = get_video_info(video_path_3d)
    info_original = get_video_info(video_path_original)
    
    cap_3d = cv2.VideoCapture(video_path_3d)
    cap_original = cv2.VideoCapture(video_path_original)
    
    fps = min(info_3d['fps'], info_original['fps'])
    bitrate = min(info_3d['bitrate'], info_original['bitrate'])
    frame_count = min(info_3d['frame_count'], info_original['frame_count'])
    
    width_3d = int(cap_3d.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_3d = int(cap_3d.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_original = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_width = width_3d + width_original
    output_height = max(height_3d, height_original)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    for _ in range(frame_count):
        ret_3d, frame_3d = cap_3d.read()
        ret_original, frame_original = cap_original.read()
        
        if not ret_3d or not ret_original:
            break
        
        frame_3d = cv2.resize(frame_3d, (width_3d, output_height))
        frame_original = cv2.resize(frame_original, (width_original, output_height))
        
        combined_frame = np.hstack((frame_3d, frame_original))
        out.write(combined_frame)
    
    cap_3d.release()
    cap_original.release()
    out.release()
    print(f"同步视频已保存到 {output_path}")

'''def sync_videos(video_path_3d, video_path_original, output_path):
    info_3d = get_video_info(video_path_3d)
    info_original = get_video_info(video_path_original)
    
    cap_3d = cv2.VideoCapture(video_path_3d)
    cap_original = cv2.VideoCapture(video_path_original)
    
    fps = min(info_3d['fps'], info_original['fps'])
    bitrate = min(info_3d['bitrate'], info_original['bitrate'])
    frame_count = min(info_3d['frame_count'], info_original['frame_count'])
    
    # 设置输出视频的尺寸
    output_width = 1920  # 可以根据需要调整
    output_height = 540  # 可以根据需要调整
    single_video_width = output_width // 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    for _ in range(frame_count):
        ret_3d, frame_3d = cap_3d.read()
        ret_original, frame_original = cap_original.read()
        
        if not ret_3d or not ret_original:
            break
        
        # 调整两个视频帧的大小
        frame_3d = cv2.resize(frame_3d, (single_video_width, output_height))
        frame_original = cv2.resize(frame_original, (single_video_width, output_height))
        
        # 合并两个帧
        combined_frame = np.hstack((frame_3d, frame_original))
        
        # 添加帧数信息
        cv2.putText(combined_frame, f"Frame: {_+1}/{frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(combined_frame)
    
    cap_3d.release()
    cap_original.release()
    out.release()
    print(f"同步视频已保存到 {output_path}")'''

# 使用示例
video_path_3d = r"C:\Users\user\Desktop\20241018\smoothed_3d_animation.mp4"
video_path_original = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camR\cam0-2.mp4"

# 调用函数
sync_videos(video_path_3d, video_path_original, 'synchronized_videos.mp4')

def reconstruct_3d_from_image(image_path_left, image_path_right):
    img_left = cv2.imread(image_path_left)
    img_right = cv2.imread(image_path_right)
    points_3d = process_frame(img_left, img_right)
    return points_3d

def calculate_limb_lengths(points_3d):
    limb_connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 25), (25, 27), 
        (27,29),  (29,31),
        (24, 26), (26, 28), (28,30), (30,32), 
       
    ]
    lengths = [np.linalg.norm(points_3d[start] - points_3d[end]) for start, end in limb_connections]
    return np.array(lengths)

'''def optimize_points_gom(points, reference_lengths, previous_points=None):
    limb_connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]

    def apply_bone_length_constraint(points_3d, connections, tolerance=1
    ):
        constrained = np.copy(points_3d)
        for connection in connections:
            start, end = connection
            bone_length = np.linalg.norm(points_3d[start] - points_3d[end])
            if previous_points is not None:
                prev_length = np.linalg.norm(previous_points[start] - previous_points[end])
                if abs(bone_length - prev_length) / prev_length > tolerance:
                    mid_point = (points_3d[start] + points_3d[end]) / 2
                    direction = points_3d[end] - points_3d[start]
                    direction /= np.linalg.norm(direction)
                    constrained[start] = mid_point - direction * prev_length / 2
                    constrained[end] = mid_point + direction * prev_length / 2
        return constrained'''
    
from scipy.optimize import minimize

def optimize_points_gom(points, reference_lengths, previous_points=None):
    limb_connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), 
        (26, 28), (28,30),  (30,32),  (27,29),  (29,31),
    ]
    points = points.astype(np.float64)  # 使用高精度

    def apply_hard_constraints(points_3d, reference_lengths, iterations=5):
        constrained = np.copy(points_3d)
        for _ in range(iterations):
            for (start, end), ref_length in zip(limb_connections, reference_lengths):
                current_vector = constrained[end] - constrained[start]
                current_length = np.linalg.norm(current_vector)
                scale_factor = ref_length / current_length
                mid_point = (constrained[start] + constrained[end]) / 2
                direction = current_vector / current_length
                constrained[start] = mid_point - direction * ref_length / 2
                constrained[end] = mid_point + direction * ref_length / 2
        return constrained

    constrained_points = apply_hard_constraints(points, reference_lengths)
    initial_guess = constrained_points.flatten()

    def objective_function(point):
        point_reshaped = point.reshape(33, 3)
        error = sum((np.linalg.norm(point_reshaped[start] - point_reshaped[end]) - ref_length) ** 2 
                    for (start, end), ref_length in zip(limb_connections, reference_lengths))
        original_deviation = np.sum((point_reshaped - points) ** 2)
        return error + 0.1 * original_deviation

    def length_constraints(point):
        point_reshaped = point.reshape(33, 3)
        return [np.linalg.norm(point_reshaped[start] - point_reshaped[end]) - ref_length 
                for (start, end), ref_length in zip(limb_connections, reference_lengths)]

    cons = {'type': 'eq', 'fun': length_constraints}
    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=cons, 
                      options={'ftol': 1e-8, 'maxiter': 1000})

    # 最后再次应用硬约束
    final_points = apply_hard_constraints(result.x.reshape(33, 3), reference_lengths)
    return final_points

def process_videos(video_path_left, video_path_right, reference_image_left=None, reference_image_right=None):
    print(f"開始處理視頻: {video_path_left} 和 {video_path_right}")
    use_gom = reference_image_left is not None and reference_image_right is not None
    
    if use_gom:
        print("使用參考圖像進行GOM優化")
        reference_points = reconstruct_3d_from_image(reference_image_left, reference_image_right)
        reference_lengths = calculate_limb_lengths(reference_points)
    else:
        print("不使用GOM優化")
        reference_lengths = None

    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise ValueError("無法打開視頻文件")

    all_points_3d = []
    previous_points = None
    frame_count = 0

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            break

        points_3d = process_frame(frame_left, frame_right)

        if points_3d is None:
            if all_points_3d:
                points_3d = all_points_3d[-1]
            else:
                continue

        if use_gom:
            print(f"正在使用GOM優化第 {frame_count} 幀...")
            optimized_points_gom = optimize_points_gom(points_3d, reference_lengths, previous_points)
            all_points_3d.append(optimized_points_gom)
            previous_points = optimized_points_gom
        else:
            all_points_3d.append(points_3d)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已處理 {frame_count} 幀")

    cap_left.release()
    cap_right.release()

    print(f"視頻處理完成。共處理了 {frame_count} 幀")
    return np.array(all_points_3d)
def calculate_limb_length_variations(all_points_3d, reference_lengths):
    limb_connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), 
        (26, 28), (28,30), (30,32), (27,29),  (29,31), 
        (28,32),(27,31)
    ]
    
    limb_names = [
        "Shoulders", "Left Upper Arm", "Left Lower Arm", "Right Upper Arm", "Right Lower Arm",
        "Left Hip", "Right Hip", "Left Thigh", "Left Calf", "Right Thigh", "Right Calf"
    ]
    
    variations = []
    
    for i, (start, end) in enumerate(limb_connections):
        lengths = []
        for frame in all_points_3d:
            length = np.linalg.norm(frame[start] - frame[end])
            lengths.append(length)
        
        lengths = np.array(lengths)
        mean_length = np.mean(lengths)
        std_dev = np.std(lengths)
        variation_percentage = (std_dev / reference_lengths[i]) * 100
        
        variations.append({
            "limb": limb_names[i],
            "mean_length": mean_length,
            "std_dev": std_dev,
            "variation_percentage": variation_percentage
        })
    
    return variations
# 主程序
video_path_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camR\cam0-2.mp4"
video_path_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camL\cam1-2.mp4"
all_points_3d_original = process_videos(video_path_left, video_path_right)
# 使用視頻的第一幀作為參考圖像
cap_left = cv2.VideoCapture(video_path_left)
cap_right = cv2.VideoCapture(video_path_right)
ret_left, reference_image_left = cap_left.read()
ret_right, reference_image_right = cap_right.read()
cap_left.release()
cap_right.release()

# 保存參考圖像
cv2.imwrite("reference_image_left.jpg", reference_image_left)
cv2.imwrite("reference_image_right.jpg", reference_image_right)

reference_image_left_path = "reference_image_left.jpg"
reference_image_right_path = "reference_image_right.jpg"

all_points_3d = process_videos(video_path_left, video_path_right, reference_image_left_path, reference_image_right_path)

# 應用Savitzky-Golay濾波器
smoothed_points_3d = smooth_points_savgol(all_points_3d, window_size=7, polyorder=3)

# 保存動畫
save_animation_as_mp4(smoothed_points_3d, 'gom_optimized_3d_animation.mp4', 'GOM Optimized 3D Points')

print("GOM優化後的動畫已保存為 MP4 文件")
# 在主程序中添加以下代碼
# 在调用 process_videos 之前添加以下代码
reference_points = reconstruct_3d_from_image(reference_image_left_path, reference_image_right_path)
reference_lengths = calculate_limb_lengths(reference_points)
variations = calculate_limb_length_variations(smoothed_points_3d, reference_lengths)

print("\n肢段長度變異度:")
for var in variations:
    print(f"{var['limb']}:")
    print(f"  mean_length: {var['mean_length']:.2f}")
    print(f"  std_dev: {var['std_dev']:.2f}")
    print(f"  variation_percentage: {var['variation_percentage']:.2f}%")
    print()

# 可視化變異度
limb_names = [var['limb'] for var in variations]

variation_percentages = [var['variation_percentage'] for var in variations]

plt.figure(figsize=(12, 6))
plt.bar(limb_names, variation_percentages)
plt.title('Limb Length Variations')
plt.xlabel('limbs')
plt.ylabel('variation (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('limb_length_variations.png')
plt.show()



