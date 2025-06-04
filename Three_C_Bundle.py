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



# ---------------------------- 1. 單獨相機校正 ---------------------------- #

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.io import savemat

# Number of inner corners in the chessboard

nx = 9#水平方向的角点数
ny = 6#垂直方向的角点

# Termination criteria for corner sub-pixel refinement

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)#

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)

objp = np.zeros((ny*nx, 3), np.float32)#創建一個形狀為（ny*nx, 3）的零矩陣
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)#將矩陣的前兩列賦值為網格坐標

# Arrays to store object points and image points from all the images

objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in left image plane
imgpoints_center = []  # 2d points in center image plane
imgpoints_right = []  # 2d points in right image plane

# Load images
# 加載校正圖像
images_left = glob.glob(r"D:\20241112\CAM_LEFT\Cam2-*.jpeg")
#"D:\20241112\CAM_LEFT\Cam2-*.jpeg"

images_center = glob.glob(r"D:\20241112\CAM_MIDDLE\Cam1-*.jpeg")
#"D:\20241112\CAM_MIDDLE\Cam1-*.jpeg"

images_right = glob.glob(r"D:\20241112\CAM_RIGHT\Cam0-*.jpeg")
#"D:\20241112\CAM_RIGHT\Cam0-*.jpeg"


# 確保所有圖像列表的順序一致
images_left.sort()#
images_center.sort()
images_right.sort()

# Find chessboard corners for each image

for i in range(len(images_left)):
    img_left = cv2.imread(images_left[i])#讀取圖像
    img_center = cv2.imread(images_center[i])#讀取中間圖像
    img_right = cv2.imread(images_right[i])#讀取圖像

    # Find corners in the chessboard pattern
    ret_left, corners_left = cv2.findChessboardCorners(img_left, (nx, ny), None)#尋找左相機棋盤角點
    ret_center, corners_center = cv2.findChessboardCorners(img_center, (nx, ny), None)#尋找中間棋盤格角點
    ret_right, corners_right = cv2.findChessboardCorners(img_right, (nx, ny), None)#尋找右相機棋盤格角點
    # If corners are found, refine corner locations, add object points, image points, and draw corners
    #image:輸入的圖像
    #patternSize:棋盤格的內部角點數量，格式為（內部角點數量x,內部角點數量y）
    #flags:尋找角點的選項，默認為cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE

    print("iter : ", i)
    print("ret_left :", ret_left)
    print("ret_center :", ret_center)
    print("ret_right :", ret_right)
    print("--------------")

    if ret_left and ret_center and ret_right:#check if corners are found in all images
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

# Perform camera calibration for each camera
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, (1920,1200), None, None)
ret_center, mtx_center, dist_center, rvecs_center, tvecs_center = cv2.calibrateCamera(
    objpoints, imgpoints_center, (1920,1200), None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, (1920,1200), None, None)

print("ret_left:", ret_left)
print("ret_center:", ret_center)
print("ret_right:", ret_right)
print("mtx_left:\n", mtx_left)
print("mtx_center:\n", mtx_center)
print("mtx_right:\n", mtx_right)

# 保單相機校正結果
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
         image_size=(1920, 1200))

print("校準結果已保存到 single_camera_calibration.npz")

# 計算並顯示每個相機的��野 (FOV)
def calculate_fov(mtx, image_size):
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    fov_x = 2 * np.arctan(image_size[0] / (2 * fx)) * (180 / np.pi)
    fov_y = 2 * np.arctan(image_size[1] / (2 * fy)) * (180 / np.pi)
    return fov_x, fov_y

# 計算每個相機的FOV
fov_x_left, fov_y_left = calculate_fov(mtx_left, (1920, 1200))
fov_x_center, fov_y_center = calculate_fov(mtx_center, (1920, 1200))
fov_x_right, fov_y_right = calculate_fov(mtx_right, (1920, 1200))

print(f"左相機水平視野 (FOV_x): {fov_x_left} degrees")
print(f"左相機垂直視野 (FOV_y): {fov_y_left} degrees")
print(f"中間相機水平視野 (FOV_x): {fov_x_center} degrees")
print(f"中間相機垂直視野 (FOV_y): {fov_y_center} degrees")
print(f"右相機水平視野 (FOV_x): {fov_x_right} degrees")
print(f"右相機垂直視野 (FOV_y): {fov_y_right} degrees")

# 顯示去畸變結果
def show_undistorted_image(img_path, mtx, dist, title):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None
        
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w_roi, h_roi = roi
    cv2.rectangle(undistorted, (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 2)
    resized = cv2.resize(undistorted, (900, 700))
    cv2.imshow(title, resized)
    return undistorted

# 顯示每個相機的去畸變結果
undist_left = show_undistorted_image(images_left[0], mtx_left, dist_left, 'Undistorted Left Image')
undist_center = show_undistorted_image(images_center[0], mtx_center, dist_center, 'Undistorted Center Image')
undist_right = show_undistorted_image(images_right[0], mtx_right, dist_right, 'Undistorted Right Image')

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import glob
from scipy.optimize import least_squares

# ---------------------------- 1.外部参数校正（固定内参） ---------------------------- #

# 棋盘格参数
pattern_size = (9, 6)#棋盤格內角數，寬×高
square_size = 3.0  # 根据实际棋盘格方格尺寸（单位：mm）

# 准备物体点
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)#創建一個形狀為（ny*nx, 3）的零矩陣
objp[:, :2] = np.mgrid[0:pattern_size[0],
                       0:pattern_size[1]].T.reshape(-1, 2)#將矩陣的前兩列賦值為網格坐標
objp *= square_size#縮放的實際尺寸

# 設定圖像尺寸
image_size = (1920, 1200)  # 根据实际图像尺寸

# ---------------------------- 2. 讀取圖像並檢測角點 ---------------------------- #

# 讀取圖像路徑並排序
images_left = glob.glob(r"D:\20241112\DUAL_2\Cam2-*.jpeg")#左相機圖像路徑
#"D:\20241112\CAM_dual0\Cam0-*.jpeg"
images_center = glob.glob(r"D:\20241112\CAM_DUAL1\Cam1-*.jpeg")#中間相機圖像路徑
#"D:\20241112\CAM_DUAL1\Cam1-*.jpeg"
images_right = glob.glob(r"D:\20241112\CAM_dual0\Cam0-*.jpeg")#右相機圖像路徑
#"D:\20241112\DUAL_2\Cam2-*.jpeg"

#保三個路徑的圖像順序一致
images_left.sort()#排序
images_center.sort()#排序
images_right.sort()#排序

# 初始化列表
objpoints = []         # 3D點（世界坐標系）
imgpoints_left = []    # 左相機的2D點
imgpoints_center = []  # 中間相機的2D點
imgpoints_right = []   # 右相機的2D點

# 檢測角點並收集數據
for idx, (img_left_path, img_center_path, img_right_path) in enumerate(zip(images_left, images_center, images_right)):
    # 讀取圖像
    img_left = cv2.imread(img_left_path)
    img_center = cv2.imread(img_center_path)
    img_right = cv2.imread(img_right_path)
    
    # 轉換為灰度圖
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 尋找棋盤格角點
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (nx, ny), None)
    ret_center, corners_center = cv2.findChessboardCorners(gray_center, (nx, ny), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (nx, ny), None)

    print(f"處理第 {idx+1} 組圖像:")
    print(f"左相機角點檢測: {ret_left}")
    print(f"中間相機角點檢測: {ret_center}") 
    print(f"右相機角點檢測: {ret_right}")
    print("--------------")

    if ret_left and ret_center and ret_right:
        # 亞像素角點精確化
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_center = cv2.cornerSubPix(gray_center, corners_center, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        # 添加物體點和圖像點
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_center.append(corners_center)
        imgpoints_right.append(corners_right)

        # 可選:繪製並顯示角點
        cv2.drawChessboardCorners(img_left, (nx, ny), corners_left, ret_left)
        cv2.drawChessboardCorners(img_center, (nx, ny), corners_center, ret_center)
        cv2.drawChessboardCorners(img_right, (nx, ny), corners_right, ret_right)

        # 顯示檢測結果
        cv2.imshow('Left Camera Corners', cv2.resize(img_left, (900, 700)))
        cv2.imshow('Center Camera Corners', cv2.resize(img_center, (900, 700)))
        cv2.imshow('Right Camera Corners', cv2.resize(img_right, (900, 700)))
        cv2.waitKey(500)
    else:
        print(f"無法在第 {idx+1} 組圖像中檢測到所有角點，跳過此組")

cv2.destroyAllWindows()

# ---------------------------- 3. 准备内参和初始外参 ---------------------------- #

# 加载单相机校准结果
calib_data = np.load('single_camera_calibration.npz')
mtx_left = calib_data['mtx_left']
dist_left = calib_data['dist_left']
mtx_center = calib_data['mtx_center']
dist_center = calib_data['dist_center']
mtx_right = calib_data['mtx_right']
dist_right = calib_data['dist_right']

# 準備固定的內參和畸變參數列表
mtx_list = [mtx_left, mtx_center, mtx_right]#  內參列表
dist_list = [dist_left, dist_center, dist_right]#畸變參數列表

# 初始化外參列表
rvecs_left = []#左相機外參列表
tvecs_left = []#左相機平移向量列表
rvecs_center = []#中間相機外參列表
tvecs_center = []#中間相機平移向量列表
rvecs_right = []#右相機外參列表
tvecs_right = []#右相機平移向量列表

n_images = len(objpoints)#圖像數量

for i in range(n_images):
    # 計算左相機外部參數
    ret_left, rvec_left, tvec_left = cv2.solvePnP(
        objpoints[i], imgpoints_left[i], mtx_left, dist_left)#求解PnP問題
    rvecs_left.append(rvec_left.reshape(3))#添加左相機外參
    tvecs_left.append(tvec_left.reshape(3))#添加左相機平移向量

    # 計算中間相機外部參數
    ret_center, rvec_center, tvec_center = cv2.solvePnP(
        objpoints[i], imgpoints_center[i], mtx_center, dist_center)#求解PnP問題
    rvecs_center.append(rvec_center.reshape(3))#添加中間相機外參
    tvecs_center.append(tvec_center.reshape(3))#添加中間相機平移向量

    # 計算右相機外部參數
    ret_right, rvec_right, tvec_right = cv2.solvePnP(
        objpoints[i], imgpoints_right[i], mtx_right, dist_right)#求解右相機PnP問題
    rvecs_right.append(rvec_right.reshape(3))#添加右相機外參
    tvecs_right.append(tvec_right.reshape(3))#添加右相機平移向量

# ---------------------------- 4. 構建全局優化問題 ---------------------------- #

def pack_params(rvecs_left, tvecs_left,
                rvecs_center, tvecs_center,
                rvecs_right, tvecs_right):
    """
    將所有相機的旋轉向量和平移向量打包成一個一維數組
    
    參數:
    - rvecs_*: 各相機的旋轉向量列表
    - tvecs_*: 各相機的平移向量列表
    
    返回:
    - 包含所有外參的一維numpy數組
    """
    params = []
    # 依照相機順序打包旋轉向量和平移向量
    for rvecs, tvecs in [(rvecs_left, tvecs_left), (rvecs_center, tvecs_center), (rvecs_right, tvecs_right)]:
        for rvec, tvec in zip(rvecs, tvecs):
            params.extend(rvec.ravel())#將rvec和tvec展平
            params.extend(tvec.ravel())#添加平移向量
    return np.array(params)#返回參數數組

# 定義重投影誤差函數
def reprojection_error(params, n_cameras, n_points, n_images,
                      camera_indices, image_indices, point_indices,
                      points_2d, objpoints,
                      mtx_list, dist_list):
    """
    計算重投影誤差的優化版本
    
    參數:
    - params: 包含所有相機外參的一維數組
    - n_cameras: 相機數量
    - n_points: 棋盤格角點數量
    - n_images: 每個相機的圖像數量
    - camera_indices: 標識每個觀測點屬於哪個相機
    - image_indices: 標識每個觀測點屬於哪張圖像
    - point_indices: 標識每個觀測點是棋盤格上的哪個角點
    - points_2d: 所有觀測到的2D點坐標
    - objpoints: 棋盤格角點的3D坐標
    - mtx_list: 所有相機的內參矩陣列表
    - dist_list: 所有相機的畸變係數列表
    """
    error = []
    idx = 0
    rvecs = []
    tvecs = []
    
    # 從params中提取相機參數
    for _ in range(n_cameras * n_images):
        rvec = params[idx:idx+3].reshape(3, 1)
        tvec = params[idx+3:idx+6].reshape(3, 1)
        rvecs.append(rvec)
        tvecs.append(tvec)
        idx += 6

    # 計算每個觀測點的重投影誤差
    for i in range(len(points_2d)):
        camera_idx = camera_indices[i]
        image_idx = image_indices[i]
        point_idx = point_indices[i]
        
        # 獲取相機參數
        mtx = mtx_list[camera_idx]
        dist_coeffs = dist_list[camera_idx]
        pose_idx = camera_idx * n_images + image_idx
        rvec = rvecs[pose_idx]
        tvec = tvecs[pose_idx]
        
        # 獲取3D點和對應的2D觀測
        objp = objpoints[image_idx][point_idx]
        imgp = points_2d[i]
        
        # 計算投影點
        imgp_proj, _ = cv2.projectPoints(
            objp.reshape(1, 3), rvec, tvec, mtx, dist_coeffs)
        
        # 計算誤差
        error.append(imgp.ravel() - imgp_proj.ravel())
    
    return np.concatenate(error)

# 准备优化所需的数据
n_cameras = 3#相機數量
n_points = len(objp)#棋盤格角點數量

## 準備優化所需的數據
camera_indices = []
image_indices = []
point_indices = []
points_2d = []

for i in range(n_images):
    for j in range(n_points):
        # 左相機
        camera_indices.append(0)
        image_indices.append(i)
        point_indices.append(j)
        points_2d.append(imgpoints_left[i][j])

        # 中間相機
        camera_indices.append(1)
        image_indices.append(i)
        point_indices.append(j)
        points_2d.append(imgpoints_center[i][j])

        # 右相機
        camera_indices.append(2)
        image_indices.append(i)
        point_indices.append(j)
        points_2d.append(imgpoints_right[i][j])

# 初始化優化參數
x0 = pack_params(rvecs_left, tvecs_left,
                 rvecs_center, tvecs_center,
                 rvecs_right, tvecs_right)

# 執行全局優化
res = least_squares(
    reprojection_error, 
    x0,
    verbose=2,
    method='trf',  # trust region reflective algorithm
    loss='huber',  # 使用 Huber loss 減少離群值影響
    args=(n_cameras, n_points, n_images,
          camera_indices, image_indices, point_indices,
          points_2d, objpoints,
          mtx_list, dist_list)
)
'''
主要參數說明：
reprojection_error: 計算重投影誤差的函數
x0: 初始參數值
verbose=2: 顯示詳細的優化過程信息
args: 傳遞給重投影誤差函數的額外參數
n_cameras: 相機數量
n_points: 每個棋盤格的角點數量
n_images: 每個相機的圖像數量
camera_indices: 標識每個觀測點屬於哪個相機
points_2d: 實際觀測到的2D點坐標
objpoints: 棋盤格角點的3D坐標
mtx_list: 固定的相機內參列表
dist_list: 固定的相機畸變係數列表
這個優化過程會不斷調整外參數，直到找到使重投影誤差最小的參數組合。
'''

# ---------------------------- 5. 提取優化後的參數並計算外部參數 ---------------------------- #

# 將優化後的參數結果存儲在optimized_params中
# res.x包含了優化後的所有相機外參數(旋轉向量和平移向量)
optimized_params = res.x

def unpack_params(params, n_cameras, n_images):
    idx = 0
    rvecs = []
    tvecs = []
    for _ in range(n_cameras * n_images):
        rvec = params[idx:idx+3].reshape(3, 1)
        idx += 3
        tvec = params[idx:idx+3].reshape(3, 1)
        idx += 3
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs

# 从优化结果中提取外参
rvecs_opt, tvecs_opt = unpack_params(res.x, n_cameras, n_images)

# 内参和畸变参数保持不变
mtx_left_opt = mtx_left
dist_left_opt = dist_left
mtx_center_opt = mtx_center
dist_center_opt = dist_center
mtx_right_opt = mtx_right
dist_right_opt = dist_right

# 构建每个相机的外参矩阵
# 左相机的外参（第一个图像）
rvec_left = rvecs_opt[0]
tvec_left = tvecs_opt[0]
R_left, _ = cv2.Rodrigues(rvec_left)
T_left = tvec_left

# 中间相机的外参（第一个图像）
rvec_center = rvecs_opt[n_images]
tvec_center = tvecs_opt[n_images]
R_center, _ = cv2.Rodrigues(rvec_center)
T_center = tvec_center

# 右相机的外参（第一个图像）
rvec_right = rvecs_opt[2 * n_images]
tvec_right = tvecs_opt[2 * n_images]
R_right, _ = cv2.Rodrigues(rvec_right)
T_right = tvec_right

# 計算相對於中間相機的外參
# 左相機相對於中間相機
R_cl = R_center @ R_left.T  # 修正旋轉順序：先將點從左相機坐標系轉回世界坐標系，再轉到中間相機坐標系
T_cl = -R_cl @ T_left + T_center  # 修正平移計算
'''# 應該修改為
R_cl = R_center @ R_left.T    # 正確
R_cr = R_center @ R_right.T   # 正確'''
# 右相機相對於中間相機
R_cr = R_center @ R_right.T  # 修正旋轉順序
T_cr = -R_cr @ T_right + T_center  # 修正平移計算
'''T_cl = -R_cl @ T_left + T_center    # 正確
T_cr = -R_cr @ T_right + T_center   # 正確'''
# 保存結果
np.savez('triple_camera_calibration_global_fixed_intrinsics.npz',
         mtx_left=mtx_left_opt, dist_left=dist_left_opt,
         mtx_center=mtx_center_opt, dist_center=dist_center_opt,
         mtx_right=mtx_right_opt, dist_right=dist_right_opt,
         R_cl=R_cl, T_cl=T_cl,
         R_cr=R_cr, T_cr=T_cr)

print("全局三相機校准完成（固定內參），結果已保存到 triple_camera_calibration_global_fixed_intrinsics.npz")








#--------------------------------------TEST--------------------------------------#


import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import sys

# ---------------------------- 1. 加载三相机的校准参数 ---------------------------- #

# 加载校准参数
calibration_data = np.load('triple_camera_calibration_global_fixed_intrinsics.npz')
mtx_left = calibration_data['mtx_left']
dist_left = calibration_data['dist_left']
mtx_center = calibration_data['mtx_center']
dist_center = calibration_data['dist_center']
mtx_right = calibration_data['mtx_right']
dist_right = calibration_data['dist_right']

R_cl = calibration_data['R_cl']
T_cl = calibration_data['T_cl']
R_cr = calibration_data['R_cr']
T_cr = calibration_data['T_cr']

# 构建各相机的外部参数
R_center = np.eye(3)
T_center = np.zeros((3, 1))

# 修改左右相機的外參計算
R_left = R_cl.T @ R_center  # 修正：從中間相機坐標系到左相機坐標系
T_left = -R_left @ T_cl  # 修正：相應的平移向量

R_right = R_cr.T @ R_center  # 修正：從中間相機坐標系到右相機坐標系
T_right = -R_right @ T_cr  # 修正：相應的平移向量

# 构建投影矩阵
P_left = np.hstack((R_left, T_left))
P_center = np.hstack((R_center, T_center))
P_right = np.hstack((R_right, T_right))





# ---------------------------- 2. 读取图像并检测关键点 ---------------------------- #

# Paths to images
imggg1 = cv2.imread(r"D:\20241112\SUBCALI2\subcaliCam2-3.jpeg")     # 左相机图像'path/to/left_image.jpg'
imggg2 = cv2.imread(r"D:\20241112\SUBCALI1\subcaliCam1-3center.jpeg")   # 中间相机图像'path/to/center_image.jpg'
imggg3 = cv2.imread(r"D:\20241112\SUBCALI0\subcaliCam0-3right.jpeg")    # 右相机图像'path/to/right_image.jpg'
#"D:\20241112\SUBCALI0\subcaliCam0-3right.jpeg"
if imggg1 is None or imggg2 is None or imggg3 is None:
    print("无法读取所有图像，请检查路径")
    sys.exit(1)

# 初始化 MediaPipe PoseLandmarker
base_options = python.BaseOptions(model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# 将图像转换为 MediaPipe Image 对象
mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg1, cv2.COLOR_BGR2RGB))
mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg2, cv2.COLOR_BGR2RGB))
mp_image3 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg3, cv2.COLOR_BGR2RGB))

# 检测关键点
detection_result1 = detector.detect(mp_image1)
detection_result2 = detector.detect(mp_image2)
detection_result3 = detector.detect(mp_image3)

# 提取关键���列表
pose_landmarks_list1 = detection_result1.pose_landmarks
pose_landmarks_list2 = detection_result2.pose_landmarks
pose_landmarks_list3 = detection_result3.pose_landmarks

# 初始化关键点数组
pose1 = np.zeros((len(pose_landmarks_list1[0]), 2))
pose2 = np.zeros((len(pose_landmarks_list2[0]), 2))
pose3 = np.zeros((len(pose_landmarks_list3[0]), 2))

# 提取左相机关键点
for idx, landmark in enumerate(pose_landmarks_list1[0]):
    a = landmark.x
    b = landmark.y
    pose1[idx, 0] = imggg1.shape[1] * a
    pose1[idx, 1] = imggg1.shape[0] * b

# 提取中间相机关键点
for idx, landmark in enumerate(pose_landmarks_list2[0]):
    a = landmark.x
    b = landmark.y
    pose2[idx, 0] = imggg2.shape[1] * a
    pose2[idx, 1] = imggg2.shape[0] * b

# 提取右相机关键点
for idx, landmark in enumerate(pose_landmarks_list3[0]):
    a = landmark.x
    b = landmark.y
    pose3[idx, 0] = imggg3.shape[1] * a
    pose3[idx, 1] = imggg3.shape[0] * b

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
'''多视图三角测量的基本原理


每个相机通过其投影矩阵 P 将3D点 X 投影到2D图像平面上
投影方程: x = PX (其中x是2D点，X是3D点，P是3×4投影矩阵)


去畸变和归一化'''
# 对每个关键点进行三角测量
points_3d = []
num_points = pose1.shape[0]  # 假设所有相机检测到的关键点数量相同

for i in range(num_points):
    # 获取三个相机的2D点（去畸变并归一化）
    pts_2d = []
    # 去畸变并归一化左相机点
    pt1_norm = cv2.undistortPoints(pose1[i].reshape(1, 1, 2), mtx_left, dist_left).reshape(2)
    pts_2d.append(pt1_norm)

    # 去畸变并归一化中间相机点
    pt2_norm = cv2.undistortPoints(pose2[i].reshape(1, 1, 2), mtx_center, dist_center).reshape(2)
    pts_2d.append(pt2_norm)

    # 去畸变并归一化右相机点
    pt3_norm = cv2.undistortPoints(pose3[i].reshape(1, 1, 2), mtx_right, dist_right).reshape(2)
    pts_2d.append(pt3_norm)

    # 三个相机的投影矩阵（仅包含外参）
    proj_matrices = [P_left, P_center, P_right]
    # 三角测量
    X = triangulate_points_nviews(proj_matrices, pts_2d)
    points_3d.append(X)

points_3d = np.array(points_3d)

# 输出3D点坐标
for i, point in enumerate(points_3d):
    print(f"3D point {i + 1}: {point}")

# ---------------------------- 4. 重新投影和误差计算 ---------------------------- #

# 定义重投影函数
def reproject_points(points_3d, rvec, tvec, mtx, dist):
    projected_points, _ = cv2.projectPoints(points_3d,
                                            rvec,
                                            tvec,
                                            mtx,
                                            dist)
    projected_points = projected_points.reshape(-1, 2)
    return projected_points

# 左相机的旋转向量和平移向量
R_left_vec, _ = cv2.Rodrigues(R_left)
tvec_left = T_left.reshape(3)
'''# 使用从世界坐标系到相机坐标系的旋转向量和平移向量
rvec_left, _ = cv2.Rodrigues(R_left)
tvec_left = T_left'''
# 中间相机的旋转向量和平移向量
R_center_vec, _ = cv2.Rodrigues(R_center)
tvec_center = T_center.reshape(3)

# 右相机的旋转向量和平移向量
R_right_vec, _ = cv2.Rodrigues(R_right)
tvec_right = T_right.reshape(3)

# 重新投影到左相机
projected_points_left = reproject_points(points_3d, R_left_vec, tvec_left, mtx_left, dist_left)

# 重新投影到中间相机
projected_points_center = reproject_points(points_3d, R_center_vec, tvec_center, mtx_center, dist_center)

# 重新投影到右相机
projected_points_right = reproject_points(points_3d, R_right_vec, tvec_right, mtx_right, dist_right)

# 计算左相机的误差
errors_left = np.linalg.norm(pose1 - projected_points_left, axis=1)
mean_error_left = np.mean(errors_left)
print(f"Left camera reprojection error: {mean_error_left}")

# 计算中间相机的误差
errors_center = np.linalg.norm(pose2 - projected_points_center, axis=1)
mean_error_center = np.mean(errors_center)
print(f"Center camera reprojection error: {mean_error_center}")

# 计算右相机的误差
errors_right = np.linalg.norm(pose3 - projected_points_right, axis=1)
mean_error_right = np.mean(errors_right)
print(f"Right camera reprojection error: {mean_error_right}")

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

# 可视化左相机的重投影
visualize_reprojection(imggg1, pose1, projected_points_left, errors_left, "Left Camera Reprojection")

# 可视化中间相机的重投影
visualize_reprojection(imggg2, pose2, projected_points_center, errors_center, "Center Camera Reprojection")

# 可视化右相机��重投影
visualize_reprojection(imggg3, pose3, projected_points_right, errors_right, "Right Camera Reprojection")
# 将所有参数展开为一个一维数组
'''def pack_params(mtx_left, dist_left, rvecs_left, tvecs_left,
                mtx_center, dist_center, rvecs_center, tvecs_center,
                mtx_right, dist_right, rvecs_right, tvecs_right):
    params = []
    # 提取左相机内参
    params.append(mtx_left[0, 0])  # fx
    params.append(mtx_left[1, 1])  # fy
    params.append(mtx_left[0, 2])  # cx
    params.append(mtx_left[1, 2])  # cy
    params.extend(dist_left.ravel())  # 畸变参数

    # 提取中间相机内参
    params.append(mtx_center[0, 0])  # fx
    params.append(mtx_center[1, 1])  # fy
    params.append(mtx_center[0, 2])  # cx
    params.append(mtx_center[1, 2])  # cy
    params.extend(dist_center.ravel())  # 畸变参数

    # 提取右相机内参
    params.append(mtx_right[0, 0])  # fx
    params.append(mtx_right[1, 1])  # fy
    params.append(mtx_right[0, 2])  # cx
    params.append(mtx_right[1, 2])  # cy
    params.extend(dist_right.ravel())  # 畸变参数

    # 外参（旋转向量和平移向量）
    for rvecs, tvecs in [(rvecs_left, tvecs_left), (rvecs_center, tvecs_center), (rvecs_right, tvecs_right)]:
        for rvec, tvec in zip(rvecs, tvecs):
            params.extend(rvec.ravel())
            params.extend(tvec.ravel())

    return np.array(params)'''