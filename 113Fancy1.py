import cv2
import numpy as np
import sys
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

import subprocess
import json
from scipy.optimize import minimize






# ---------------------------- 1. 單獨相機校正 ---------------------------- #
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Number of inner corners in the chessboard
nx, ny = 9, 6

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# Prepare object points
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints_left = []
objpoints_right = []
imgpoints_left = []
imgpoints_right = []

# Load images
images_left = glob.glob(r"D:/calibration1029_orth/single/camR/Cam-1_*.jpg")
images_right = glob.glob(r"D:/calibration1029_orth/single/camL/Cam-0_*.jpg")

# Function to resize image
def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Set display size
display_width, display_height = 900, 700

# 此函數用於處理棋盤格校正圖像
def process_images(images, objpoints, imgpoints):
    # 遍歷每張校正圖像
    for fname in images:
        # 讀取圖像並轉換為灰度圖
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 在灰度圖中尋找棋盤格角點
        # nx, ny 為棋盤格內角點數量
        # ret 為是否成功找到所有角點的標誌
        # corners 為找到的角點坐標
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret:  # 如果成功找到所有角點
            # 將世界坐標系中的點加入 objpoints
            objpoints.append(objp)
            # 使用亞像素級別優化角點位置
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 將優化後的角點坐標加入 imgpoints
            imgpoints.append(corners2)
            
            # 在圖像上繪製找到的角點
            img_with_corners = cv2.drawChessboardCorners(img.copy(), (nx, ny), corners2, ret)
            
            # 調整圖像大小以便顯示
            img_resized = resize_image(img_with_corners, display_width, display_height)
            
            # 顯示帶有角點的圖像
            cv2.imshow('Chessboard Corners', img_resized)
            cv2.waitKey(500)  # 暫停500毫秒
    
    cv2.destroyAllWindows()
    return objpoints, imgpoints

# Process left and right images
objpoints_left, imgpoints_left = process_images(images_left, objpoints_left, imgpoints_left)
objpoints_right, imgpoints_right = process_images(images_right, objpoints_right, imgpoints_right)

# Perform camera calibration
def calibrate_camera(objpoints, imgpoints, image_size):
    # 設置校正標誌:
    # CALIB_RATIONAL_MODEL: 使用有理模型來處理畸變
    # CALIB_FIX_K3/K4/K5: 固定高階畸變係數為0,避免過擬合
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
    # 執行相機校正,返回:
    # ret: 重投影誤差
    # mtx: 相機內參矩陣 
    # dist: 畸變係數
    # rvecs: 旋轉向量
    # tvecs: 平移向量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags)
    return ret, mtx, dist, rvecs, tvecs

# Get image size
img = cv2.imread(images_left[0])#
image_size = img.shape[:2][::-1]  # width, height

# Print the number of images processed
print(f"Number of left images processed: {len(objpoints_left)}")
print(f"Number of right images processed: {len(objpoints_right)}")

# Calibrate both cameras
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = calibrate_camera(objpoints_left, imgpoints_left, image_size)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = calibrate_camera(objpoints_right, imgpoints_right, image_size)

def calc_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    errors = []
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        errors.append(error)
    mean_error = np.mean(errors)
    return errors, mean_error

# Calculate and print errors
errors_left, mean_error_left = calc_reprojection_errors(
    objpoints_left, imgpoints_left, rvecs_left, tvecs_left, mtx_left, dist_left
)
errors_right, mean_error_right = calc_reprojection_errors(
    objpoints_right, imgpoints_right, rvecs_right, tvecs_right, mtx_right, dist_right
)

print("Left camera mean reprojection error:", mean_error_left)
print("Right camera mean reprojection error:", mean_error_right)

for i, error in enumerate(errors_left):
    print(f"Left image {i+1} reprojection error: {error}")

for i, error in enumerate(errors_right):
    print(f"Right image {i+1} reprojection error: {error}")

def visualize_reprojection_errors(images, objpoints, imgpoints, rvecs, tvecs, mtx, dist, title):
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgpoints_proj, _ = cv2.projectPoints(objpoints[idx], rvecs[idx], tvecs[idx], mtx, dist)
        for point in imgpoints[idx]:
            img = cv2.circle(img, tuple(point.ravel().astype(int)), 5, (0, 0, 255), -1)
        for point in imgpoints_proj:
            img = cv2.circle(img, tuple(point.ravel().astype(int)), 5, (255, 0, 0), -1)
        error = cv2.norm(imgpoints[idx], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        cv2.putText(img, f'Reprojection Error: {error:.4f}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img_resized = resize_image(img, display_width, display_height)
        cv2.imshow(title, img_resized)
        key = cv2.waitKey(500)
        if key == 27:
            break
    cv2.destroyAllWindows()

# Visualize reprojection errors
visualize_reprojection_errors(
    images_left, objpoints_left, imgpoints_left, rvecs_left, tvecs_left, mtx_left, dist_left,
    'Left Camera Reprojection Error'
)
visualize_reprojection_errors(
    images_right, objpoints_right, imgpoints_right, rvecs_right, tvecs_right, mtx_right, dist_right,
    'Right Camera Reprojection Error'
)

# Set error threshold and filter data
error_threshold = 0.03

# Filter left camera data
filtered_objpoints_left = []
filtered_imgpoints_left = []
for i in range(len(errors_left)):
    if errors_left[i] < error_threshold:
        filtered_objpoints_left.append(objpoints_left[i])
        filtered_imgpoints_left.append(imgpoints_left[i])
    else:
        print(f"Excluding left image {i+1} with reprojection error: {errors_left[i]:.4f}")

# Filter right camera data
filtered_objpoints_right = []
filtered_imgpoints_right = []
for i in range(len(errors_right)):
    if errors_right[i] < error_threshold:
        filtered_objpoints_right.append(objpoints_right[i])
        filtered_imgpoints_right.append(imgpoints_right[i])
    else:
        print(f"Excluding right image {i+1} with reprojection error: {errors_right[i]:.4f}")

# Recalibrate with filtered data and update original variables
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = calibrate_camera(
    filtered_objpoints_left, filtered_imgpoints_left, image_size
)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = calibrate_camera(
    filtered_objpoints_right, filtered_imgpoints_right, image_size
)

# Calculate new reprojection errors
# Calculate new reprojection errors
errors_left, mean_error_left = calc_reprojection_errors(
    filtered_objpoints_left, filtered_imgpoints_left, rvecs_left, tvecs_left, mtx_left, dist_left
)
errors_right, mean_error_right = calc_reprojection_errors(
    filtered_objpoints_right, filtered_imgpoints_right, rvecs_right, tvecs_right, mtx_right, dist_right
)
# Print filtered calibration results
print("\nAfter filtering based on reprojection error:")
print(f"Number of left images used: {len(filtered_objpoints_left)}")
print(f"Number of right images used: {len(filtered_objpoints_right)}")
print(f"左相機平均重投影誤差: {mean_error_left:.4f}")
print(f"右相機平均重投影誤差: {mean_error_right:.4f}")

# Save calibration results
# Save calibration results with original variable names
np.savez('calibration_left.npz', mtx=mtx_left, dist=dist_left,
         rvecs=rvecs_left, tvecs=tvecs_left)
np.savez('calibration_right.npz', mtx=mtx_right, dist=dist_right,
         rvecs=rvecs_right, tvecs=tvecs_right)

# ---------------------------- 2. 雙目相機校正 ---------------------------- #
# 右相機部分
img_right = cv2.imread(images_right[0])
h_right, w_right = img_right.shape[:2]

newcameramtx_right, roi_right = cv2.getOptimalNewCameraMatrix(
    mtx_right, dist_right, (w_right, h_right), 1, (w_right, h_right))

undistorted_img_right = cv2.undistort(img_right, mtx_right, dist_right, None, newcameramtx_right)

x_right, y_right, w_roi_right, h_roi_right = roi_right
cv2.rectangle(undistorted_img_right, (x_right, y_right), 
              (x_right + w_roi_right, y_right + h_roi_right), (0, 255, 0), 2)

resized_img_right = cv2.resize(undistorted_img_right, (900, 700))

cv2.imshow('Undistorted Right Image with Valid ROI', resized_img_right)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 左相機部分
img_left = cv2.imread(images_left[0])  # 修改這裡
h_left, w_left = img_left.shape[:2]    # 修改這裡

newcameramtx_left, roi_left = cv2.getOptimalNewCameraMatrix(
    mtx_left, dist_left, (w_left, h_left), 1, (w_left, h_left))  # 修改這裡

undistorted_img_left = cv2.undistort(img_left, mtx_left, dist_left, None, newcameramtx_left)  # 修改這裡

x_left, y_left, w_roi_left, h_roi_left = roi_left  # 修改這裡
cv2.rectangle(undistorted_img_left, (x_left, y_left), 
              (x_left + w_roi_left, y_left + h_roi_left), (0, 255, 0), 2)  # 修改這裡

resized_img_left = cv2.resize(undistorted_img_left, (900, 700))  # 修改這裡

cv2.imshow('Undistorted Left Image with Valid ROI', resized_img_left)  # 修改這裡
cv2.waitKey(0)
cv2.destroyAllWindows()

# 計算視野角度部分也需要相應修改
fx_right = mtx_right[0, 0]
fy_right = mtx_right[1, 1]
cx_right = mtx_right[0, 2]
cy_right = mtx_right[1, 2]
image_size_right = (w_right, h_right)

fx_left = mtx_left[0, 0]  # 修改這裡
fy_left = mtx_left[1, 1]  # 修改這裡
cx_left = mtx_left[0, 2]  # 修改這裡
cy_left = mtx_left[1, 2]  # 修改這裡
image_size_left = (w_left, h_left)  # 修改這裡

# 計算視野角度
fov_x_right = 2 * np.arctan(image_size_right[0] / (2 * fx_right)) * (180 / np.pi)
fov_y_right = 2 * np.arctan(image_size_right[1] / (2 * fy_right)) * (180 / np.pi)

fov_x_left = 2 * np.arctan(image_size_left[0] / (2 * fx_left)) * (180 / np.pi)  # 修改這裡
fov_y_left = 2 * np.arctan(image_size_left[1] / (2 * fy_left)) * (180 / np.pi)  # 修改這裡

print(f"右相機水平視野 (FOV_x): {fov_x_right:.2f} degrees")
print(f"右相機垂直視野 (FOV_y): {fov_y_right:.2f} degrees")
print(f"左相機水平視野 (FOV_x): {fov_x_left:.2f} degrees")  # 修改這裡
print(f"左相機垂直視野 (FOV_y): {fov_y_left:.2f} degrees")  # 修改這裡
'''def detect_chessboard_corners(image, pattern_size=(9, 6)):
    # 1. 圖像預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 使用CLAHE（對比度受限自適應直方圖均衡化）增強對比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 3. 降噪處理
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 4. 自適應閾值處理
    binary = cv2.adaptiveThreshold(
        gray, 
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # 5. 使用增強的findChessboardCorners參數
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +    # 使用自適應閾值
        cv2.CALIB_CB_NORMALIZE_IMAGE +    # 對圖像進行歸一化
        cv2.CALIB_CB_FILTER_QUADS +       # 過濾錯誤的四邊形
        cv2.CALIB_CB_FAST_CHECK          # 快速檢查模式
    )
    
    ret, corners = cv2.findChessboardCorners(binary, pattern_size, flags)
    
    if ret:
        # 6. 使用改進的亞像素角點檢測
        criteria = (
            cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 
            100,    # 最大迭代次數
            1e-5    # 精度
        )
        
        # 使用較大的搜索窗口
        corners = cv2.cornerSubPix(
            gray, 
            corners, 
            (11, 11),  # 搜索窗口大小
            (-1, -1),  # 死區大小
            criteria
        )
        
        # 7. 添加可視化檢查（可選）
        debug_image = cv2.drawChessboardCorners(image.copy(), pattern_size, corners, ret)
        cv2.imshow('Detected Corners', debug_image)
        cv2.waitKey(1)
        
        return True, corners
    
    return False, None'''

'''def detect_chessboard_corners(image, pattern_size=(9, 6)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 增加預處理步驟
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊減少噪聲
    gray = cv2.equalizeHist(gray)  # 直方圖均衡化增強對比度
    
    # 使用更多的檢測標誌
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FILTER_QUADS +  # 添加此標誌以過濾假四邊形
        cv2.CALIB_CB_FAST_CHECK
    )
    
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    
    if ret:
        # 使用更精確的亞像素角點檢測參數
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners2
    else:
        return False, None'''
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
    
''''def detect_chessboard_corners(image, pattern_size=(9, 6)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                             cv2.CALIB_CB_FAST_CHECK)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners2
    else:
        return False, None'''
# 修改外部參數校正部分的角點檢測
def detect_chessboard_corners(image, pattern_size=(9, 6)):
    """統一的角點檢測函數，增加預處理和精確化參數"""
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 增加預處理步驟
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊減少噪聲
    gray = cv2.equalizeHist(gray)  # 直方圖均衡化增強對比度
    
    # 使用更多的檢測標誌
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +  # 使用自適應閾值
        cv2.CALIB_CB_NORMALIZE_IMAGE +  # 對圖像進行歸一化
        cv2.CALIB_CB_FILTER_QUADS +     # 過濾假四邊形
        cv2.CALIB_CB_FAST_CHECK         # 快速檢查模式
    )
    
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    
    if ret:
        # 使用更精確的亞像素角點檢測參數
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
            100,    # 增加最大迭代次數
            1e-6    # 提高精度要求
        )
        
        # 使用較大的搜索窗口和較小的死區
        refined_corners = cv2.cornerSubPix(
            gray, 
            corners,
            (5, 5),     #減小搜索窗口大小從 (11,11) 改為 (5,5)
                    #適合更小的棋盤格角點間距
            #減少相鄰角點的干擾
            (-1, -1),  # 死區大小
            criteria
        )
        return True, refined_corners
    
    return False, None


        

# 設定棋盤格的參數
pattern_size = (9, 6)  # 根據您的棋盤格尺寸修改
square_size = 3.0  # 根據實際棋盤格方格的尺寸修改

# 準備物體點
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# 讀取影像
images_left = glob.glob(r"D:\calibration1029_orth\dual\camR\Cam-1_*.jpg")#cam-0_1.jpg=left
#"D:\calibration1029_orth\dual\camL\Cam-0_1.jpg"
images_right = glob.glob(r"D:\calibration1029_orth\dual\camL\Cam-0_*.jpg")#cam-1_1.jpg=right
#"D:\calibration1029_orth\dual\camR\Cam-1_1.jpg"
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1013\dual\camR\Cam-1_1.jpg"
images_right.sort()

objpoints = []  # 3D點
imgpoints_left = []  # 左相機2D點
imgpoints_right = []  # 右相機2D點
# 假設我們已經有了以下變數:
# objpoints_single, imgpoints_left_single, imgpoints_right_single
# image_width, image_height
# nx, ny (棋盤格的內角點數)
# square_size (棋盤格方格的實際尺寸,單位為mm)
# dual_calibration_path_left, dual_calibration_path_right (雙目校正圖像的路徑)
for idx, (img_left_path, img_right_path) in enumerate(zip(images_left, images_right)):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    
    ret_left, corners_left = detect_chessboard_corners(img_left, pattern_size)
    ret_right, corners_right = detect_chessboard_corners(img_right, pattern_size)
    
    if ret_left and ret_right:
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        
        # 繪製並顯示角點
        cv2.drawChessboardCorners(img_left, pattern_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, pattern_size, corners_right, ret_right)
        
        plt.figure(figsize=(16, 8))
        plt.subplot(121), plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
        plt.title(f'Left Image {idx+1}'), plt.axis('off')
        plt.subplot(122), plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
        plt.title(f'Right Image {idx+1}'), plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"無法在圖片對 {idx+1} 中找到角點")
        print(f"左圖: {img_left_path}")
        print(f"右圖: {img_right_path}")
        print("跳過這對圖片\n")
# 修改顯示角點的部分
'''for idx, (img_left_path, img_right_path) in enumerate(zip(images_left, images_right)):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    
    ret_left, corners_left = detect_chessboard_corners(img_left, pattern_size)
    ret_right, corners_right = detect_chessboard_corners(img_right, pattern_size)
    
    if ret_left and ret_right:
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        
        # 調整角點顯示參數
        img_left_display = img_left.copy()
        img_right_display = img_right.copy()
        
        # 使用較小的圓點和線條來顯示角點
        cv2.drawChessboardCorners(
            img_left_display, 
            pattern_size, 
            corners_left,
            ret_left,
            (0, 255, 0)  # 綠色
        )
        cv2.drawChessboardCorners(
            img_right_display, 
            pattern_size, 
            corners_right,
            ret_right,
            (0, 255, 0)  # 綠色
        )
        
        # 調整顯示大小
        display_width = 900
        display_height = 700
        img_left_resized = cv2.resize(img_left_display, (display_width, display_height))
        img_right_resized = cv2.resize(img_right_display, (display_width, display_height))
        
        # 顯示結果
        cv2.imshow(f'Left Image {idx+1}', img_left_resized)
        cv2.imshow(f'Right Image {idx+1}', img_right_resized)
        key = cv2.waitKey(500)
        if key == 27:  # ESC鍵退出
            break

cv2.destroyAllWindows()'''


# 進行立體校正
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
retval, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, 
    mtx_left, dist_left, mtx_right, dist_right, 
    img_left.shape[:2][::-1], criteria=criteria, flags=flags)

print("Stereo calibration completed")
print("Calibration error:", retval)
print("Rotation matrix:")
print(R)
print("Translation vector:")
print(T)
P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = np.hstack([R, T.reshape((3, 1))])

#保存立體校正結果
np.savez('stereo_camera_calibration.npz',
         # 左相機內部參數
         mtx_left=mtx_left,
         dist_left=dist_left,
         # 右相機內部參數
         mtx_right=mtx_right,
         dist_right=dist_right,
         # 立體校正結果
         R=R,  # 旋轉矩陣
         T=T,  # 平移向量
         E=E,  # 本質矩陣
         F=F,  # 基礎矩陣
         # 投影矩陣
         P1=P1,
         P2=P2,
         # 其他相關參數
         image_size=img_left.shape[:2][::-1],
         stereo_calibration_error=retval)

print("立體校正結果已保存到 stereo_camera_calibration.npz")
# Paths to MediaPipe model and images for triangulation
imggg1 = cv2.imread(r"D:\calibration1029_orth\subcali\camR\Cam-1_1.jpg")#主left
#"D:\calibration1029_orth\subcali\camL\Cam-0_1.jpg"
imggg2 = cv2.imread(r"D:\calibration1029_orth\subcali\camL\Cam-0_1.jpg")#輔=right
#"D:\calibration1029_orth\subcali\camR\Cam-1_1.jpg"
base_options = python.BaseOptions(model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(

    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg1, cv2.COLOR_BGR2RGB))
mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imggg2, cv2.COLOR_BGR2RGB))

detection_result1 = detector.detect(mp_image1)
detection_result2 = detector.detect(mp_image2)

### mp 2D landmark ###

pose_landmarks_list1 = detection_result1.pose_landmarks
pose_landmarks_list2 = detection_result2.pose_landmarks


pose1 = np.zeros((len(pose_landmarks_list1[0]),2))
pose2 = np.zeros((len(pose_landmarks_list2[0]),2))

count1 = 0
count2 = 0

for landmark in pose_landmarks_list1[0]:
    a = landmark.x
    b = landmark.y
    pose1[count1,0] = int(1920*a)
    pose1[count1,1] = int(1200*b)#1200-1200b
    count1 = count1+1

for landmark in pose_landmarks_list2[0]:
    a = landmark.x
    b = landmark.y
    pose2[count2,0] = int(1920*a)
    pose2[count2,1] = int(1200*b)#1200-1200b貌似在做Y軸的反向
    count2 = count2+1

points_3d = []
for i in range(len(pose1)):
    # 确保输入点的形状正确
    pose_temp_1 = cv2.undistortPoints(pose1[i].reshape(1,1,2), mtx_left, dist_left, None, None)
    pose_temp_2 = cv2.undistortPoints(pose2[i].reshape(1,1,2), mtx_right, dist_right, None, None)
    
    # 执行三角测量
    pose_i = cv2.triangulatePoints(P1, P2, pose_temp_1, pose_temp_2)
    points_pose = cv2.convertPointsFromHomogeneous(pose_i.T)
    points_3d.append(points_pose[0][0])

# 将列表转换为 NumPy 数组
points_3d = np.array(points_3d)
# 左相机的旋转向量和平移向量
rvec_left = np.zeros(3)
tvec_left = np.zeros(3)

# 右相机的旋转向量和平移向量
rvec_right, _ = cv2.Rodrigues(R)
tvec_right = T
projected_points_left, _ = cv2.projectPoints(points_3d,
                                             rvec_left,
                                             tvec_left,
                                             mtx_left,
                                             dist_left)
projected_points_right, _ = cv2.projectPoints(points_3d,
                                              rvec_right,
                                              tvec_right,
                                              mtx_right,
                                              dist_right)
# 确保形状匹配
projected_points_left = projected_points_left.reshape(-1, 2)
projected_points_right = projected_points_right.reshape(-1, 2)


points_3d = []
for i in range(len(pose1)):
    # 确保输入点的形状正确
    pose_temp_1 = cv2.undistortPoints(pose1[i].reshape(1,1,2), mtx_left, dist_left, None, None)
    pose_temp_2 = cv2.undistortPoints(pose2[i].reshape(1,1,2), mtx_right, dist_right, None, None)
    
    # 执行三角测量
    pose_i = cv2.triangulatePoints(P1, P2, pose_temp_1, pose_temp_2)
    points_pose = cv2.convertPointsFromHomogeneous(pose_i.T)
    points_3d.append(points_pose[0][0])
    

# 将列表转换为 NumPy 数组
points_3d = np.array(points_3d)
# 输出3D点坐标
for i, point in enumerate(points_3d):
    print(f"3D point {i + 1}: {point}")
#左相机的旋转向量和平移向量
rvec_left = np.zeros(3)
tvec_left = np.zeros(3)

# 右相机的旋转向量和平移向量
rvec_right, _ = cv2.Rodrigues(R)
tvec_right = T
projected_points_left, _ = cv2.projectPoints(points_3d,
                                             rvec_left,
                                             tvec_left,
                                             mtx_left,
                                             dist_left)
projected_points_right, _ = cv2.projectPoints(points_3d,
                                              rvec_right,
                                              tvec_right,
                                              mtx_right,
                                              dist_right)
# 确保形状匹配
projected_points_left = projected_points_left.reshape(-1, 2)
projected_points_right = projected_points_right.reshape(-1, 2)

# 计算左相机的误差
errors_left = np.linalg.norm(pose1 - projected_points_left, axis=1)
mean_error_left = np.mean(errors_left)
print(f"left cmaera reprojection: {mean_error_left}")

# 计算右相机的误差
errors_right = np.linalg.norm(pose2 - projected_points_right, axis=1)
mean_error_right = np.mean(errors_right)
print(f"right camera reprojection: {mean_error_right}")

def visualize_reprojection(image, original_points, projected_points, window_name):
    image_copy = image.copy()
    for orig_pt, proj_pt in zip(original_points, projected_points):
        cv2.circle(image_copy, (int(orig_pt[0]), int(orig_pt[1])), 5, (0, 0, 255), -1)
        cv2.circle(image_copy, (int(proj_pt[0]), int(proj_pt[1])), 5, (0, 255, 0), -1)
    # 调整图像尺寸
    desired_size = (900, 700)  # (宽度，高度)
    # 使用INTER_AREA插值方法將圖像縮放到指定大小
    # image_copy: 原始圖像
    # desired_size: 目標尺寸 (寬度,高度)
    # interpolation: 使用INTER_AREA插值算法,適合縮小圖像
    image_resized = cv2.resize(image_copy, desired_size, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, image_resized)

# 调用可视化函数
visualize_reprojection(imggg1, pose1, projected_points_left, "left cmaera reprojection")
visualize_reprojection(imggg2, pose2, projected_points_right,"right camera reprojection")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate errors for the left camera
errors_left = np.linalg.norm(pose1 - projected_points_left, axis=1)
mean_error_left = np.mean(errors_left)
print(f"left camera reprojection: {mean_error_left}")

# Print individual errors for the left camera
print("left camera each joint reprojection:")
for i, error in enumerate(errors_left):
    print(f"joint {i + 1}: error = {error}")

# Calculate errors for the right camera
errors_right = np.linalg.norm(pose2 - projected_points_right, axis=1)
mean_error_right = np.mean(errors_right)
print(f"right camera reprojection: {mean_error_right}")

# Print individual errors for the right camera
print("right camera each joint reprojection:")
for i, error in enumerate(errors_right):
    print(f"joint {i + 1}: error = {error}")

def visualize_reprojection(image, original_points, projected_points, errors, window_name):
    # 目标显示尺寸
    desired_display_size = (900, 700)  # (宽度, 高度)
    
    # 计算缩放比例
    scale_x = desired_display_size[0] / image.shape[1]
    scale_y = desired_display_size[1] / image.shape[0]
    
    # 调整图像大小
    image_resized = cv2.resize(image, desired_display_size, interpolation=cv2.INTER_AREA)
    
    # 缩放关键点坐标
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

# 调用可视化函数
visualize_reprojection(imggg1, pose1, projected_points_left, errors_left, "Left Camera Reprojection")
visualize_reprojection(imggg2, pose2, projected_points_right, errors_right, "Right Camera Reprojection")




import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import matplotlib.pyplot as plt
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# 1. 导入必要的库# 已在上面导入

# 2. 加载相机参数
def load_camera_params(file_path):
    data = np.load(file_path)
    return data['mtx_left'], data['dist_left'], data['mtx_right'], data['dist_right'], data['R'], data['T']

# 请确保路径正确
mtx_left, dist_left, mtx_right, dist_right, R, T = load_camera_params('stereo_camera_calibration.npz')

# 3. 设置MediaPipe姿态估计模型
# 设置 MediaPipe 模型
base_options = python.BaseOptions(model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True,)
detector = vision.PoseLandmarker.create_from_options(options)

# 初始化 MediaPipe Pose
#mp_pose = mp.solutions.pose
#pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
'''mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    # 靜態圖像模式
    static_image_mode=False,      
    # False: 適用於視頻流，會使用追蹤來提高效率
    # True: 每一幀都進行完整檢測，更準確但更慢
    
    # 模型複雜度
    model_complexity=2,           
    # 0: 最輕量級模型，速度最快，準確度最低
    # 1: 中等複雜度
    # 2: 最複雜模型，最準確但最慢
    
    # 關鍵點平滑處理
    smooth_landmarks=True,        
    # True: 啟用時間序列平滑，減少抖動
    # False: 關閉平滑，獲得原始檢測結果
    
    # 人體分割
    enable_segmentation=True,     
    # True: 額外輸出人體分割遮罩，有助於提高準確性
    # False: 不進行分割，節省計算資源
    
    # 檢測置信度閾值
    min_detection_confidence=0.7, 
    # 範圍 0-1,越高要求越嚴格
    # 0.7 表示只有置信度>70%的檢測結果才會被接受
    
    # 追蹤置信度閾值
    min_tracking_confidence=0.7   
    # 範圍 0-1,越高要求越嚴格
    # 0.7 表示只有置信度>70%的追蹤結果才會被接受
)'''

# 4. 定义处理单帧的函数
def process_frame(frame_left, frame_right):
    try:
        # 使MediaPipe检测人体姿势
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
    except Exception as e:
        print(f"处理单帧时出现错误: {e}")
        return None

# 5. 定义三角测量函数
def triangulate_points(points_left, points_right, mtx_left, dist_left, mtx_right, dist_right, R, T):
    # 构建投影矩阵
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T.reshape(3, 1)))

    # 校正点
    points_left_undist = cv2.undistortPoints(points_left.reshape(-1, 1, 2), mtx_left, dist_left)
    points_right_undist = cv2.undistortPoints(points_right.reshape(-1, 1, 2), mtx_right, dist_right)

    # 三角测量
    points_4d_hom = cv2.triangulatePoints(P1, P2, points_left_undist, points_right_undist)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

    return points_3d

# 6. 定义处理视频的主循环（更新以支持GOM优化）
def process_videos(video_path_left, video_path_right):
    print(f"开始处理视频: {video_path_left} 和 {video_path_right}")

    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise ValueError("无法打开视频文件")

    all_points_3d = []
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

        all_points_3d.append(points_3d)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧")

    cap_left.release()
    cap_right.release()

    print(f"视频处理完成。共处理了 {frame_count} 帧")
    return np.array(all_points_3d)

# 7. 定义平滑函数
def smooth_points_savgol(points, window_size=7, polyorder=3):
    smoothed_points = np.zeros_like(points)
    for i in range(points.shape[1]):
        for j in range(3):  # x, y, z
            smoothed_points[:, i, j] = savgol_filter(points[:, i, j], window_size, polyorder, mode='nearest')
    return smoothed_points

# 8. 定义可视化函数（包括您提供的函数）
from matplotlib.animation import FuncAnimation
LIMB_CONNECTIONS = [
    (11, 12),  # 肩膀
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (11, 23), (12, 24),  # 肩膀到臀部
    (23, 24),  # 臀部
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # 左腿
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)   # 右腿
]
def visualize_3d_animation_comparison1(points1, points2, R, T, title1='Original', title2='Processed'):
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 计算所有点的范围
    all_points = np.vstack((points1.reshape(-1, 3), points2.reshape(-1, 3)))
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
        ax.set_ylabel('Z')  # 修改Y轴标签为Z
        ax.set_zlabel('Y')  # 修改Z轴标签为Y
    
    # 设置视角
    for ax in [ax1, ax2]:
        ax.view_init(elev=10, azim=-60)
    
    # 添加地板
    floor_y = 80
    x_floor = np.array([min_vals[0] - margin[0], max_vals[0] + margin[0]])
    z_floor = np.array([min_vals[2] - margin[2], max_vals[2] + margin[2]])
    X_floor, Z_floor = np.meshgrid(x_floor, z_floor)
    Y_floor = np.full(X_floor.shape, floor_y)
    
    for ax in [ax1, ax2]:
        ax.plot_surface(X_floor, Z_floor, Y_floor, alpha=0.2, color='gray')
    
    # 初始化散点图和柱状体
    scatter1 = ax1.scatter([], [], [], s=20, c='r', alpha=0.6)
    scatter2 = ax2.scatter([], [], [], s=20, c='b', alpha=0.6)
    cylinders1 = []
    cylinders2 = []
    
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
    ax1.quiver(*left_camera_pos, 0, 0, axis_length, color='g', label='Z_axis')
    ax1.quiver(*left_camera_pos, 0, axis_length, 0, color='b', label='Y_axis')
    ax2.quiver(*left_camera_pos, axis_length, 0, 0, color='r')
    ax2.quiver(*left_camera_pos, 0, 0, axis_length, color='g')
    ax2.quiver(*left_camera_pos, 0, axis_length, 0, color='b')
    
    # 右相机坐标轴
    ax1.quiver(*right_camera_pos, *(R @ np.array([axis_length, 0, 0])), color='r')
    ax1.quiver(*right_camera_pos, *(R @ np.array([0, 0, axis_length])), color='g')
    ax1.quiver(*right_camera_pos, *(R @ np.array([0, axis_length, 0])), color='b')
    ax2.quiver(*right_camera_pos, *(R @ np.array([axis_length, 0, 0])), color='r')
    ax2.quiver(*right_camera_pos, *(R @ np.array([0, 0, axis_length])), color='g')
    ax2.quiver(*right_camera_pos, *(R @ np.array([0, axis_length, 0])), color='b')
    
    # 添加图例
    ax1.legend()
    ax2.legend()
    
    # 修改创建圆柱体/锥体的函数
    def create_limb(p1, p2, radius1, radius2, ax, color):
        v = p2 - p1
        mag = np.linalg.norm(v)
        v = v / mag
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, mag, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        t, theta = np.meshgrid(t, theta)
        
        # 使用线性插值创建从radius1到radius2的过渡
        radius = radius1 + (radius2 - radius1) * (t / mag)
        
        X, Y, Z = [p1[i] + v[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        return ax.plot_surface(X, Y, Z, color=color, alpha=0.7)

    # 为每个连接创建锥体
    for _ in LIMB_CONNECTIONS:
        cylinders1.append(None)
        cylinders2.append(None)
    
    # 更新函数
    def update(frame):
        points_set1 = points1[frame]
        points_set2 = points2[frame]
        
        scatter1._offsets3d = (points_set1[:, 0], points_set1[:, 1], points_set1[:, 2])
        scatter2._offsets3d = (points_set2[:, 0], points_set2[:, 1], points_set2[:, 2])
        
        # 更新柱状体的位置
        for i, (start, end) in enumerate(LIMB_CONNECTIONS):
            if cylinders1[i] is not None:
                cylinders1[i].remove()
            if cylinders2[i] is not None:
                cylinders2[i].remove()
            
            p1 = points_set1[start]
            p2 = points_set1[end]
            cylinders1[i] = create_limb(p1, p2, 1, 0.5, ax1, 'r')
            
            p1 = points_set2[start]
            p2 = points_set2[end]
            cylinders2[i] = create_limb(p1, p2, 1, 0.5, ax2, 'b')
        
        ax1.set_title(f'{title1} - Frame {frame}')
        ax2.set_title(f'{title2} - Frame {frame}')
        
        return scatter1, scatter2, *cylinders1, *cylinders2
    
    # 建动画
    anim = FuncAnimation(fig, update, frames=len(points1), interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()

# 9. 定义GOM优化函数（按照您的逻辑）
from scipy.optimize import minimize

def optimize_points_gom(points_sequence, reference_lengths):
    start_time = time.time()  # 开始计时整个序列的优化
    optimized_sequence = []
    for i, frame_points in enumerate(points_sequence):
        optimized_frame = optimize_frame_gom(frame_points, reference_lengths)
        optimized_sequence.append(optimized_frame)
        if (i + 1) % 10 == 0:  # 每10帧打印一进度
            print(f"已完成 {i+1}/{len(points_sequence)} 帧的优化")
    end_time = time.time()  # 结束计时
    total_optimization_time = end_time - start_time
    print(f"整个序列优化时间: {total_optimization_time:.4f} 秒")
    return np.array(optimized_sequence)

def optimize_frame_gom(points, reference_lengths):
    start_time = time.time()
    points = points.astype(np.float64)

    def apply_hard_constraints(points_3d, reference_lengths, iterations=5):
        constrained = np.copy(points_3d)
        for _ in range(iterations):
            for (start, end), ref_length in zip(LIMB_CONNECTIONS, reference_lengths):
                current_vector = constrained[end] - constrained[start]
                current_length = np.linalg.norm(current_vector)
                if current_length == 0:
                    continue
                scale_factor = ref_length / current_length
                mid_point = (constrained[start] + constrained[end]) / 2
                direction = current_vector / current_length
                constrained[start] = mid_point - direction * ref_length / 2
                constrained[end] = mid_point + direction * ref_length / 2
        return constrained

    constrained_points = apply_hard_constraints(points, reference_lengths)
    initial_guess = constrained_points.flatten()

    def objective_function(point):
        point_reshaped = point.reshape(-1, 3)
        error = sum(
            (np.linalg.norm(point_reshaped[start] - point_reshaped[end]) - ref_length) ** 2
            for (start, end), ref_length in zip(LIMB_CONNECTIONS, reference_lengths)
        )
        original_deviation = np.sum((point_reshaped - points) ** 2)
        return error + 0.1 * original_deviation

    def length_constraints(point):
        point_reshaped = point.reshape(-1, 3)
        return [np.linalg.norm(point_reshaped[start] - point_reshaped[end]) - ref_length 
                for (start, end), ref_length in zip(LIMB_CONNECTIONS, reference_lengths)]

    cons = {'type': 'eq', 'fun': length_constraints}
    result = minimize(
        objective_function,
        initial_guess,
        method='SLSQP',
        constraints=cons,
        options={'ftol': 1e-8, 'maxiter': 1000}
    )

    final_points = apply_hard_constraints(result.x.reshape(-1, 3), reference_lengths)
    end_time = time.time()
    optimization_time = end_time - start_time
    print(f"单帧GOM优化时间: {optimization_time:.4f} 秒")
    return final_points

# 10. 计算肢段长度的变异度
def calculate_limb_length_variations(all_points_3d, reference_lengths):
    limb_definitions = {
        "shoulders": (11, 12),
        "left_upper_arm": (11, 13), "left_lower_arm": (13, 15),
        "right_upper_arm": (12, 14), "right_lower_arm": (14, 16),
        "left_shoulder_to_hip": (11, 23), "right_shoulder_to_hip": (12, 24),
        "left_thigh": (23, 25), "left_calf": (25, 27),
        "left_ankle_to_heel": (27, 29), "left_heel_to_toe": (29, 31),
        "left_ankle_to_toe": (27, 31),
        "right_thigh": (24, 26), "right_calf": (26, 28),
        "right_ankle_to_heel": (28, 30), "right_heel_to_toe": (30, 32),
        "right_ankle_to_toe": (28, 32)
    }

    variations = []
    for i, (name, (start, end)) in enumerate(limb_definitions.items()):
        lengths = [np.linalg.norm(frame[start] - frame[end]) for frame in all_points_3d]
        lengths = np.array(lengths)
        mean_length = np.mean(lengths)
        std_dev = np.std(lengths)
        variation_percentage = (std_dev / reference_lengths[i]) * 100 if i < len(reference_lengths) else 0

        variations.append({
            "limb": name,
            "mean_length": mean_length,
            "std_dev": std_dev,
            "variation_percentage": variation_percentage
        })

    return variations

# 辅助函数：从图像中重建3D点
def reconstruct_3d_from_image(image_path_left, image_path_right):
    img_left = cv2.imread(image_path_left)
    img_right = cv2.imread(image_path_right)
    points_3d = process_frame(img_left, img_right)
    return points_3d

# 辅助函数：计算肢段长度
def calculate_limb_lengths(points_3d):
    return np.array([np.linalg.norm(points_3d[start] - points_3d[end]) for start, end in LIMB_CONNECTIONS])
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# 可視化使用的骨架連接（包含所有連接）
LIMB_CONNECTIONS_VIS = [
    # 臉部連接
    (0, 1), (1, 2), (2, 3), (3, 7),  # 右臉輪廓
    (0, 4), (4, 5), (5, 6), (6, 8),  # 左臉輪廓
    (9, 10),  # 嘴巴
    (0, 9), (0, 10),  # 連接臉部到嘴巴
    
    # 身體骨架（與GOM相同的部分）
    (11, 12),  # 肩膀
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (11, 23), (12, 24),  # 肩膀到臀部
    (23, 24),  # 臀部
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # 左腿
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # 右腿
]

def visualize_camera_pixel_world(points_3d, mtx_left, mtx_right, dist_left, dist_right, R, T, video_path_left, video_path_right):
    # 打開視頻
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("無法打開視頻文件")
        return
        
    # 獲取視頻的幀率和寬高
    fps_left = cap_left.get(cv2.CAP_PROP_FPS)
    fps_right = cap_right.get(cv2.CAP_PROP_FPS)
    print(f"左相機幀率: {fps_left}, 右相機幀率: {fps_right}")
    
    width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用實際視頻幀率設置動畫間隔（毫秒）
    interval = 1000 / fps_left  # 轉換為毫秒

    # 創建圖形
    fig = plt.figure(figsize=(15, 5))
    ax_3d = fig.add_subplot(131, projection='3d')
    ax_left = fig.add_subplot(132)
    ax_right = fig.add_subplot(133)
    
    # 禁用互動模式，避免重複更新
    plt.ioff()
    
    # 初始化3D視圖的骨架條
    lines_3d = []
    for _ in LIMB_CONNECTIONS_VIS:
        line_3d, = ax_3d.plot([], [], [], 'g-', alpha=0.8, linewidth=2)
        lines_3d.append(line_3d)
    
    # 初始化2D視圖的骨架線條
    lines_left = []
    lines_right = []
    for _ in LIMB_CONNECTIONS_VIS:
        line_left, = ax_left.plot([], [], 'g-', alpha=0.8, linewidth=2)
        line_right, = ax_right.plot([], [], 'g-', alpha=0.8, linewidth=2)
        lines_left.append(line_left)
        lines_right.append(line_right)
    
    # 設置3D圖的範圍
    all_points = points_3d.reshape(-1, 3)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    margin = 0.1 * range_vals
    
    # 設置3D圖的顯示範圍
    ax_3d.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax_3d.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    ax_3d.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])

    # 設置相機位置
    left_camera_pos = np.array([0, 0, 0])
    right_camera_pos = -R.T @ T.flatten()

    # 計算合適的坐標軸長度
    all_points = points_3d.reshape(-1, 3)
    range_vals = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    axis_length = np.min(range_vals) * 0.1  # 使用最小範圍的10%作為軸長度
    
    # 繪製左相機位置和坐標軸
    ax_3d.scatter(*left_camera_pos, color='r', s=100, label='Left Camera')
    ax_3d.quiver(*left_camera_pos, axis_length, 0, 0, color='r', alpha=0.5, label='X')
    ax_3d.quiver(*left_camera_pos, 0, axis_length, 0, color='g', alpha=0.5, label='Y')
    ax_3d.quiver(*left_camera_pos, 0, 0, axis_length, color='b', alpha=0.5, label='Z')
    
    # 繪製右相機位置和坐標軸（修正旋轉）
    ax_3d.scatter(*right_camera_pos, color='b', s=100, label='Right Camera')
    # 使用R而不是R.T
    ax_3d.quiver(*right_camera_pos, *(R @ np.array([axis_length, 0, 0])), color='r', alpha=0.5)
    ax_3d.quiver(*right_camera_pos, *(R @ np.array([0, axis_length, 0])), color='g', alpha=0.5)
    ax_3d.quiver(*right_camera_pos, *(R @ np.array([0, 0, axis_length])), color='b', alpha=0.5)
    
    ax_3d.legend()

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    # 設置2D圖的標題
    ax_left.set_title('Left Camera View')
    ax_right.set_title('Right Camera View')

    # 初始化顯示
    im_left = ax_left.imshow(np.zeros((height_left, width_left, 3)))
    im_right = ax_right.imshow(np.zeros((height_right, width_right, 3)))
    scatter_left = ax_left.scatter([], [], c='r', s=5)
    scatter_right = ax_right.scatter([], [], c='r', s=5)

    def update(frame):
        # 讀取視頻幀
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            return None
            
        # 更新視頻幀
        im_left.set_array(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
        im_right.set_array(cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))
        
        points = points_3d[frame]
        
        # 更新3D骨架
        for i, (start, end) in enumerate(LIMB_CONNECTIONS_VIS):
            if start < len(points) and end < len(points):
                # 3D骨架
                lines_3d[i].set_data_3d([points[start, 0], points[end, 0]],
                                      [points[start, 1], points[end, 1]],
                                      [points[start, 2], points[end, 2]])
        
        # 投影到左右相機
        points_2d_left, _ = cv2.projectPoints(points, np.eye(3), np.zeros(3), mtx_left, dist_left)
        points_2d_right, _ = cv2.projectPoints(points, R, T, mtx_right, dist_right)
        
        points_2d_left = points_2d_left.reshape(-1, 2)
        points_2d_right = points_2d_right.reshape(-1, 2)
        
        # 更新2D投影點
        scatter_left.set_offsets(points_2d_left)
        scatter_right.set_offsets(points_2d_right)
        
        # 更新2D骨架
        for i, (start, end) in enumerate(LIMB_CONNECTIONS_VIS):
            if start < len(points_2d_left) and end < len(points_2d_left):
                # 左相機2D骨架
                lines_left[i].set_data([points_2d_left[start, 0], points_2d_left[end, 0]],
                                     [points_2d_left[start, 1], points_2d_left[end, 1]])
                
                # 右相機2D骨架
                lines_right[i].set_data([points_2d_right[start, 0], points_2d_right[end, 0]],
                                      [points_2d_right[start, 1], points_2d_right[end, 1]])
        
        return [im_left, im_right, scatter_left, scatter_right] + lines_3d + lines_left + lines_right
    
    anim = FuncAnimation(fig, update, 
                        frames=min(len(points_3d), int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))),
                        interval=interval,
                        blit=True,
                        repeat=True)
    
    plt.show()

    # 釋放視頻資源
    cap_left.release()
    cap_right.release()
def save_data(filename, **kwargs):
    np.savez(filename, **kwargs)
    print(f"数据已保存至 {filename}")

def load_data(filename):
    if os.path.exists(filename):
        print(f"正在从 {filename} 加载数据...")
        return np.load(filename)
    return None

def visualize_3d_animation_comparison_skeleton(points1, points2, R, T, title1='Sequence 1', title2='Sequence 2'):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 設置標題
    ax1.set_title(title1)
    ax2.set_title(title2)
    
    # 初始化兩個視圖的骨架線條
    lines1 = []
    lines2 = []
    for _ in LIMB_CONNECTIONS_VIS:
        line1, = ax1.plot([], [], [], 'g-', alpha=0.8, linewidth=2)
        line2, = ax2.plot([], [], [], 'g-', alpha=0.8, linewidth=2)
        lines1.append(line1)
        lines2.append(line2)
    
    # 設置相機位置
    left_camera_pos = np.array([0, 0, 0])
    right_camera_pos = -R.T @ T.flatten()  # 修正右相機位置
    
    # 計算合適的坐標軸長度
    all_points = np.concatenate([points1.reshape(-1, 3), points2.reshape(-1, 3)])
    range_vals = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    axis_length = np.min(range_vals) * 0.1  # 使用最小範圍的10%作為軸長度
    
    # 在兩個視圖中都顯示相機位置和坐標軸
    for ax in [ax1, ax2]:
        # 左相機
        ax.scatter(*left_camera_pos, color='r', s=100, label='Left Camera')
        ax.quiver(*left_camera_pos, axis_length, 0, 0, color='r', alpha=0.5, label='X')
        ax.quiver(*left_camera_pos, 0, axis_length, 0, color='g', alpha=0.5, label='Y')
        ax.quiver(*left_camera_pos, 0, 0, axis_length, color='b', alpha=0.5, label='Z')
        
        # 右相機
        ax.scatter(*right_camera_pos, color='b', s=100, label='Right Camera')
        ax.quiver(*right_camera_pos, *(R.T @ np.array([axis_length, 0, 0])), color='r', alpha=0.5)
        ax.quiver(*right_camera_pos, *(R.T @ np.array([0, axis_length, 0])), color='g', alpha=0.5)
        ax.quiver(*right_camera_pos, *(R.T @ np.array([0, 0, axis_length])), color='b', alpha=0.5)
        
        ax.legend()
    
    # 設置顯示範圍
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    margin = 0.1 * range_vals
    
    for ax in [ax1, ax2]:
        ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
        ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
        ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    def update(frame):
        # 清除之前的線條
        for line in lines1 + lines2:
            line.set_data_3d([], [], [])
        
        # 更新第一個序列的骨架
        points = points1[frame]
        for i, (start, end) in enumerate(LIMB_CONNECTIONS_VIS):
            if start < len(points) and end < len(points):
                lines1[i].set_data_3d([points[start, 0], points[end, 0]],
                                    [points[start, 1], points[end, 1]],
                                    [points[start, 2], points[end, 2]])
        
        # 更新第二個序列的骨架
        points = points2[frame]
        for i, (start, end) in enumerate(LIMB_CONNECTIONS_VIS):
            if start < len(points) and end < len(points):
                lines2[i].set_data_3d([points[start, 0], points[end, 0]],
                                    [points[start, 1], points[end, 1]],
                                    [points[start, 2], points[end, 2]])
        
        return lines1 + lines2
    
    anim = FuncAnimation(fig, update, 
                        frames=min(len(points1), len(points2)),
                        interval=50,
                        blit=True)
    
    plt.show()

def calculate_reprojection_error(points_3d, points_2d_left, points_2d_right, mtx_left, dist_left, mtx_right, dist_right, R, T):
    """
    计算3D点重投影到2D图像平面的误差
    
    参数:
    points_3d: 3D点坐标 (N, 3)
    points_2d_left/right: 实际观察到的2D点坐标 (N, 2)
    mtx_left/right: 相机内参矩阵
    dist_left/right: 畸变系数
    R, T: 右相机相对于左相机的旋转矩阵和平移向量
    
    返回:
    mean_error_left: 左相机平均重投影误差
    mean_error_right: 右相机平均重投影误差
    """
    # 投影3D点到左相机
    projected_left, _ = cv2.projectPoints(points_3d, np.eye(3), np.zeros(3), mtx_left, dist_left)
    projected_left = projected_left.reshape(-1, 2)
    
    # 投影3D点到右相机
    projected_right, _ = cv2.projectPoints(points_3d, R, T, mtx_right, dist_right)
    projected_right = projected_right.reshape(-1, 2)
    
    # 计算欧氏距离
    error_left = np.sqrt(np.sum((projected_left - points_2d_left) ** 2, axis=1))
    error_right = np.sqrt(np.sum((projected_right - points_2d_right) ** 2, axis=1))
    
    return np.mean(error_left), np.mean(error_right)

def process_frame_with_2d(frame_left, frame_right):
    """
    处理单帧并返回3D点和对应的2D点
    """
    try:
        # 使用MediaPipe检测人体姿势
        mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
        mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))

        detection_result_left = detector.detect(mp_image_left)
        detection_result_right = detector.detect(mp_image_right)

        # 提取2D关键点
        points_2d_left = np.array([[landmark.x * frame_left.shape[1], landmark.y * frame_left.shape[0]] 
                                  for landmark in detection_result_left.pose_landmarks[0]])
        points_2d_right = np.array([[landmark.x * frame_right.shape[1], landmark.y * frame_right.shape[0]] 
                                   for landmark in detection_result_right.pose_landmarks[0]])

        # 三维重建
        points_3d = triangulate_points(points_2d_left, points_2d_right, mtx_left, dist_left, mtx_right, dist_right, R, T)

        return points_3d, points_2d_left, points_2d_right
    except Exception as e:
        print(f"处理单帧时出现错误: {e}")
        return None, None, None

def process_videos_with_reprojection(video_path_left, video_path_right):
    """
    process videos and calculate reprojection error
    """
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise ValueError("无法打开视频文件")

    all_points_3d = []
    all_points_2d_left = []
    all_points_2d_right = []
    reprojection_errors = []
    frame_count = 0

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            break

        points_3d, points_2d_left, points_2d_right = process_frame_with_2d(frame_left, frame_right)

        if points_3d is not None:
            all_points_3d.append(points_3d)
            all_points_2d_left.append(points_2d_left)
            all_points_2d_right.append(points_2d_right)

            # 计算当前帧的重投影误差
            error_left, error_right = calculate_reprojection_error(
                points_3d, points_2d_left, points_2d_right,
                mtx_left, dist_left, mtx_right, dist_right, R, T
            )
            reprojection_errors.append((error_left, error_right))

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧")

    cap_left.release()
    cap_right.release()

    # 计算平均重投影误差
    mean_errors = np.mean(reprojection_errors, axis=0)
    print(f"mean reprojection error - left: {mean_errors[0]:.2f}pixel, right: {mean_errors[1]:.2f}pixel")

    return np.array(all_points_3d), np.array(all_points_2d_left), np.array(all_points_2d_right), np.array(reprojection_errors)

# 主程序
def main():
    # 定义文件名
    raw_data_file = '1033_raw_3d_points.npz'
    smoothed_data_file = '1033_smoothed_3d_points.npz'
    gom_data_file = '1033_gom_optimized_points_3d.npz'
    reference_data_file = '1033_reference_data.npz'

    # 视频路径
    video_path_left = r"D:\calibration1029_orth\cam1-4.mp4"#左相機
#"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camR\cam0-2.mp4"
    video_path_right = r"D:\calibration1029_orth\cam0-4.mp4"#右相機

    # 处理原始视频数据并计算重投影误差
    raw_data = load_data(raw_data_file)
    if raw_data is None or 'points_2d_left' not in raw_data.files:  # 检查是否存在2D点数据
        print("未找到完整的原始数据，开始处理视频...")#
        all_points_3d_original, points_2d_left, points_2d_right, reprojection_errors = process_videos_with_reprojection(
            video_path_left, video_path_right
        )
        save_data(raw_data_file, 
                 all_points_3d_original=all_points_3d_original,
                 points_2d_left=points_2d_left,
                 points_2d_right=points_2d_right,
                 reprojection_errors=reprojection_errors)
    else:
        try:
            all_points_3d_original = raw_data['all_points_3d_original']
            points_2d_left = raw_data['points_2d_left']
            points_2d_right = raw_data['points_2d_right']
            reprojection_errors = raw_data['reprojection_errors']
            
            # 显示平均重投影误差
            mean_errors = np.mean(reprojection_errors, axis=0)
            print(f"mean reprojection error - left: {mean_errors[0]:.2f}pixel, right: {mean_errors[1]:.2f}pixel")
        except KeyError as e:
            print(f"数据文件缺少必要的键: {e}")
            print("重新处理视频...")
            all_points_3d_original, points_2d_left, points_2d_right, reprojection_errors = process_videos_with_reprojection(
                video_path_left, video_path_right
            )
            save_data(raw_data_file, 
                     all_points_3d_original=all_points_3d_original,
                     points_2d_left=points_2d_left,
                     points_2d_right=points_2d_right,
                     reprojection_errors=reprojection_errors)

    # 处理参考数据
    reference_data = load_data(reference_data_file)
    if reference_data is None:
        print("未找到参考数据，开始处理参考图像...")
        # 使用视频的第一帧作为参考图像
        cap_left = cv2.VideoCapture(video_path_left)
        cap_right = cv2.VideoCapture(video_path_right)
        ret_left, reference_image_left = cap_left.read()
        ret_right, reference_image_right = cap_right.read()
        cap_left.release()
        cap_right.release()

        # 保存参考图像
        cv2.imwrite("reference_image_left.jpg", reference_image_left)
        cv2.imwrite("reference_image_right.jpg", reference_image_right)

        reference_image_left_path = "reference_image_left.jpg"
        reference_image_right_path = "reference_image_right.jpg"

        # 重建参考点并计算参考长度
        reference_points = reconstruct_3d_from_image(reference_image_left_path, reference_image_right_path)
        reference_lengths = calculate_limb_lengths(reference_points)
        save_data(reference_data_file, reference_points=reference_points, reference_lengths=reference_lengths)
    else:
        reference_points = reference_data['reference_points']
        reference_lengths = reference_data['reference_lengths']

    # 应用Savitzky-Golay滤波器
    smoothed_data = load_data(smoothed_data_file)
    if smoothed_data is None:
        print("未找到平滑后的数据，开始应用Savitzky-Golay滤波器...")
        smoothed_points_3d = smooth_points_savgol(all_points_3d_original, window_size=7, polyorder=3)
        save_data(smoothed_data_file, smoothed_points_3d=smoothed_points_3d)
    else:
        smoothed_points_3d = smoothed_data['smoothed_points_3d']

    # GOM优化
    gom_data = load_data(gom_data_file)
    if gom_data is None:
        print("未找到GOM优化后的数据，开始GOM优化...")
        gom_optimized_points_3d = optimize_points_gom(smoothed_points_3d, reference_lengths)
        save_data(gom_data_file, gom_optimized_points_3d=gom_optimized_points_3d)
    else:
        gom_optimized_points_3d = gom_data['gom_optimized_points_3d']

    # 可视化
    visualize_camera_pixel_world(
        gom_optimized_points_3d,  # 3D點雲數據
        mtx_left, mtx_right,      # 相機內參
        dist_left, dist_right,    # 畸變係數
        R, T,                     # 外參矩陣
        video_path_left,          # 左相機視頻路徑
        video_path_right          # 右相機視頻路徑
    )
    
    # 在main函數中
# 可視化原始與平滑後的數據比較
    visualize_3d_animation_comparison_skeleton(
    all_points_3d_original, 
    smoothed_points_3d, 
    R, T, 
    title1='Original', 
    title2='Smoothed'
)

# 可視化平滑後與GOM優化後的數據比較
    visualize_3d_animation_comparison_skeleton(
    smoothed_points_3d, 
    gom_optimized_points_3d, 
    R, T, 
    title1='Smoothed', 
    title2='GOM Optimized'
)
    # 可视化原始与平滑后的数据比较
    #visualize_3d_animation_comparison1(all_points_3d_original, smoothed_points_3d, R, T, 
                                       #title1='Original', title2='Smoothed')

    # 可视化平滑后与GOM优化后的数据比较
    #visualize_3d_animation_comparison1(smoothed_points_3d, gom_optimized_points_3d, R, T, 
                                       #title1='Smoothed', title2='GOM Optimized')

    # 计算变异度
    variations = calculate_limb_length_variations(gom_optimized_points_3d, reference_lengths)
    for var in variations:
        print(f"{var['limb']}:")
        print(f"  Mean Length = {var['mean_length']:.2f}")
        print(f"  Std Dev = {var['std_dev']:.2f}")
        print(f"  Variation Percentage = {var['variation_percentage']:.2f}%\n")



if __name__ == "__main__":
    main()