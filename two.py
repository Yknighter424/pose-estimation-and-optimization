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