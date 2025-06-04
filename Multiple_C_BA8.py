import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys

# =======================================
# =========== 0. 通用參數設定 ===========
# =======================================

# 棋盤格內角點數 (nx, ny)，請改成自己實驗用的數量
nx = 9  # 水平方向角點數
ny = 6  # 垂直方向角點數

# 亞像素優化停止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# 每個角點在世界座標系的 3D 位置（尚未放大成真實尺寸），這裡假設每格尺寸=1
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# 影像大小 (w, h)，你可依實際影像調整
IMAGE_SIZE = (1920, 1200)

# 假設你有 8 台相機，路徑如下(請自行修改成你的實際路徑)
# Cam0 ~ Cam7，各自存放了多張棋盤格影像
cam_paths = [
    r"D:\CALIB_CAM0\Cam0-*.jpeg",
    r"D:\CALIB_CAM1\Cam1-*.jpeg",
    r"D:\CALIB_CAM2\Cam2-*.jpeg",
    r"D:\CALIB_CAM3\Cam3-*.jpeg",
    r"D:\CALIB_CAM4\Cam4-*.jpeg",
    r"D:\CALIB_CAM5\Cam5-*.jpeg",
    r"D:\CALIB_CAM6\Cam6-*.jpeg",
    r"D:\CALIB_CAM7\Cam7-*.jpeg",
]

# =======================================
# =========== 1. 單獨相機校正 ===========
# =======================================

def single_camera_calibration(cam_image_paths, nx, ny, objp, criteria, image_size):
    """
    給定某一台相機的棋盤格影像路徑，執行單相機校正。
    回傳該相機的內參、畸變和PNP解算時的旋轉、平移向量(若有需求)。
    """
    # 排序影像路徑
    cam_image_paths.sort()

    # 用來存放該相機所有影像對應的3D-2D關係
    objpoints = []  # 3D object points in world space
    imgpoints = []  # 2D points in image plane

    for i, path in enumerate(cam_image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"無法讀取影像: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 尋找棋盤角點
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            # 亞像素精細化
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)

    # 執行單相機校正
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints

print("開始對 8 台相機分別進行單獨校正...")

all_mtx = []
all_dist = []
all_objpoints = []   # 也可視情況保留，用於後續統一整理
all_imgpoints = []
num_cameras = 8

for cam_idx in range(num_cameras):
    cam_images = glob.glob(cam_paths[cam_idx])
    if len(cam_images) == 0:
        print(f"相機 {cam_idx} 路徑下沒有任何影像，請檢查路徑!")
        sys.exit(1)

    ret, mtx, dist, rvecs, tvecs, objpts_i, imgpts_i = single_camera_calibration(
        cam_images, nx, ny, objp, criteria, IMAGE_SIZE
    )
    print(f"Camera {cam_idx} single calib RMS error = {ret}, mtx = \n{mtx}")
    all_mtx.append(mtx)
    all_dist.append(dist)
    all_objpoints.append(objpts_i)
    all_imgpoints.append(imgpts_i)

# 統一存成 npz（僅示範）
np.savez('eight_cameras_single_calib.npz',
         all_mtx=all_mtx,
         all_dist=all_dist,
         image_size=IMAGE_SIZE)
print("8 台相機單獨校正完成, 結果已存為 eight_cameras_single_calib.npz")


# =============================================
# =========== 2. 外參固定內參全局優化 ===========
# =============================================
"""
思路：
1) 先把所有相機對應的棋盤格影像「對應關係」對齊。即同一個世界上棋盤格 Pose, 
   它在 8 張照片中的角點 (nx*ny個) 要對應同一組 objp。
2) 先以 solvePnP 為每台相機、每張影像求初始外參 (rvec, tvec)。
3) 整理所有外參 + 觀測的 2D / 3D，丟給 least_squares 做全局優化。
   (內參與畸變保持固定不動，僅優化外參)
"""

# 你需要將對應同一塊棋盤格板的 8 張影像放一起做對齊。
# 下面僅示範核心架構，假設我們已有 “對應” 的影像路徑。實際上需要確保攝影同步或檔名對齊。

# 假設我們有 N 張「同一塊棋盤」的八視角影像:
# cam0_images[i], cam1_images[i], ..., cam7_images[i] (第 i 組)
# 以下僅示範如何組裝外參參數到 least_squares 與擷取2D觀測

# 範例: 先載入單相機校正結果
calib_data = np.load('eight_cameras_single_calib.npz', allow_pickle=True)
all_mtx = calib_data['all_mtx']
all_dist = calib_data['all_dist']
image_size = tuple(calib_data['image_size'])

# 這裡假設我們有 8 組對應影像路徑(每一組包含同一個棋盤在8視角的照片)。
# 例如:
cam0_images = sorted(glob.glob(r"D:\GLOBAL\CAM0\*.jpeg"))
cam1_images = sorted(glob.glob(r"D:\GLOBAL\CAM1\*.jpeg"))
cam2_images = sorted(glob.glob(r"D:\GLOBAL\CAM2\*.jpeg"))
cam3_images = sorted(glob.glob(r"D:\GLOBAL\CAM3\*.jpeg"))
cam4_images = sorted(glob.glob(r"D:\GLOBAL\CAM4\*.jpeg"))
cam5_images = sorted(glob.glob(r"D:\GLOBAL\CAM5\*.jpeg"))
cam6_images = sorted(glob.glob(r"D:\GLOBAL\CAM6\*.jpeg"))
cam7_images = sorted(glob.glob(r"D:\GLOBAL\CAM7\*.jpeg"))
# 注意：必須保證 i 索引對應的是同一個棋盤拍攝

all_images_list = [
    cam0_images,
    cam1_images,
    cam2_images,
    cam3_images,
    cam4_images,
    cam5_images,
    cam6_images,
    cam7_images
]

# 用來存放所有影像對應的 objpoints 與 imgpoints
objpoints_global = []
imgpoints_global = [[] for _ in range(num_cameras)]  # 8 個相機，每個相機都對應很多張圖

n_boards = len(cam0_images)  # 這裡假設所有相機都有相同張數

for board_idx in range(n_boards):
    # 同一組 8 張影像 (同一塊棋盤)
    # 把對應的3D objp 放入 objpoints_global
    # 此處假設所有棋盤格都相同(尺寸、角點數)
    # objp (ny*nx, 3)
    # 可根據實際方格大小再做縮放
    # 假設 square_size=3.0 => objp * 3.0
    if board_idx == 0:
        # 第一塊棋盤可以直接 append
        objpoints_global.append(objp)
    else:
        # 也可以每塊棋盤都視作相同 objp(或有不同的世界座標)
        objpoints_global.append(objp)

    # 針對 8 台相機，分別讀圖找角點
    for cam_idx in range(num_cameras):
        img_path = all_images_list[cam_idx][board_idx]
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_global[cam_idx].append(corners_refined)
        else:
            # 若其中某台相機沒成功偵測到角點, 你可選擇捨棄這塊棋盤.
            # 簡化起見，這裡直接 pass
            pass

# 先用 solvePnP 針對每台相機、每張圖像估初始外參
rvecs_init = [[] for _ in range(num_cameras)]
tvecs_init = [[] for _ in range(num_cameras)]

for cam_idx in range(num_cameras):
    for i in range(n_boards):
        # solvePnP
        objp_i = objpoints_global[i]
        imgp_i = imgpoints_global[cam_idx][i]
        ret, rvec, tvec = cv2.solvePnP(objp_i, imgp_i, all_mtx[cam_idx], all_dist[cam_idx])
        rvecs_init[cam_idx].append(rvec.reshape(3, 1))
        tvecs_init[cam_idx].append(tvec.reshape(3, 1))

# ------ 組裝成 least_squares 需要的參數 ------
# pack / unpack
def pack_params(rvecs_list, tvecs_list):
    # rvecs_list, tvecs_list: shape=[num_cameras][n_boards]
    # 依順序打包
    param_vec = []
    for c_idx in range(num_cameras):
        for b_idx in range(n_boards):
            rv = rvecs_list[c_idx][b_idx].ravel()
            tv = tvecs_list[c_idx][b_idx].ravel()
            param_vec.extend(rv)
            param_vec.extend(tv)
    return np.array(param_vec)

def unpack_params(params, num_cameras, n_boards):
    rvecs_out = [[] for _ in range(num_cameras)]
    tvecs_out = [[] for _ in range(num_cameras)]
    idx = 0
    for c_idx in range(num_cameras):
        for b_idx in range(n_boards):
            rvec = params[idx: idx + 3].reshape(3, 1)
            tvec = params[idx + 3: idx + 6].reshape(3, 1)
            idx += 6
            rvecs_out[c_idx].append(rvec)
            tvecs_out[c_idx].append(tvec)
    return rvecs_out, tvecs_out

def reprojection_error(params, num_cameras, n_boards,
                       all_mtx, all_dist, 
                       objpoints_global, imgpoints_global):
    # 先拆包
    rvecs_list, tvecs_list = unpack_params(params, num_cameras, n_boards)

    total_error = []
    for c_idx in range(num_cameras):
        mtx = all_mtx[c_idx]
        dist = all_dist[c_idx]
        for b_idx in range(n_boards):
            rvec = rvecs_list[c_idx][b_idx]
            tvec = tvecs_list[c_idx][b_idx]
            # 3D-2D
            objp_i = objpoints_global[b_idx]
            imgp_i = imgpoints_global[c_idx][b_idx]

            projected, _ = cv2.projectPoints(objp_i, rvec, tvec, mtx, dist)
            diff = imgp_i.reshape(-1, 2) - projected.reshape(-1, 2)
            total_error.append(diff.ravel())
    return np.concatenate(total_error)

# 初始參數
x0 = pack_params(rvecs_init, tvecs_init)

# 全局優化
res = least_squares(
    reprojection_error,
    x0,
    method='trf',
    verbose=2,
    args=(num_cameras, n_boards,
          all_mtx, all_dist,
          objpoints_global, imgpoints_global)
)

optimized_params = res.x
rvecs_opt, tvecs_opt = unpack_params(optimized_params, num_cameras, n_boards)

# 這裡僅示範存前幾張(或第一張)的外參做示例
# 若你要計算「某台相機」對「參考相機(如第0台)」的外參，則可自行再做坐標轉換
np.savez('eight_cameras_global_fixed_intrinsics.npz',
         all_mtx=all_mtx,
         all_dist=all_dist,
         rvecs=rvecs_opt,
         tvecs=tvecs_opt)
print("八相機全局外部參數優化完成, 結果儲存於 eight_cameras_global_fixed_intrinsics.npz")


# ====================================================
# =========== 3. 測試：多視角三角測量 (8 相機) ===========
# ====================================================
"""
此處示範：假設我們已載入八相機的 內參(all_mtx, all_dist) & 外參(rvecs_opt, tvecs_opt)，
以及想用 8 張影像(同一瞬間拍攝)來估算某些特徵點(例如人臉、骨架關節等)的3D座標。

示例中依舊用 MediaPipe Pose 做範例。你可替換成任何關鍵點偵測方式。
重點在於：對應好 8 張圖中的同一個特徵點 (u,v)，再用多視角三角化。
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 假設我們要測 8 張圖, 路徑:
test_img_paths = [
    r"D:\TEST_8V\cam0_test.jpg",
    r"D:\TEST_8V\cam1_test.jpg",
    r"D:\TEST_8V\cam2_test.jpg",
    r"D:\TEST_8V\cam3_test.jpg",
    r"D:\TEST_8V\cam4_test.jpg",
    r"D:\TEST_8V\cam5_test.jpg",
    r"D:\TEST_8V\cam6_test.jpg",
    r"D:\TEST_8V\cam7_test.jpg",
]

# 載入校正
calib_data = np.load('eight_cameras_global_fixed_intrinsics.npz', allow_pickle=True)
all_mtx = calib_data['all_mtx']
all_dist = calib_data['all_dist']
rvecs_opt = calib_data['rvecs']  # shape=[num_cameras][n_boards]
tvecs_opt = calib_data['tvecs']

# 若只想取「第一張板(第0塊棋盤)」對應的外參當作各相機的(世界->相機)Pose，可示範:
# 你可自行調整想用哪一個 board_idx
board_idx_for_test = 0

# 取得對應相機的R、T
def getRT_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

# 建立投影矩陣 P = K [R | t]
def build_projection_matrix(K, R, t):
    RT = np.hstack([R, t])
    return K @ RT

camera_RT = []
proj_mats = []
for cam_idx in range(num_cameras):
    rvec = rvecs_opt[cam_idx][board_idx_for_test]
    tvec = tvecs_opt[cam_idx][board_idx_for_test]
    R, t = getRT_from_rvec_tvec(rvec, tvec)
    camera_RT.append((R, t))
    P = build_projection_matrix(all_mtx[cam_idx], R, t)
    proj_mats.append(P)

# --- 使用 MediaPipe Pose 偵測 8 張圖的骨架 ---
base_options = python.BaseOptions(model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# 這裡假設每張圖都能偵測到一樣多的關鍵點
eight_view_kpts = []  # shape=[num_points, 8, 2]，每個關鍵點在8視角的 2D 座標
n_landmarks = 33  # MediaPipe Pose 預設 33 個關鍵點

for cam_idx in range(num_cameras):
    img_path = test_img_paths[cam_idx]
    img = cv2.imread(img_path)
    if img is None:
        print(f"無法讀取圖像: {img_path}")
        sys.exit(1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    pose_landmarks_list = detection_result.pose_landmarks

    if not pose_landmarks_list:
        print(f"MediaPipe 在第 {cam_idx} 台相機的影像未偵測到任何人！")
        sys.exit(1)

    # 假設只偵測到 1 個人 pose_landmarks_list[0]，有 33 個點
    landmarks2d = []
    for lmk in pose_landmarks_list[0]:
        x_2d = lmk.x * img.shape[1]  # 轉成像素座標
        y_2d = lmk.y * img.shape[0]
        landmarks2d.append([x_2d, y_2d])

    # 去畸變(若要更精準，可使用 cv2.undistortPoints，再乘回 K)
    # 但為了和投影矩陣對應，我們可直接先把 (u,v,1) 投影到相機座標，再做三角。
    # 這裡簡化示範，僅把 2D 座標保存
    if cam_idx == 0:
        # 初始化eight_view_kpts (n_landmarks, 8, 2)
        eight_view_kpts = np.zeros((n_landmarks, num_cameras, 2), dtype=np.float32)
    for i_lmk in range(n_landmarks):
        eight_view_kpts[i_lmk, cam_idx] = landmarks2d[i_lmk]

# --- 定義多視角三角測量函式 (線性最小平方) ---
def linear_triangulate_nviews(proj_mats, points_2d):
    """
    proj_mats: list of shape=[num_views], each P is 3x4
    points_2d: list of 2D coordinates, shape=[num_views, 2]
    回傳: 3D點 (x, y, z)
    """
    num_views = len(proj_mats)
    A = []
    for i in range(num_views):
        x, y = points_2d[i]
        P = proj_mats[i]
        # 公式： x * p3 - p1 = 0,  y * p3 - p2 = 0
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)  # shape=(2*num_views, 4)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]
    return X[:3]

# --- 對 33 個關鍵點做三角測量 ---
all_3d_points = []
for i_lmk in range(n_landmarks):
    # 收集 8 相機 2D
    points_2d_i = []
    for cam_idx in range(num_cameras):
        pt_uv = eight_view_kpts[i_lmk, cam_idx]
        points_2d_i.append(pt_uv)
    # 做三角
    X_3d = linear_triangulate_nviews(proj_mats, points_2d_i)
    all_3d_points.append(X_3d)

all_3d_points = np.array(all_3d_points)
print("=== 8 相機三角測量結果 ===")
for i_lmk, pt3d in enumerate(all_3d_points):
    print(f"Landmark {i_lmk} => 3D: {pt3d}")


# ====================================================
# =========== 4. 重新投影誤差 (可選) ==================
# ====================================================
"""
若要看每台相機的重投影誤差, 可以使用:
cv2.projectPoints(某些3D點, rvec, tvec, K, dist)
和原始2D做比較。
這裡僅示範對某一 landmark 做計算。
"""

def compute_reproj_error(pts3d, pts2d, rvec, tvec, K, dist):
    """
    pts3d: shape=(N, 3)
    pts2d: shape=(N, 2)
    """
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.sqrt(np.sum((pts2d - proj)**2, axis=1))
    return err.mean()

for cam_idx in range(num_cameras):
    rvec = rvecs_opt[cam_idx][board_idx_for_test]
    tvec = tvecs_opt[cam_idx][board_idx_for_test]
    K = all_mtx[cam_idx]
    dist = all_dist[cam_idx]
    # 取 33 個 3D 點對應 2D
    pts2d_all = eight_view_kpts[:, cam_idx, :]  # shape=(33, 2)
    err_mean = compute_reproj_error(all_3d_points, pts2d_all, rvec, tvec, K, dist)
    print(f"Camera {cam_idx} reprojection error = {err_mean:.3f} pixels")

print("=== 完成: 八相機校正、外參優化、三角重建與重投影誤差示例 ===")
