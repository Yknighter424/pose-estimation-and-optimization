import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import matplotlib.pyplot as plt
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import time
import os
import pandas as pd  # 用于读取CSV文件
import pyvista as pv  # 用于3D可视化
from pyvistaqt import BackgroundPlotter  # 用于交互式3D绘图
import math
from matplotlib.animation import FuncAnimation
os.environ['PATH'] += ';D:\\software\\ffmpeg-7.1-essentials_build'
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = 'D:\\software\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'
from collections import defaultdict  

# 1. 加载相机参数
def load_camera_params(file_path):
    data = np.load(file_path)
    return data['mtx_left'], data['dist_left'], data['mtx_right'], data['dist_right']

# 2. 定义函数以获取 ArUco 标记坐标和姿态估计
def get_aruco_axis(img_L, img_R, aruco_detector, board_coord, cams_params):
    (mtx_left, dist_left, mtx_right, dist_right) = cams_params
    axis_coord = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ],dtype=np.float32)
    axis_coord = axis_coord * 500

    # 在左右图像中检测 ArUco 标记
    corners_L, ids_L, _ = aruco_detector.detectMarkers(img_L)
    corners_R, ids_R, _ = aruco_detector.detectMarkers(img_R)

    if ids_L is None or ids_R is None:
        print("至少有一个相机未检测到ArUco标记")
        return (None,) * 10

    # 获取共同检测到的ArUco标记ID
    ids_L_set = set([id[0] for id in ids_L if id[0] in board_coord])
    ids_R_set = set([id[0] for id in ids_R if id[0] in board_coord])
    common_ids = ids_L_set & ids_R_set

    if not common_ids:
        print("左右相机没有共同检测到的有效ArUco标记")
        return (None,) * 10

    # 为左相机收集点
    img_coords_L = []
    point_coords_L = []
    aruco_ps_L_2camL = []
    id_order_L = []

    for i, id_val in enumerate(ids_L):
        if id_val[0] in common_ids:
            tmp_marker = corners_L[i][0]
            # 绘制标记点和ID
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            cv2.circle(img_L, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_L, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_L, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_L, tmp_marker_bl, 10, (0, 170, 255), -1)
            cv2.putText(img_L, f"ID: {id_val[0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            img_coord = np.array([tmp_marker[0], tmp_marker[1], tmp_marker[2], tmp_marker[3]])
            img_coords_L.append(img_coord)
            tem_coord = np.hstack((board_coord[id_val[0]], np.zeros(len(board_coord[id_val[0]]))[:,None]))
            point_coords_L.append(tem_coord)
            id_order_L.append(id_val[0])

            # 计算在左相机坐标系中的位置
            image_C_L = np.ascontiguousarray(img_coord).reshape((-1,1,2))
            ret_L, rvec_L, tvec_L = cv2.solvePnP(tem_coord, image_C_L, mtx_left, dist_left)
            R_aruco2camL, _ = cv2.Rodrigues(rvec_L)
            aruco_p_L_2camL = np.dot(R_aruco2camL, tem_coord.T).T + tvec_L.T
            aruco_ps_L_2camL.append(aruco_p_L_2camL)

    # 为右相机收集点
    img_coords_R = []
    point_coords_R = []
    aruco_ps_R_2camR = []
    id_order_R = []

    for i, id_val in enumerate(ids_R):
        if id_val[0] in common_ids:
            tmp_marker = corners_R[i][0]
            # 绘制标记点和ID
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            cv2.circle(img_R, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_R, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_R, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_R, tmp_marker_bl, 10, (0, 170, 255), -1)
            cv2.putText(img_R, f"ID: {id_val[0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            img_coord = np.array([tmp_marker[0], tmp_marker[1], tmp_marker[2], tmp_marker[3]])
            img_coords_R.append(img_coord)
            tem_coord = np.hstack((board_coord[id_val[0]], np.zeros(len(board_coord[id_val[0]]))[:,None]))
            point_coords_R.append(tem_coord)
            id_order_R.append(id_val[0])

            # 计算在右相机坐标系中的位置
            image_C_R = np.ascontiguousarray(img_coord).reshape((-1,1,2))
            ret_R, rvec_R, tvec_R = cv2.solvePnP(tem_coord, image_C_R, mtx_right, dist_right)
            R_aruco2camR, _ = cv2.Rodrigues(rvec_R)
            aruco_p_R_2camR = np.dot(R_aruco2camR, tem_coord.T).T + tvec_R.T
            aruco_ps_R_2camR.append(aruco_p_R_2camR)

    # 确保左右相机点的顺序一致
    sorted_indices_L = np.argsort(id_order_L)
    sorted_indices_R = np.argsort(id_order_R)

    img_coords_L = np.array([img_coords_L[i] for i in sorted_indices_L])
    point_coords_L = np.array([point_coords_L[i] for i in sorted_indices_L])
    aruco_ps_L_2camL = np.array([aruco_ps_L_2camL[i] for i in sorted_indices_L])

    img_coords_R = np.array([img_coords_R[i] for i in sorted_indices_R])
    point_coords_R = np.array([point_coords_R[i] for i in sorted_indices_R])
    aruco_ps_R_2camR = np.array([aruco_ps_R_2camR[i] for i in sorted_indices_R])

    # 打印调试信息
    print(f"共同检测到的ArUco标记ID: {sorted(common_ids)}")
    print(f"左相机标记顺序: {[id_order_L[i] for i in sorted_indices_L]}")
    print(f"右相机标记顺序: {[id_order_R[i] for i in sorted_indices_R]}")

    
        # 对左相机数据进行聚类处理
    if len(img_coords_L) > 0 and len(point_coords_L) > 0:
        # 初始化聚类字典
        clusters_L = defaultdict(list)
        cluster_ids_L = []
        cluster_indx_L = {}
        
        for i, point in enumerate(aruco_ps_L_2camL):
            new_cluster = True
            for cluster_id, cluster_points in clusters_L.items():
                for cluster_point in cluster_points:
                    if np.linalg.norm(point-cluster_point) <= board_coord[5][3,1]:
                        clusters_L[cluster_id].append(point)
                        cluster_indx_L[cluster_id].append(i)
                        new_cluster = False
                        break
                if not new_cluster:
                    break
            
            if new_cluster:
                cluster_id = len(cluster_ids_L)
                cluster_ids_L.append(cluster_id)
                clusters_L[cluster_id] = [point]
                cluster_indx_L[cluster_id] = [i]

        # 找出最大聚类
        cluster_max_indxs_L = []
        cluster_max_id_L = None
        for cluster_id, indxs in cluster_indx_L.items():
            if len(indxs) > len(cluster_max_indxs_L):
                cluster_max_indxs_L = indxs
                cluster_max_id_L = cluster_id
        
        cluster_max_indxs_L.sort()
        image_C_L = np.ascontiguousarray(img_coords_L.reshape(-1, 2)).reshape((-1,1,2))
        ret_L, rvec_L, tvec_L = cv2.solvePnP(point_coords_L.reshape(-1, 3), image_C_L, mtx_left, dist_left)
        
        # 绘制坐标轴
        image_points, _ = cv2.projectPoints(axis_coord, rvec_L, tvec_L, mtx_left, dist_left)
        image_points = image_points.reshape(-1, 2).astype(np.int16)
        cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5)
        cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5)
        cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5)

    if len(img_coords_R) > 0 and len(point_coords_R) > 0:
        # 初始化一个字典来存储聚类  
        clusters = defaultdict(list)  
        cluster_ids = []  
        cluster_indx = {}
        for i, point in enumerate(aruco_ps_R_2camR):  
            new_cluster = True  
            # 检查当前点是否已经属于某个聚类  
            for cluster_id, cluster_points in clusters.items():  
                # 检查当前点与聚类中每个点的距离  
                for cluster_point in cluster_points:  
                    # import pdb
                    # pdb.set_trace()
                    if np.linalg.norm(point-cluster_point) <= board_coord[5][3,1]:  
                        # 如果距离小于等于阈值，则将该点添加到现有聚类中  
                        clusters[cluster_id].append(point)  
                        cluster_indx[cluster_id].append(i)
                        new_cluster = False  
                        break  
                if not new_cluster:  
                    break  # 不需要再检查其他聚类了  
        
            # 如果当前点是一个新的聚类，则为其分配一个新的聚类ID  
            if new_cluster:  
                cluster_id = len(cluster_ids)  
                cluster_ids.append(cluster_id)  
                clusters[cluster_id] = [point]  
                cluster_indx[cluster_id] = [i]
        # import pdb
        # pdb.set_trace()
        # 统计每个聚类中的点数  
        cluster_max_indxs = []
        cluster_max_id = None
        for cluster_id, indxs in cluster_indx.items():
            if len(indxs) > len(cluster_max_indxs):
                cluster_max_indxs = indxs
                cluster_max_id = cluster_id
        cluster_max_indxs.sort()
        img_coords_R = img_coords_R[cluster_max_indxs]
        point_coords_R = point_coords_R[cluster_max_indxs]
        # ... 现有的聚类代码 ...
        image_C_R = np.ascontiguousarray(img_coords_R.reshape(-1, 2)).reshape((-1,1,2))
        ret_R, rvec_R, tvec_R = cv2.solvePnP(point_coords_R.reshape(-1, 3), image_C_R, mtx_right, dist_right)
        
        # 绘制坐标轴
        image_points, _ = cv2.projectPoints(axis_coord, rvec_R, tvec_R, mtx_right, dist_right)
        image_points = image_points.reshape(-1, 2).astype(np.int16)
        cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5)
        cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5)
        cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5)

    # 计算最终的转换矩阵
    R_aruco2camL, _ = cv2.Rodrigues(rvec_L)
    t_aruco2camL = tvec_L
    R_aruco2camR, _ = cv2.Rodrigues(rvec_R)
    t_aruco2camR = tvec_R

    # 修改这部分：计算相机到 ArUco 的变换
    # 原来的代码
    # R_camL2aruco = R_aruco2camL.T
    # t_camL2aruco = -R_aruco2camL.T @ t_aruco2camL
    # R_camR2aruco = R_aruco2camR.T
    # t_camR2aruco = -R_aruco2camR.T @ t_aruco2camR

    # 新的代码：正确处理相机到ArUco的变换
    R_camL2aruco = R_aruco2camL.T
    t_camL2aruco = -R_aruco2camL.T @ t_aruco2camL.reshape(3,1)
    R_camR2aruco = R_aruco2camR.T
    t_camR2aruco = -R_aruco2camR.T @ t_aruco2camR.reshape(3,1)

    return R_aruco2camL, t_aruco2camL, R_aruco2camR, t_aruco2camR, R_camL2aruco, t_camL2aruco.flatten(), R_camR2aruco, t_camR2aruco.flatten(), img_L, img_R


# 3. 定义处理单帧的函数
def process_frame(detector, frame_left, frame_right, img_L, img_R, cams_params, cam_P):
    # try:
    # 使用MediaPipe检测人体姿势
    mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
    mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))

    detection_result_left = detector.detect(mp_image_left)
    detection_result_right = detector.detect(mp_image_right)

    # 检查是否检测到关键点
    if detection_result_left.pose_landmarks and len(detection_result_left.pose_landmarks) > 0:
        pose_left = np.array([[landmark.x * frame_left.shape[1], landmark.y * frame_left.shape[0]] 
                                for landmark in detection_result_left.pose_landmarks[0]])
    else:
        print("左侧图像未检测到人体姿势")
        return None

    if detection_result_right.pose_landmarks and len(detection_result_right.pose_landmarks) > 0:
        pose_right = np.array([[landmark.x * frame_right.shape[1], landmark.y * frame_right.shape[0]] 
                                for landmark in detection_result_right.pose_landmarks[0]])
    else:
        print("右侧图像未检测到人体姿势")
        return None

    # 三维重建
    (R_aruco2camL, t_aruco2camL, R_aruco2camR, t_aruco2camR) = cam_P
    P1 = np.hstack((R_aruco2camL, t_aruco2camL))
    P2 = np.hstack((R_aruco2camR, t_aruco2camR))
    (mtx_left, dist_left, mtx_right, dist_right) = cams_params
    points_3d = triangulate_points(pose_left, pose_right, mtx_left, dist_left, mtx_right, dist_right, P1, P2)

    #将二维点映射至左右二图
    for mark_i in range(33):
        mark_coord = (int(pose_left[mark_i,0]), int(pose_left[mark_i,1]))
        cv2.circle(img_L, mark_coord, 10, (0, 0, 255), -1)
        mark_coord = (int(pose_right[mark_i,0]), int(pose_right[mark_i,1]))
        cv2.circle(img_R, mark_coord, 10, (0, 0, 255), -1)

    return points_3d, img_L, img_R
    # except Exception as e:
    #     print(f"处理单帧时出现错误: {e}")
    #     return None

# 4. 定三角量函数
def triangulate_points(points_left, points_right, mtx_left, dist_left, mtx_right, dist_right, P1, P2):
    # 构建投影矩阵
    # P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    # P2 = np.hstack((R, T.reshape(3, 1)))
    
    points_3d = []
    for pt_left, pt_right in zip(points_left, points_right):
        pt_left_undist = cv2.undistortPoints(pt_left.reshape(1, 1, 2), mtx_left, dist_left)
        pt_right_undist = cv2.undistortPoints(pt_right.reshape(1, 1, 2), mtx_right, dist_right)
        
        point_4d = cv2.triangulatePoints(P1, P2, pt_left_undist, pt_right_undist)
        point_3d = (point_4d / point_4d[3])[:3]
        
        points_3d.append(point_3d.ravel())
    
    return np.array(points_3d)

# 4. 将实际骨头点安装到骨架上
def transform_bone_points(points_3d, bone_points):
    """
    將骨骼點轉換到人體姿態坐標系中
    
    參數:
        points_3d: 3D人體關鍵點坐標
        bone_points: 包含左右腳骨骼點的元組 (bone_points_L, bone_points_R)
        
    返回:
        轉換後的左右腳骨骼點組合
    """
    # 解包左右腳骨骼點
    bone_points_L, bone_points_R = bone_points
    
    # 左腳骨骼點精細化旋轉
    # eulerAnglesToRotationMatrix函數用於將歐拉角轉換為旋轉矩陣
    # 參數為[x,y,z]三個旋轉角度(弧度制):
    # x: 繞x軸旋轉角度 = 0度
    # y: 繞y軸旋轉角度 = 70度 
    # z: 繞z軸旋轉角度 = 180度
    R_refine1 = eulerAnglesToRotationMatrix([0,70*np.pi/180,180*np.pi/180])  # 第一次旋轉
    R_refine2 = eulerAnglesToRotationMatrix([0,140*np.pi/180,0])  # 第二次旋轉
    R_refine = np.dot(R_refine2, R_refine1)  # 組合旋轉矩陣
    bone_points_refined_L = np.dot(R_refine,bone_points_L.T)  # 應用旋轉到左腳骨骼點
    
    # 右腳骨骼點精細化旋轉
    R_refine1 = eulerAnglesToRotationMatrix([0,-10*np.pi/180,180*np.pi/180])  # 第一次旋轉
    R_refine2 = eulerAnglesToRotationMatrix([0,70*np.pi/180,0])  # 第二次旋轉
    R_refine = np.dot(R_refine2, R_refine1)  # 組合旋轉矩陣
    bone_points_refined_R = np.dot(R_refine,bone_points_R.T)  # 應用旋轉到右腳骨骼點
    
    # 計算左右腳的坐標系統
    rotation_L, translation_L = define_coordinate_system([points_3d[27],points_3d[31],points_3d[29]])  # 左腳坐標系
    rotation_R, translation_R = define_coordinate_system([points_3d[28],points_3d[32],points_3d[30]])  # 右腳坐標系
    
    # 將骨骼點轉換到對應的坐標系統中
    bone_points_transed_L = (bone_points_refined_L.T @ rotation_L.T + translation_L)  # 左腳轉換
    bone_points_transed_R = (bone_points_refined_R.T @ rotation_R.T + translation_R)  # 右腳轉換

    # 合併並返回轉換後的左右腳骨骼點
    return np.concatenate([bone_points_transed_L,bone_points_transed_R])
    
# 5. 定义处理视频的主循环
def process_videos(video_path_left, video_path_right, start_frame=0):
    ## 0. 导入相机内参
    # camera_params_path = 'stereo_camera_calibration.npz'
    camera_params_path = r"C:\Users\user\Desktop\113Camera\ARUCOstereo_camera_calibration.npz"
    if not os.path.exists(camera_params_path):
        print('camera_params_path is error.')
        return
    mtx_left, dist_left, mtx_right, dist_right = load_camera_params(camera_params_path)
    cams_params = (mtx_left, dist_left, mtx_right, dist_right)

    ## 1. 设置aruco参数
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    # 定义 ArUco 标记板的坐标
    board_length = 160
    board_gap = 30
    base_coord = np.array([[0,0],[0,1],[1,1],[1,0]])
    board_coord = {
        0: base_coord * board_length + [0,0],
        1: base_coord * board_length + [board_length+board_gap,0],
        2: base_coord * board_length + [0,board_length+board_gap],
        3: base_coord * board_length + [board_length+board_gap,board_length+board_gap],
        4: base_coord * board_length + [0,(board_length+board_gap)*2],
        5: base_coord * board_length + [board_length+board_gap,(board_length+board_gap)*2],
    }

    ## 2. 设置MediaPipe姿态估计模型
    # model_asset_path=r"C:\Users\user\Desktop\pose_landmarker_full.task"
    model_asset_path = r"C:\Users\user\Desktop\pose_landmarker_full.task"
    if not os.path.exists(model_asset_path):
        print('model_asset_path is error.')
        return
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    mp_pose = mp.solutions.pose

    ## 3. 加载骨骼点
    # bone_points = load_bone_points('./Bone_points_scaled.csv') 
    bone_points_L = load_bone_points(r"C:\Users\user\Desktop\L_footpoints.csv")
    bone_points_R = load_bone_points(r"C:\Users\user\Desktop\R_footpoints.csv")#"C:\Users\user\Desktop\R_footpoints.csv"
   # bone_points_L = load_bone_points(r"D:\projects\01jobs\002skeleton\20241025\L_footpoints.csv")
    #bone_points_R = load_bone_points(r"D:\projects\01jobs\002skeleton\20241025\R_footpoints.csv")
    scale_factor = 0.001 # 0.0959
    bone_points = (bone_points_L*scale_factor, bone_points_R*scale_factor)
    
    ## 4. 读取视频
    print(f"开始处理视频从帧 {start_frame} 开始: {video_path_left} 和 {video_path_right}")
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise ValueError("无法打开视频文件")
    # 跳过已经处理的帧
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # 收集每帧的点
    all_points_3d = [] # 3d骨骼点
    all_bone_points = [] # 预设骨头点
    aruco_axis = [] # aruco坐标系
    camL_axis = [] # 左相机坐标系
    camR_axis = [] # 右相机坐标系
    frame_count = start_frame
    # 坐标系的4个点
    # 定義坐標系的四個點:
    # [0,0,0] - 原點
    # [1,0,0] - X軸方向的單位向量端點
    # [0,1,0] - Y軸方向的單位向量端點  
    # [0,0,1] - Z軸方向的單位向量端點
    axis_coord = np.array([
        [0,0,0],  # 原點
        [1,0,0],  # X軸
        [0,1,0],  # Y軸 
        [0,0,1]   # Z軸
    ],dtype=np.float32)
    axis_coord = axis_coord * 200
    # 开始循环读取每一帧视频图像
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            break
        print(f"处理第 {frame_count + 1} 帧")
        if frame_count == 912:#可修改
            break
        # try:
        result = get_aruco_axis(frame_left, frame_right, aruco_detector, board_coord, cams_params)
        if result[0] is None:
            print(f"第 {frame_count + 1} 帧未检测到 ArUco 标记")
            all_points_3d.append(np.zeros((33, 3)))  # 假设有33个关键点
            all_bone_points.append(np.zeros((1,3)))
            aruco_axis.append(np.zeros((4,3)))
            camL_axis.append(np.zeros((4,3)))
            camR_axis.append(np.zeros((4,3)))
            frame_count += 1
            continue
        R_aruco2camL, t_aruco2camL, R_aruco2camR, t_aruco2camR, R_camL2aruco, t_camL2aruco, R_camR2aruco, t_camR2aruco, img_L, img_R = result
        aruco_axis.append(axis_coord)
        camL_axis.append(np.dot(R_camL2aruco, (axis_coord).T).T + t_camL2aruco.T)
        camR_axis.append(np.dot(R_camR2aruco, (axis_coord).T).T + t_camR2aruco.T)
        # except Exception as e:
        #     print(f"处理第 {frame_count + 1} 帧时出错: {str(e)}")
        #     all_points_3d.append(np.zeros((33, 3)))  # 假设有33个关键点
        #     all_bone_points.append(np.zeros((1,3)))
        #     aruco_axis.append(np.zeros((4,3)))
        #     camL_axis.append(np.zeros((4,3)))
        #     camR_axis.append(np.zeros((4,3)))
        #     frame_count += 1
        #     continue
        # import pdb
        # pdb.set_trace()
        cam_P = (R_aruco2camL, t_aruco2camL, R_aruco2camR, t_aruco2camR)
        points_3d, img_L, img_R = process_frame(detector, frame_left, frame_right, img_L, img_R, cams_params, cam_P)
        per_bone_points = transform_bone_points(points_3d, bone_points)
        all_bone_points.append(per_bone_points)
        all_points_3d.append(points_3d)

        # 显示左右图像（可选）
        cv2.imshow('Left and Right Images', np.hstack((cv2.resize(img_L, (640, 480)), cv2.resize(img_R, (640, 480)))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    print(f"视频处理完成。共处理了 {frame_count - start_frame} 帧")
    return np.array(all_points_3d), np.array(all_bone_points), np.array(aruco_axis), np.array(camL_axis), np.array(camR_axis)

# 6. 定义平滑函数
def smooth_points_savgol(points, window_size=7, polyorder=3):
    smoothed_points = np.zeros_like(points)
    for i in range(points.shape[1]):
        for j in range(3):  # x, y, z
            smoothed_points[:, i, j] = savgol_filter(points[:, i, j], window_size, polyorder, mode='nearest')
    return smoothed_points

# 7. 使用PyVista进行3D可视化和点选
def select_points_with_pyvista(points_3d):
    # 使用第一帧的点进行选择
    frame_points = points_3d[0]

    # 创建一个PyVista点云
    point_cloud = pv.PolyData(frame_points)
    point_cloud['labels'] = np.arange(frame_points.shape[0])

    plotter = pv.Plotter()
    plotter.add_points(point_cloud, render_points_as_spheres=True, point_size=10, pickable=True)
    plotter.add_axes()

    selected_points = []

    def callback(picked_point):
        if picked_point is None:
            return
        if len(selected_points) < 3:
            selected_points.append(picked_point)
            # 在选中的点上画一个球体
            sphere = pv.Sphere(radius=0.02, center=picked_point)
            plotter.add_mesh(sphere, color='red')
            plotter.render()
            print(f"已选择点：{picked_point}")
        if len(selected_points) == 3:
            # 不立即关闭绘图，等待用户手动关闭
            print("已选择三个点，请关闭窗口继续。")

    plotter.enable_point_picking(callback=callback, use_mesh=False, show_message=True)

    plotter.show()

    return np.array(selected_points)

# 8. 定义坐标系转换函数
def define_coordinate_system(selected_points):
    """根据选定的三个点定义坐标系"""
    if len(selected_points) != 3:
        raise ValueError("Please select exactly 3 points.")

    # 获取三个点
    origin = np.array(selected_points[0])
    point_x = np.array(selected_points[1])
    point_y_temp = np.array(selected_points[2])

    # 计算X轴
    # x軸指向point_x
    x_axis = point_x - origin
    if np.linalg.norm(x_axis) == 0:
        raise ValueError("Point 1 and Point 2 are the same.")
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 计算临时向量（Point 1 到 Point 3）
    temp_vec = point_y_temp - origin
    if np.linalg.norm(temp_vec) == 0:
        raise ValueError("Point 1 and Point 3 are the same.")

    # 计算Y轴（temp_vec 与 X轴的叉积）
    y_axis = np.cross(temp_vec, x_axis)
    if np.linalg.norm(y_axis) == 0:
        raise ValueError("Points are collinear.")
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 计算Z轴（X轴 与 Y轴的叉积）
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 构建旋转矩阵
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3矩阵
    translation = origin  # 偏移向量

    return rotation_matrix, translation

# 9. 转换3D关键点到新坐标系
def transform_points(points_3d, rotation_matrix, translation):
    # points_3d: (N, 3)
    transformed_points = (points_3d - translation) @ rotation_matrix
    return transformed_points

def transform_all_frames(all_points_3d, rotation_matrix, translation):
    transformed_all_points = []
    for frame_points in all_points_3d:
        transformed_points = transform_points(frame_points, rotation_matrix, translation)
        transformed_all_points.append(transformed_points)
    return np.array(transformed_all_points)

# 10. 加载骨骼点
def load_bone_points(csv_file_path):
    start_time = time.time()  # 开始计时

    df = pd.read_csv(csv_file_path)
    bone_points = df[['X', 'Y', 'Z']].values  # 假设CSV文件有这些列

    # 对骨骼点进行下采样，例如每隔10个点取一个
    bone_points = bone_points[::10]

    end_time = time.time()  # 结束计时
    load_time = end_time - start_time
    print(f"加载骨骼点数据耗时: {load_time:.4f} 秒")
    print(f"加载了 {len(bone_points)} 个骨骼点")

    return bone_points

# 将欧拉角转换为旋转矩阵x->y->z
def eulerAnglesToRotationMatrix(ang):
    R_x = np.array([[1, 0, 0 ], [0, math.cos(ang[0]), -math.sin(ang[0]) ], [0, math.sin(ang[0]), math.cos(ang[0]) ]]) 
    R_y = np.array([[math.cos(ang[1]), 0, math.sin(ang[1]) ], [0, 1, 0 ], [-math.sin(ang[1]), 0, math.cos(ang[1]) ]]) 
    R_z = np.array([[math.cos(ang[2]), -math.sin(ang[2]), 0], [math.sin(ang[2]), math.cos(ang[2]), 0], [0, 0, 1] ]) 
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R
def visualize_with_pltAnimation_2(transformed_points_3d, aruco_axises, camL_axises, camR_axises):
    # 创建图形窗口
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置标题和标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 计算显示范围
    all_points = np.vstack((transformed_points_3d.reshape(-1, 3), 
                        aruco_axises.reshape(-1, 3), 
                        camL_axises.reshape(-1, 3), 
                        camR_axises.reshape(-1, 3)))
    
    # 设置坐标轴范围
    margin = 100  
    ax.set_xlim([np.min(all_points[:,0])-margin, np.max(all_points[:,0])+margin])
    ax.set_ylim([np.min(all_points[:,1])-margin, np.max(all_points[:,1])+margin])
    ax.set_zlim([np.min(all_points[:,2])-margin, np.max(all_points[:,2])+margin])
    
    # 设置初始视角
    ax.view_init(elev=10, azim=45)
    
    # 初始化骨骼点散点图对象
    points_scatter = ax.scatter([], [], [], c='b', marker='.', s=20)
    
    # 初始化骨架线条
    skeleton_lines = []
    connections = [
        # 头部
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (3, 6),  # 连接头部两侧
        # 颈部
        (9, 10), (10, 11),
        # 躯干
        (11, 12), (11, 23), (12, 24), (23, 24),
        # 左臂
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # 手掌和手指
        (17, 19), (19, 21),  # 手指连接
        # 右臂
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # 手掌和手指
        (18, 20), (20, 22),  # 手指连接
        # 左腿
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        # 右腿
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
    ]
    
    # 为不同部位设置不同颜色
    colors = {
        'head': 'purple',     # 头部
        'spine': 'blue',      # 脊柱
        'arms': 'green',      # 手臂
        'legs': 'red',        # 腿部
        'hands': 'orange'     # 手部
    }
    
    # 定义每个连接的颜色
    connection_colors = []
    for start, end in connections:
        if start <= 8 or end <= 8:  # 头部连接
            connection_colors.append(colors['head'])
        elif start in [9, 10, 11] or end in [9, 10, 11]:  # 脊柱
            connection_colors.append(colors['spine'])
        elif (start in [13, 14, 15, 16] or end in [13, 14, 15, 16]):  # 手臂
            connection_colors.append(colors['arms'])
        elif start >= 17 or end >= 17:  # 手部
            connection_colors.append(colors['hands'])
        else:  # 腿部
            connection_colors.append(colors['legs'])
    
    # 创建线条
    for color in connection_colors:
        line, = ax.plot([], [], [], color=color, alpha=0.8, linewidth=2)
        skeleton_lines.append(line)
    
    # 添加坐标系线条
    skeleton_lines.extend([
        ax.plot([], [], [], 'r-', lw=2, alpha=0.7)[0],  # aruco x
        ax.plot([], [], [], 'g-', lw=2, alpha=0.7)[0],  # aruco y
        ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0],  # aruco z
        ax.plot([], [], [], 'r-', lw=2, alpha=0.7)[0],  # camL x
        ax.plot([], [], [], 'g-', lw=2, alpha=0.7)[0],  # camL y
        ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0],  # camL z
        ax.plot([], [], [], 'r-', lw=2, alpha=0.7)[0],  # camR x
        ax.plot([], [], [], 'g-', lw=2, alpha=0.7)[0],  # camR y
        ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0]   # camR z
    ])
    
    def update(frame):
        # 更新骨骼点
        print(frame)
        point_cloud = transformed_points_3d[frame]
        aruco_axis = aruco_axises[frame]
        camL_axis = camL_axises[frame]
        camR_axis = camR_axises[frame]

        points_scatter._offsets3d = (point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
        
        # 更新骨架线条
        pose_points = transformed_points_3d[frame]
        for indx in range(len(connections)):
            line = skeleton_lines[indx]
            (start, end) = connections[indx]
            line.set_data_3d([pose_points[start,0], pose_points[end,0]],
                           [pose_points[start,1], pose_points[end,1]],
                           [pose_points[start,2], pose_points[end,2]])
        
        # 更新坐标系线条
        # aruco坐标系
        skeleton_lines[-9].set_data_3d([aruco_axis[0, 0], aruco_axis[1, 0]],  # X轴
                                    [aruco_axis[0, 1], aruco_axis[1, 1]],
                                    [aruco_axis[0, 2], aruco_axis[1, 2]])
        skeleton_lines[-8].set_data_3d([aruco_axis[0, 0], aruco_axis[2, 0]],  # Y轴
                                    [aruco_axis[0, 1], aruco_axis[2, 1]],
                                    [aruco_axis[0, 2], aruco_axis[2, 2]])
        skeleton_lines[-7].set_data_3d([aruco_axis[0, 0], aruco_axis[3, 0]],  # Z轴
                                    [aruco_axis[0, 1], aruco_axis[3, 1]],
                                    [aruco_axis[0, 2], aruco_axis[3, 2]])
        # 左相机坐标系
        skeleton_lines[-6].set_data_3d([camL_axis[0, 0], camL_axis[1, 0]],  # X轴
                                    [camL_axis[0, 1], camL_axis[1, 1]],
                                    [camL_axis[0, 2], camL_axis[1, 2]])
        skeleton_lines[-5].set_data_3d([camL_axis[0, 0], camL_axis[2, 0]],  # Y轴
                                    [camL_axis[0, 1], camL_axis[2, 1]],
                                    [camL_axis[0, 2], camL_axis[2, 2]])
        skeleton_lines[-4].set_data_3d([camL_axis[0, 0], camL_axis[3, 0]],  # Z轴
                                    [camL_axis[0, 1], camL_axis[3, 1]],
                                    [camL_axis[0, 2], camL_axis[3, 2]])
        # 右相机坐标系
        skeleton_lines[-3].set_data_3d([camR_axis[0, 0], camR_axis[1, 0]],  # X轴
                                    [camR_axis[0, 1], camR_axis[1, 1]],
                                    [camR_axis[0, 2], camR_axis[1, 2]])
        skeleton_lines[-2].set_data_3d([camR_axis[0, 0], camR_axis[2, 0]],  # Y轴
                                    [camR_axis[0, 1], camR_axis[2, 1]],
                                    [camR_axis[0, 2], camR_axis[2, 2]])
        skeleton_lines[-1].set_data_3d([camR_axis[0, 0], camR_axis[3, 0]],  # Z轴
                                    [camR_axis[0, 1], camR_axis[3, 1]],
                                    [camR_axis[0, 2], camR_axis[3, 2]])

        # 更新标题显示当前帧
        ax.set_title(f'Frame: {frame}')
        
        return [points_scatter] + skeleton_lines
    
    # 创建动画
    anim = FuncAnimation(
        fig, 
        update,
        frames=len(transformed_points_3d),
        interval=50,
        blit=False,
        repeat=True
    )
    plt.show()
    
    # 返回动画对象以防止被垃圾回收
    return anim

# 11-2. 使用PyVista可视化转换后的关键点和骨骼点
def visualize_with_pltAnimation(transformed_points_3d, bone_points_all, aruco_axises, camL_axises, camR_axises):
    # bone_points_all = np.concatenate(bone_points_all)
    # 创建图形窗口
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置标题和标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 计算显示范围
    all_points = np.vstack((transformed_points_3d.reshape(-1, 3), 
                        bone_points_all.reshape(-1, 3),
                        aruco_axises.reshape(-1, 3), 
                        camL_axises.reshape(-1, 3), 
                        camR_axises.reshape(-1, 3)))
    
    # 设置坐标轴范围
    margin = 100  
    ax.set_xlim([np.min(all_points[:,0])-margin, np.max(all_points[:,0])+margin])
    ax.set_ylim([np.min(all_points[:,1])-margin, np.max(all_points[:,1])+margin])
    ax.set_zlim([np.min(all_points[:,2])-margin, np.max(all_points[:,2])+margin])
    
    # 设置初始视角
    ax.view_init(elev=10, azim=45)
    
    # 初始化骨骼点散点图对象（移除了姿态关键点的散点图）
    points_scatter = ax.scatter([], [], [], c='b', marker='.', s=20)
    bone_scatter = ax.scatter([], [], [], c='#C0C0C0', marker='.', s=20)
    # 初始化骨架线条
    skeleton_lines = []
    connections = [
        # 头部
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (3, 6),  # 连接头部两侧
        # 颈部
        (9, 10), (10, 11),
        # 躯干
        (11, 12), (11, 23), (12, 24), (23, 24),
        # 左臂
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # 手掌和手指
        (17, 19), (19, 21),  # 手指连接
        # 右臂
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # 手掌和手指
        (18, 20), (20, 22),  # 手指连接
        # 左腿
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        # 右腿
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
    ]
    
    # 为不同部位设置不同颜色
    colors = {
        'head': 'purple',     # 头部
        'spine': 'blue',      # 脊柱
        'arms': 'green',      # 手臂
        'legs': 'red',        # 腿部
        'hands': 'orange'     # 手部
    }
    
    # 定义每个连接的颜色
    connection_colors = []
    for start, end in connections:
        if start <= 8 or end <= 8:  # 头部连接
            connection_colors.append(colors['head'])
        elif start in [9, 10, 11] or end in [9, 10, 11]:  # 脊柱
            connection_colors.append(colors['spine'])
        elif (start in [13, 14, 15, 16] or end in [13, 14, 15, 16]):  # 手臂
            connection_colors.append(colors['arms'])
        elif start >= 17 or end >= 17:  # 手部
            connection_colors.append(colors['hands'])
        else:  # 腿部
            connection_colors.append(colors['legs'])
    
    # 创建线条
    for color in connection_colors:
        line, = ax.plot([], [], [], color=color, alpha=0.8, linewidth=2)
        skeleton_lines.append(line)
    line, = ax.plot([], [], [], color=color, alpha=0.8, linewidth=2)
    skeleton_lines.extend([ax.plot([], [], [], 'r-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'g-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'r-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'g-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'r-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'b-', lw=2, alpha=0.7)[0],
                        ax.plot([], [], [], 'g-', lw=2, alpha=0.7)[0]])
    
    def update(frame):
        # 更新骨骼点
        print(frame)
        point_cloud = transformed_points_3d[frame]
        bone_cloud = bone_points_all[frame]
        aruco_axis = aruco_axises[frame]
        camL_axis = camL_axises[frame]
        camR_axis = camR_axises[frame]

        points_scatter._offsets3d = (point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
        bone_scatter._offsets3d = (bone_cloud[:,0], bone_cloud[:,1], bone_cloud[:,2])
        
        # 更新骨架线条
        pose_points = transformed_points_3d[frame]
        for indx in range(len(connections)):
            line = skeleton_lines[indx]
            (start, end) = connections[indx]
            line.set_data_3d([pose_points[start,0], pose_points[end,0]],
                           [pose_points[start,1], pose_points[end,1]],
                           [pose_points[start,2], pose_points[end,2]])
        
        # ArUco坐标系
        skeleton_lines[-9].set_data_3d([aruco_axis[0, 0], aruco_axis[1, 0]],  # X轴
                                    [aruco_axis[0, 1], aruco_axis[1, 1]],
                                    [aruco_axis[0, 2], aruco_axis[1, 2]])
        skeleton_lines[-8].set_data_3d([aruco_axis[0, 0], aruco_axis[2, 0]],  # Y轴
                                    [aruco_axis[0, 1], aruco_axis[2, 1]],
                                    [aruco_axis[0, 2], aruco_axis[2, 2]])
        skeleton_lines[-7].set_data_3d([aruco_axis[0, 0], aruco_axis[3, 0]],  # Z轴
                                    [aruco_axis[0, 1], aruco_axis[3, 1]],
                                    [aruco_axis[0, 2], aruco_axis[3, 2]])

        # 左相机坐标系
        skeleton_lines[-6].set_data_3d([camL_axis[0, 0], camL_axis[1, 0]],  # X轴
                                    [camL_axis[0, 1], camL_axis[1, 1]],
                                    [camL_axis[0, 2], camL_axis[1, 2]])
        skeleton_lines[-5].set_data_3d([camL_axis[0, 0], camL_axis[2, 0]],  # Y轴
                                    [camL_axis[0, 1], camL_axis[2, 1]],
                                    [camL_axis[0, 2], camL_axis[2, 2]])
        skeleton_lines[-4].set_data_3d([camL_axis[0, 0], camL_axis[3, 0]],  # Z轴
                                    [camL_axis[0, 1], camL_axis[3, 1]],
                                    [camL_axis[0, 2], camL_axis[3, 2]])

        # 右相机坐标系
        skeleton_lines[-3].set_data_3d([camR_axis[0, 0], camR_axis[1, 0]],  # X轴
                                    [camR_axis[0, 1], camR_axis[1, 1]],
                                    [camR_axis[0, 2], camR_axis[1, 2]])
        skeleton_lines[-2].set_data_3d([camR_axis[0, 0], camR_axis[2, 0]],  # Y轴
                                    [camR_axis[0, 1], camR_axis[2, 1]],
                                    [camR_axis[0, 2], camR_axis[2, 2]])
        skeleton_lines[-1].set_data_3d([camR_axis[0, 0], camR_axis[3, 0]],  # Z轴
                                    [camR_axis[0, 1], camR_axis[3, 1]],
                                    [camR_axis[0, 2], camR_axis[3, 2]])

        # 更新标题显示当前帧
        ax.set_title(f'Frame: {frame}')
        
        return [points_scatter, bone_scatter] + skeleton_lines
    
    # 创建动画
    anim = FuncAnimation(
        fig, 
        update,
        frames=len(transformed_points_3d),
        interval=50,
        blit=False,
        repeat=True
    )
    plt.show()
    return anim

# 14. 辅助函数：计算肢段长度
def calculate_limb_lengths(points_3d):
    limb_connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32)
    ]
    lengths = [np.linalg.norm(points_3d[start] - points_3d[end]) for start, end in limb_connections]
    return np.array(lengths)

# 15. GOM优化相关函数
def optimize_points_gom(points_sequence, reference_lengths):
    start_time = time.time()  # 开始计时整个序列的优化
    optimized_sequence = []

    for i, frame_points in enumerate(points_sequence):
        optimized_frame = optimize_frame_gom(frame_points, reference_lengths)
        optimized_sequence.append(optimized_frame)
        if (i + 1) % 10 == 0:  # 每10打印一次进度
            print(f"已完成 {i+1}/{len(points_sequence)} 帧的优化")

    end_time = time.time()  # 结束计时
    total_optimization_time = end_time - start_time
    print(f"整个序列优化时间: {total_optimization_time:.4f} 秒")
    return np.array(optimized_sequence)

def optimize_frame_gom(points, reference_lengths):
    start_time = time.time()  # 开始计时
    limb_connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32)
    ]
    points = points.astype(np.float64)  # 使用高精度

    def apply_hard_constraints(points_3d, reference_lengths, iterations=5):
        constrained = np.copy(points_3d)
        for _ in range(iterations):
            for (start, end), ref_length in zip(limb_connections, reference_lengths):
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
    final_points = constrained_points  # 由于优化耗时较长，这里暂时仅应用硬约束

    end_time = time.time()  # 束计时
    optimization_time = end_time - start_time
    return final_points

def calculate_limb_length_variations(all_points_3d, reference_lengths):
    limb_definitions = {
        "shoulders": (11, 12),
        "left upper arm": (11, 13),
        "left lower arm": (13, 15),
        "right upper arm": (12, 14),
        "right lower arm": (14, 16),
        "left hip": (11, 23),
        "right hip": (12, 24),
        "left thigh": (23, 25),
        "left calf": (25, 27),
        "left ankle": (27, 29),
        "left foot": (29, 31),
        "right thigh": (24, 26),
        "right calf": (26, 28),
        "right ankle": (28, 30),
        "right foot": (30, 32)
    }
    variations = []

    for i, (name, (start, end)) in enumerate(limb_definitions.items()):
        lengths = []
        for frame in all_points_3d:
            length = np.linalg.norm(frame[start] - frame[end])
            lengths.append(length)

        lengths = np.array(lengths)
        mean_length = np.mean(lengths)
        std_dev = np.std(lengths)
        variation_percentage = (std_dev / reference_lengths[i]) * 100

        variations.append({
            "limb": name,
            "mean_length": mean_length,
            "std_dev": std_dev,
            "variation_percentage": variation_percentage
        })

    return variations

# 13. 主程
def main():
    # # 确保视频路径正确
    video_path_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camR\cam0-2.mp4"
    video_path_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration1016_orth\dual4\video\camL\cam1-2.mp4"
    #video_path_left = r"D:\projects\01jobs\002skeleton\orth_1020\camR\cam0-2.mp4"
    #video_path_right = r"D:\projects\01jobs\002skeleton\orth_1020\camL\cam1-2.mp4"

    # # 处理所有帧
    all_points_3d_original, all_bone_points, aruco_axises, camL_axises, camR_axises = process_videos(video_path_left, video_path_right)

    if len(all_points_3d_original) == 0:
        print("未能从视频中提取任何3D关键点。")
        return

    # 使用第一帧的关键点计算参考长度
    reference_lengths = calculate_limb_lengths(all_points_3d_original[0])

    # 应用Savitzky-Golay滤波器到原始数据
    smoothed_points_3d = smooth_points_savgol(all_points_3d_original, window_size=7, polyorder=3)

    # 应用GOM优化
    print("开始对所有帧进行GOM优化...")
    gom_optimized_points_3d = optimize_points_gom(smoothed_points_3d, reference_lengths)
    
    # # 保存为NPZ文件
    # print("保存数据为NPZ文件...")
    # np.savez('motion_data.npz',
    #          smoothed_points=smoothed_points_3d,          # 平滑后的点
    #          gom_optimized_points=gom_optimized_points_3d,  # GOM优化后的点
    #          all_bone_points=all_bone_points,                 # 实体骨头点
    #          aruco_axises=aruco_axises,                  # Aruco坐标系
    #          camL_axises=camL_axises,                    # 左相机坐标系
    #          camR_axises=camR_axises                     # 右相机坐标系
    # )
    # print("数据已保存到 motion_data.npz")
    
    # # 保存单独的NPZ文件（如果需要）
    # np.savez('smoothed_points.npz', points=smoothed_points_3d)
    # np.savez('gom_optimized_points.npz', points=gom_optimized_points_3d)
    # print("单独的点数据文件已保存")
    
    # 打印数据形状信息
    print("\n数据形状信息:")
    print(f"平滑点形状: {smoothed_points_3d.shape}")
    print(f"GOM优化点形状: {gom_optimized_points_3d.shape}")
    print(f"实体骨头形状: {all_bone_points.shape}")
    print(f"Aruco坐标系形状: {aruco_axises.shape}")
    print(f"左相机坐标系形状: {camL_axises.shape}")
    print(f"右相机坐标系形状: {camR_axises.shape}")
    
    # 骨骼模型转换至骨骼点坐标系，并显示
    print("开始可视化...")
    visualize_with_pltAnimation_2(gom_optimized_points_3d, aruco_axises, camL_axises, camR_axises)
    visualize_with_pltAnimation(gom_optimized_points_3d, all_bone_points, aruco_axises, camL_axises, camR_axises)


if __name__ == "__main__":
    main()
