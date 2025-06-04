import cv2

# 输入和输出视频文件路径
input_video = r"C:\Users\user\Desktop\20241016\cam0-2.mp4"
output_video = r"C:\Users\user\Desktop\20241016\cam0-2_30fps.mp4"

# 打开输入视频
cap = cv2.VideoCapture(input_video)

# 获取原始视频的帧率和总帧数
original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 设置目标帧率
target_fps = 20.0

# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

frame_number = 0  # 当前帧编号
next_output_time = 0.0  # 下一帧输出的时间

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_number / original_fps  # 计算当前帧的时间戳

    # 判断是否需要写入当前帧
    if current_time >= next_output_time:
        out.write(frame)
        next_output_time += 1.0 / target_fps  # 更新下一帧输出的时间

    frame_number += 1

# 释放资源
cap.release()
out.release()

print(f"视频已成功转换为{target_fps}fps，并保存为 {output_video}")
