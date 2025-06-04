# 113Camera1 - 计算机视觉项目

## 项目简介
这是一个计算机视觉项目，主要实现相机标定、立体视觉和ArUco标记检测功能。

## 主要功能
- 单相机标定
- 立体相机标定
- 多相机系统标定
- ArUco标记检测与跟踪
- 3D点云处理
- 束调整优化

## 主要文件说明
- `113ARfinally.py` - ArUco标记检测主程序
- `113Fancy.py` - 增强版图像处理程序
- `Mutiple_camera.py` - 多相机系统处理
- `Three_C_Bundle.py` - 三相机束调整
- `stereo_camera_calibration.npz` - 立体相机标定参数
- `calibration_left.npz` / `calibration_right.npz` - 左右相机标定参数

## 环境要求
- Python 3.x
- OpenCV
- NumPy
- SciPy
- matplotlib

## 安装和运行
1. 安装依赖包：
   ```bash
   pip install opencv-python numpy scipy matplotlib
   ```

2. 运行标定程序：
   ```bash
   python opencv_test.py  # 测试OpenCV环境
   ```

3. 检查环境：
   ```bash
   python check_env.py
   ```

## 注意事项
- 确保相机正确连接
- 标定板准备充分
- 光照条件良好

## 开发者
- 项目开发用于相机标定和计算机视觉研究

## 许可证
本项目仅供学习和研究使用。 