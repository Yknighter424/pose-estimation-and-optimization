import sys
import os

def check_package(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

packages = [
    'cv2',
    'numpy',
    'matplotlib',
    'mediapipe',
    'scipy',
    'pandas'
]

# 將結果寫入文件
with open('package_check_result.txt', 'w', encoding='utf-8') as f:
    f.write(f'Python版本: {sys.version}\n\n')
    f.write('套件安裝狀態：\n')
    for package in packages:
        status = '已安裝 ✓' if check_package(package) else '未安裝 ✗'
        f.write(f'{package}: {status}\n')

print('檢查完成，結果已寫入 package_check_result.txt') 