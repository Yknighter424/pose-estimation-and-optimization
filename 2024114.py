import os

# .lnk 文件的完整路徑
shortcut_path = r"C:\Users\user\Desktop\DeepL.lnk"

# 嘗試打開 .lnk 文件
try:
    os.startfile(shortcut_path)
    print("已成功打開捷徑。")
except Exception as e:
    print(f"無法打開捷徑: {e}")
import subprocess

# 使用 subprocess 啟動捷徑
try:
    subprocess.Popen(shortcut_path, shell=True)
    print("已成功打開捷徑。")
except Exception as e:
    print(f"無法打開捷徑: {e}")
