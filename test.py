import sys
import os

# 这是一个更健壮的路径获取方法
if getattr(sys, 'frozen', False):
    # 如果是打包后的 exe，路径是可执行文件所在目录
    base_path = os.path.dirname(sys.executable)
else:
    # 如果是脚本运行，路径是脚本所在目录
    base_path = os.path.dirname(os.path.abspath(__file__))

# 然后再加到 PATH
os.environ["PATH"] = base_path + os.pathsep + os.environ["PATH"]

# 针对 Python 3.8+ 的额外保险措施
try:
    os.add_dll_directory(base_path)
except AttributeError:
    pass

import mpv
# 如果这步不报错，说明它成功在当前目录下找到了 mpv-1.dll
player = mpv.MPV() 
print("MPV 后端加载成功！")
player.terminate()