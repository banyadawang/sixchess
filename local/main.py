import sys
import os

# Windows 中文控制台默认 GBK 编码, 无法输出 emoji (app 日志大量使用):
# 兜底避免 AI 线程 print 崩溃 (UnicodeEncodeError)
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() != "utf-8":
        _stream.reconfigure(errors="replace")

# 修复 PyQt5 与 torch 的 MSVC 运行时冲突:
# PyQt5 自带 VS2015 时代的老版 msvcp140.dll/vcruntime140.dll (位于 PyQt5/Qt5/bin),
# 若先于 torch 加载, torch 的 c10.dll (VS2022 编译) 绑定到老运行时会导致
# OSError: [WinError 1114] DLL 初始化例程失败。
# 先显式加载系统 (System32) 的新版运行时, 老版本就不会被加载。
import ctypes
for _dll in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140_1.dll", "concrt140.dll"):
    try:
        ctypes.CDLL(_dll)
    except OSError:
        pass

# Add subdirectories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == '__main__':
    # Ensure working directory is correct
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Use Fusion style for better CSS support
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
