"""
编译 MCTS 引擎 (跨平台支持 Windows/Linux)
使用方法: python compile_dll.py
"""
import subprocess
import os
import sys
import shutil
import platform

# Windows 中文控制台默认 GBK 编码, 无法输出 emoji; 兜底避免 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(errors="replace")

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 目标目录
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core")

def find_vs_dev_cmd():
    """通过 vswhere 自动定位 VsDevCmd.bat (Visual Studio 2017+ / Build Tools)"""
    for vswhere in [
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
    ]:
        if not os.path.exists(vswhere):
            continue
        try:
            result = subprocess.run(
                [vswhere, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            continue
        install = result.stdout.strip()
        if install:
            devcmd = os.path.join(install, "Common7", "Tools", "VsDevCmd.bat")
            if os.path.exists(devcmd):
                return devcmd
    return None

def compile_windows():
    """Windows: 使用 MSVC (cl.exe)"""
    cl_cmd = 'cl /utf-8 /LD /std:c++17 /EHsc /O2 /openmp mcts_engine.cpp /I . /Fe:mcts.dll'

    if shutil.which("cl"):
        # cl 已在 PATH 中 (开发者命令提示符)
        cmd = cl_cmd
    else:
        # 普通终端: 自动定位 VS 并加载编译环境
        devcmd = find_vs_dev_cmd()
        if not devcmd:
            print("❌ 错误: 找不到 cl.exe")
            print("请安装 Visual Studio 2022 Build Tools (勾选 '使用 C++ 的桌面开发' 工作负载)")
            print("或从 'x64 Native Tools Command Prompt for VS 2022' 运行此脚本")
            return 1
        print(f"🔧 自动加载编译环境: {devcmd}")
        cmd = f'call "{devcmd}" -arch=x64 -host_arch=x64 >nul && {cl_cmd}'

    print("🔨 [Windows] 编译 mcts_engine.cpp...")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print("❌ 编译失败")
        return result.returncode
    
    # 复制 DLL 到 local/core
    if os.path.exists("mcts.dll"):
        target_path = os.path.join(TARGET_DIR, "mcts.dll")
        shutil.copy("mcts.dll", target_path)
        print(f"✅ 编译成功，已复制到: {target_path}")
    else:
        print("❌ 找不到编译输出的 mcts.dll")
        return 1
    
    # 清理中间文件
    for f in ["mcts.obj", "mcts_engine.obj", "mcts.exp", "mcts.lib"]:
        if os.path.exists(f):
            os.remove(f)
    
    return 0

def compile_linux():
    """Linux: 使用 g++"""
    if not shutil.which("g++"):
        print("❌ 错误: 找不到 g++")
        print("请安装 g++: sudo apt install g++")
        return 1
    
    print("🔨 [Linux] 编译 mcts_engine.cpp...")
    
    output_path = os.path.join(TARGET_DIR, "libmcts.so")
    cmd = f'g++ -shared -fPIC -O3 -fopenmp -std=c++17 -I. mcts_engine.cpp -o {output_path}'
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print("❌ 编译失败")
        return result.returncode
    
    if os.path.exists(output_path):
        print(f"✅ 编译成功: {output_path}")
    else:
        print("❌ 找不到编译输出的 libmcts.so")
        return 1
    
    return 0

def compile_and_copy():
    system = platform.system()
    
    if system == "Windows":
        return compile_windows()
    elif system == "Linux":
        return compile_linux()
    else:
        print(f"❌ 不支持的操作系统: {system}")
        return 1

if __name__ == "__main__":
    sys.exit(compile_and_copy())
