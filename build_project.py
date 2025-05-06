import os
import shutil
import subprocess
import platform

build_dir = "build"
source_dir = "."

# if os.path.exists(build_dir):
#     print(f"> Removing existing build directory: {build_dir}")
#     shutil.rmtree(build_dir)

# print(f"> Creating build directory: {build_dir}")
# os.makedirs(build_dir)

cmake_configure_cmd = [
    "cmake",
    "-G", "MinGW Makefiles",
    "-DCMAKE_C_COMPILER=gcc",
    "-DCMAKE_CXX_COMPILER=g++",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-B", build_dir,
    "-S", source_dir
]
print(f"> Running command: {' '.join(cmake_configure_cmd)}")
subprocess.check_call(cmake_configure_cmd)

cpu_count = os.cpu_count() or 1
print(f"> Detected {cpu_count} CPU cores")

cmake_build_cmd = [
    "cmake", 
    "--build", build_dir,
    # "--verbose",
    "--", f"-j{cpu_count}"
]
print(f"> Running command: {' '.join(cmake_build_cmd)}")
subprocess.check_call(cmake_build_cmd)

exe_path = os.path.join(build_dir, "main.exe")
print(f"> Running executable: {exe_path}")
# env = os.environ.copy()
subprocess.run([exe_path])
