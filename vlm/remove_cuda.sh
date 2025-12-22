# 强制清理所有 CUDA 进程
sudo fuser -v /dev/nvidia* 2>/dev/null | xargs -r sudo kill -9