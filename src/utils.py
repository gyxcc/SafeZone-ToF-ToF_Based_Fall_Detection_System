# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import os
import time
from datetime import datetime


def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def list_image_files(folder_path, extensions=None):
    """
    列出文件夹中的所有图像文件
    
    Args:
        folder_path: 文件夹路径
        extensions: 文件扩展名列表，默认 ['.png', '.jpg', '.jpeg']
    
    Returns:
        list: 排序后的文件名列表
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    
    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in extensions
    ]
    return sorted(files)


def list_json_files(folder_path):
    """列出文件夹中的所有JSON文件"""
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    return sorted(files)


class FPSCounter:
    """帧率计数器"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []
    
    def tick(self):
        """记录一帧"""
        now = time.time()
        self.timestamps.append(now)
        
        # 保持窗口大小
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self):
        """获取当前帧率"""
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        
        return (len(self.timestamps) - 1) / elapsed


class RingBuffer:
    """环形缓冲区，用于存储最近N帧的数据"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def append(self, item):
        """添加元素"""
        self.buffer.append(item)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def get_all(self):
        """获取所有元素"""
        return self.buffer.copy()
    
    def is_full(self):
        """检查是否已满"""
        return len(self.buffer) >= self.capacity
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


def print_banner(text, char='=', width=60):
    """打印横幅"""
    print(char * width)
    print(text.center(width))
    print(char * width)


def format_duration(seconds):
    """格式化时长"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
