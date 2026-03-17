# -*- coding: utf-8 -*-
"""
深度数据转换模块
将ToF原始数据转换为深度图像
"""

import numpy as np
from PIL import Image
import json
import os

from config import COLOR_MAP, TOF_CONFIG


def raw_to_depth_image(raw_data, width=None, height=None):
    """
    将原始深度数据转换为RGB深度图像（内存中）
    
    Args:
        raw_data: 一维深度数据列表 (长度应为 width * height)
        width: 图像宽度，默认从配置读取
        height: 图像高度，默认从配置读取
    
    Returns:
        PIL.Image: RGB深度图像
    """
    if width is None:
        width = TOF_CONFIG['frame_width']
    if height is None:
        height = TOF_CONFIG['frame_height']
    
    expected_length = width * height
    
    # 处理数据长度问题
    if len(raw_data) == expected_length + 1:
        raw_data = raw_data[:-1]
    elif len(raw_data) != expected_length:
        raise ValueError(f"数据长度 {len(raw_data)} 不匹配预期 {expected_length}")
    
    # 转换为2D数组
    frame = np.array(raw_data).reshape((height, width))
    
    # 应用颜色映射
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            depth = int(frame[row, col])
            image_array[row, col] = COLOR_MAP[int(depth) % len(COLOR_MAP)]
    
    return Image.fromarray(image_array)


def raw_to_depth_image_fast(raw_data, width=None, height=None):
    """
    快速版本：使用向量化操作将原始深度数据转换为RGB深度图像
    
    Args:
        raw_data: 一维深度数据列表
        width: 图像宽度
        height: 图像高度
    
    Returns:
        PIL.Image: RGB深度图像
    """
    if width is None:
        width = TOF_CONFIG['frame_width']
    if height is None:
        height = TOF_CONFIG['frame_height']
    
    expected_length = width * height
    
    if len(raw_data) == expected_length + 1:
        raw_data = raw_data[:-1]
    elif len(raw_data) != expected_length:
        raise ValueError(f"数据长度 {len(raw_data)} 不匹配预期 {expected_length}")
    
    # 转换为numpy数组
    depth_array = np.array(raw_data, dtype=np.uint8).reshape((height, width))
    
    # 使用查找表进行向量化颜色映射
    color_lut = np.array(COLOR_MAP, dtype=np.uint8)
    indices = depth_array.astype(np.int32) % len(COLOR_MAP)
    image_array = color_lut[indices]
    
    return Image.fromarray(image_array)


def load_json_frames(json_path):
    """
    从JSON文件加载所有帧数据
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        list: 帧数据列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)
    return frames


def convert_json_to_images(json_path, output_dir):
    """
    将JSON文件中的所有帧转换为图像并保存
    
    Args:
        json_path: JSON文件路径
        output_dir: 输出目录
    
    Returns:
        list: 保存的图像路径列表
    """
    frames = load_json_frames(json_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_paths = []
    for i, frame_data in enumerate(frames):
        try:
            img = raw_to_depth_image_fast(frame_data)
            output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            img.save(output_path)
            saved_paths.append(output_path)
        except Exception as e:
            print(f"转换第 {i} 帧失败: {e}")
    
    return saved_paths


if __name__ == "__main__":
    # 测试代码
    print("深度转换模块测试")
    
    # 创建测试数据
    test_data = list(range(256)) * 39 + list(range(16))  # 10000个数据点
    print(f"测试数据长度: {len(test_data)}")
    
    img = raw_to_depth_image_fast(test_data)
    print(f"生成图像大小: {img.size}")
    
    # 保存测试图像
    test_output = os.path.join(os.path.dirname(__file__), 'output', 'test_depth.png')
    os.makedirs(os.path.dirname(test_output), exist_ok=True)
    img.save(test_output)
    print(f"测试图像已保存: {test_output}")
