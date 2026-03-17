# -*- coding: utf-8 -*-
"""
YOLOv8-pose 姿态估计模块
负责从深度图像中检测人体姿态关键点
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from config import YOLO_MODEL_PATH, YOLO_CONFIG, KEYPOINT_NAMES, CSV_COLUMNS


class PoseEstimator:
    """YOLOv8姿态估计器"""
    
    def __init__(self, model_path=None, device='cuda'):
        """
        初始化姿态估计器
        
        Args:
            model_path: 模型路径，默认使用配置文件中的路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        if model_path is None:
            model_path = YOLO_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 检查 CUDA 可用性
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA 不可用，使用 CPU")
            device = 'cpu'
        
        self.device = device
        print(f"加载YOLO模型: {model_path} (device={device})")
        self.model = YOLO(model_path)
        self.conf_threshold = YOLO_CONFIG['conf_threshold']
    
    def predict_single(self, image):
        """
        对单张图像进行姿态估计
        
        Args:
            image: PIL.Image 或 图像路径 或 numpy数组
        
        Returns:
            dict: 关键点数据，格式为 {
                'keypoints': [[x, y, conf], ...],  # 5个关键点
                'bbox': [x1, y1, x2, y2, conf],    # 边界框
                'detected': bool                   # 是否检测到人体
            }
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )
        
        result = results[0]
        
        # 检查是否检测到人体
        if len(result.boxes) == 0 or result.keypoints is None:
            return {
                'keypoints': [[0.0, 0.0, 0.0] for _ in range(5)],
                'bbox': None,
                'detected': False
            }
        
        # 获取第一个检测到的人体（假设只有一个人）
        keypoints = result.keypoints.data[0].cpu().numpy()  # shape: (5, 3)
        bbox = result.boxes.data[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        
        return {
            'keypoints': keypoints.tolist(),
            'bbox': bbox[:5].tolist(),  # 只取前5个值
            'detected': True
        }
    
    def predict_batch(self, images):
        """
        批量处理图像
        
        Args:
            images: 图像列表（路径或PIL.Image）
        
        Returns:
            list: 每张图像的关键点数据列表
        """
        all_results = []
        
        for img in images:
            result = self.predict_single(img)
            all_results.append(result)
        
        return all_results
    
    def predict_folder(self, folder_path, save_csv=True, output_path=None):
        """
        处理文件夹中的所有图像
        
        Args:
            folder_path: 图像文件夹路径
            save_csv: 是否保存CSV文件
            output_path: CSV输出路径
        
        Returns:
            pd.DataFrame: 关键点数据表
        """
        # 获取所有图像文件
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        
        if not image_files:
            print(f"警告: 文件夹 {folder_path} 中没有找到图像文件")
            return None
        
        print(f"处理 {len(image_files)} 张图像...")
        
        # 收集所有结果
        rows = []
        for i, filename in enumerate(image_files):
            image_path = os.path.join(folder_path, filename)
            result = self.predict_single(image_path)
            
            # 构建行数据
            row = [i]  # frame_index
            for kp in result['keypoints']:
                row.extend(kp)  # x, y, conf
            rows.append(row)
            
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(image_files)} 张")
        
        # 创建DataFrame
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        
        # 保存CSV
        if save_csv:
            if output_path is None:
                output_path = os.path.join(folder_path, 'keypoints.csv')
            df.to_csv(output_path, index=False)
            print(f"关键点数据已保存: {output_path}")
        
        return df
    
    def keypoints_to_csv_row(self, frame_index, keypoints_data):
        """
        将关键点数据转换为CSV行格式
        
        Args:
            frame_index: 帧索引
            keypoints_data: predict_single 返回的数据
        
        Returns:
            list: CSV行数据
        """
        row = [frame_index]
        for kp in keypoints_data['keypoints']:
            row.extend(kp)
        return row


def extract_keypoints_from_results(results):
    """
    从YOLO结果中提取关键点坐标（用于实时处理）
    
    Args:
        results: YOLO预测结果
    
    Returns:
        numpy.ndarray: shape (5, 3) 的关键点数组，或全0数组
    """
    if len(results) == 0:
        return np.zeros((5, 3))
    
    result = results[0]
    
    if len(result.boxes) == 0 or result.keypoints is None:
        return np.zeros((5, 3))
    
    return result.keypoints.data[0].cpu().numpy()


if __name__ == "__main__":
    # 测试代码
    print("姿态估计模块测试")
    
    # 初始化估计器
    try:
        estimator = PoseEstimator()
        print("模型加载成功！")
        
        # 测试单张图像（如果有的话）
        test_image_dir = r'C:\Users\LTTC\Desktop\fyp\fall\raw_data_20251106_150429'
        if os.path.exists(test_image_dir):
            test_images = [f for f in os.listdir(test_image_dir) if f.endswith('.png')]
            if test_images:
                test_path = os.path.join(test_image_dir, test_images[0])
                result = estimator.predict_single(test_path)
                print(f"测试结果: detected={result['detected']}")
                if result['detected']:
                    print(f"关键点: {result['keypoints']}")
    except Exception as e:
        print(f"测试失败: {e}")
