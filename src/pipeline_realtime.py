# -*- coding: utf-8 -*-
"""
实时ToF相机跌倒检测系统
支持24小时持续运行

使用方法:
    # 连接ToF相机（需要实现 ToFCamera 类）
    python pipeline_realtime.py --camera
    
    # 从JSON文件模拟
    python pipeline_realtime.py --json raw_data_1130/raw_data_20251130_123456.json
    
    # 从文件夹模拟
    python pipeline_realtime.py --folder processed_images_320x240
""" 

import os
import sys
import time
import json
import logging
import threading
import queue
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import cv2
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import YOLO_MODEL_PATH, YOLO_CONFIG, RF_CONFIG
from depth_converter import raw_to_depth_image_fast
from pose_estimator import PoseEstimator
from fall_classifier_v6 import FallClassifierV6 as FallClassifier
from keypoint_filter import KeypointFilter, FilterConfig
from visualizer import RealtimeVisualizer


# ======================= 配置 =======================

@dataclass
class RealtimeConfig:
    """实时检测配置"""
    # 滑动窗口（帧数）- V5: 50帧 @ 10 FPS = 5秒
    window_size: int = 50       # 检测窗口大小 (V5: 10 FPS)
    window_stride: int = 10     # 窗口滑动步长（每N帧判断一次）
    
    # 跌倒检测
    fall_confidence_threshold: float = 0.65  # 跌倒置信度阈值
    fall_cooldown_seconds: float = 5.0       # 跌倒报警冷却时间
    fall_consecutive_required: int = 2       # 需要连续N次检测到跌倒才触发
    fall_margin_threshold: float = 0.10      # 最高概率与第二高的最小差距
    fall_min_probability: float = 0.45       # Fall/Backward 概率最低要求
    
    # 关键点过滤
    use_keypoint_filter: bool = True
    min_keypoint_confidence: float = 0.5
    edge_margin: int = 8
    
    # 缓冲区大小
    max_buffer_size: int = 100
    
    # 输出
    save_fall_events: bool = True
    output_dir: str = "fall_events"
    log_dir: str = "logs"
    
    # 性能
    target_fps: int = 10  # V5: 匹配实际相机帧率
    
    # 显示
    show_preview: bool = False


# ======================= 跌倒事件 =======================

@dataclass 
class FallEvent:
    """跌倒事件 (V4: 支持跌倒类型)"""
    event_id: str
    timestamp: datetime
    confidence: float
    keypoints_data: List[np.ndarray]
    frames: List[np.ndarray] = field(default_factory=list)
    fall_type: str = "Fall"  # V4: 'Fall' or 'Backward'
    
    def save(self, output_dir: str):
        """保存跌倒事件"""
        event_dir = os.path.join(output_dir, self.event_id)
        os.makedirs(event_dir, exist_ok=True)
        
        # 保存元信息
        meta = {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'fall_type': self.fall_type,  # V4
            'num_frames': len(self.frames)
        }
        with open(os.path.join(event_dir, 'event.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        # 保存帧
        for i, frame in enumerate(self.frames):
            cv2.imwrite(os.path.join(event_dir, f'frame_{i:04d}.png'), frame)
        
        # 保存关键点CSV
        if self.keypoints_data:
            columns = ['nose_x', 'nose_y', 'nose_conf',
                      'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_conf',
                      'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_conf',
                      'right_hip_x', 'right_hip_y', 'right_hip_conf',
                      'left_hip_x', 'left_hip_y', 'left_hip_conf']
            df = pd.DataFrame(self.keypoints_data, columns=columns)
            df['frame_index'] = range(len(df))
            df.to_csv(os.path.join(event_dir, 'keypoints.csv'), index=False)


# ======================= 日志设置 =======================

def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"fall_detection_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


# ======================= 相机接口 =======================

class ToFCameraBase:
    """ToF相机基类"""
    
    def connect(self) -> bool:
        raise NotImplementedError
    
    def disconnect(self):
        raise NotImplementedError
    
    def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧深度数据 (100x100 或 10000字节)"""
        raise NotImplementedError
    
    def is_connected(self) -> bool:
        raise NotImplementedError


class JsonFileCamera(ToFCameraBase):
    """从JSON文件模拟相机"""
    
    def __init__(self, json_path: str, fps: int = 30, loop: bool = True):
        self.json_path = json_path
        self.fps = fps
        self.loop = loop
        self.frames = []
        self.index = 0
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        self._connected = False
    
    def connect(self) -> bool:
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            for entry in data:
                if isinstance(entry, dict):
                    depth_data = np.array(entry['depth'], dtype=np.uint8)
                else:
                    depth_data = np.array(entry, dtype=np.uint8)
                self.frames.append(depth_data)
            
            self._connected = True
            print(f"[JsonFileCamera] 加载了 {len(self.frames)} 帧数据")
            return True
        except Exception as e:
            print(f"[JsonFileCamera] 加载失败: {e}")
            return False
    
    def disconnect(self):
        self._connected = False
        self.frames = []
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self._connected or len(self.frames) == 0:
            return None
        
        # 检查是否到达末尾
        if self.index >= len(self.frames):
            if self.loop:
                self.index = 0
            else:
                self._connected = False
                return None
        
        # 控制帧率
        now = time.time()
        if now - self.last_frame_time < self.frame_interval:
            return None
        self.last_frame_time = now
        
        frame = self.frames[self.index]
        self.index += 1
        
        return frame
    
    def is_connected(self) -> bool:
        return self._connected


class FolderCamera(ToFCameraBase):
    """从图像文件夹模拟相机"""
    
    def __init__(self, folder_path: str, fps: int = 30, loop: bool = True):
        self.folder_path = folder_path
        self.fps = fps
        self.loop = loop
        self.image_files = []
        self.index = 0
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        self._connected = False
    
    def connect(self) -> bool:
        try:
            exts = ('.png', '.jpg', '.jpeg', '.bmp')
            self.image_files = sorted([
                os.path.join(self.folder_path, f)
                for f in os.listdir(self.folder_path)
                if f.lower().endswith(exts)
            ])
            self._connected = len(self.image_files) > 0
            print(f"[FolderCamera] 加载了 {len(self.image_files)} 张图像")
            return self._connected
        except Exception as e:
            print(f"[FolderCamera] 加载失败: {e}")
            return False
    
    def disconnect(self):
        self._connected = False
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self._connected or len(self.image_files) == 0:
            return None
        
        now = time.time()
        if now - self.last_frame_time < self.frame_interval:
            return None
        self.last_frame_time = now
        
        img = cv2.imread(self.image_files[self.index])
        self.index += 1
        
        if self.index >= len(self.image_files):
            if self.loop:
                self.index = 0
            else:
                return None
        
        return img
    
    def is_connected(self) -> bool:
        return self._connected


# ======================= 实时跌倒检测器 =======================

class RealtimeFallDetector:
    """
    实时跌倒检测器
    支持24小时持续运行
    """
    
    def __init__(self, config: RealtimeConfig = None,
                 on_fall_detected: Callable[[FallEvent], None] = None):
        self.config = config or RealtimeConfig()
        self.on_fall_detected = on_fall_detected
        
        # 日志
        self.logger = setup_logging(self.config.log_dir)
        
        # 初始化模型
        self.logger.info("=" * 50)
        self.logger.info("初始化实时跌倒检测系统")
        self.logger.info("=" * 50)
        
        self.pose_estimator = PoseEstimator()
        self.fall_classifier = FallClassifier()
        
        # 关键点过滤器
        if self.config.use_keypoint_filter:
            filter_config = FilterConfig(
                min_confidence=self.config.min_keypoint_confidence,
                edge_margin=self.config.edge_margin
            )
            self.keypoint_filter = KeypointFilter(filter_config)
        else:
            self.keypoint_filter = None
        
        # 缓冲区
        self.frame_buffer = deque(maxlen=self.config.max_buffer_size)
        self.keypoints_buffer = deque(maxlen=self.config.max_buffer_size)
        
        # 可视化器
        if self.config.show_preview:
            self.visualizer = RealtimeVisualizer()
        else:
            self.visualizer = None
        
        # 状态
        self.running = False
        self.frame_count = 0
        self.fall_count = 0
        self.last_fall_time = 0
        self.start_time = None
        self.current_keypoints = None  # 当前帧的关键点
        self.current_fall_confidence = 0.0  # 当前跌倒置信度
        self.frames_since_last_check = 0
        
        # 连续检测计数（降低误报）
        self.consecutive_fall_count = 0      # 连续检测到跌倒的次数
        self.consecutive_fall_type = None     # 连续检测中的跌倒类型
        self.consecutive_fall_confidence = 0.0  # 累计置信度
        
        # 帧率控制 (确保缓冲区以 target_fps 速率填充)
        self.last_buffer_add_time = 0
        self.buffer_add_interval = 1.0 / self.config.target_fps  # 每帧间隔
        self.buffer_frame_count = 0  # 添加到缓冲区的帧数
        
        # 输出目录
        if self.config.save_fall_events:
            os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.logger.info("系统初始化完成！")
    
    def process_frame(self, depth_data: np.ndarray) -> dict:
        """
        处理单帧深度数据
        
        Args:
            depth_data: 原始深度数据 (10000字节) 或深度图像 (100x100 或 BGR)
        
        Returns:
            dict: {
                'frame_index': int,
                'detected': bool,
                'keypoints': array or None,
                'filtered': bool,
                'fall_detected': bool,
                'fall_confidence': float
            }
        """
        self.frame_count += 1
        result = {
            'frame_index': self.frame_count,
            'detected': False,
            'keypoints': None,
            'filtered': False,
            'fall_detected': False,
            'fall_confidence': 0.0
        }
        
        # 1. 转换深度图
        if len(depth_data.shape) == 1 and len(depth_data) == 10000:
            depth_pil = raw_to_depth_image_fast(depth_data)
            depth_image = np.array(depth_pil)  # 转换为 NumPy 数组
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)  # PIL是RGB，OpenCV是BGR
        elif len(depth_data.shape) == 2 and depth_data.shape == (100, 100):
            depth_image = cv2.applyColorMap(depth_data.astype(np.uint8), cv2.COLORMAP_JET)
        else:
            depth_image = depth_data
        
        # 2. 姿态估计
        pose_result = self.pose_estimator.predict_single(depth_image)
        
        # 3. 转换关键点格式
        if pose_result['detected']:
            kp_list = pose_result['keypoints']
            keypoints = np.array([
                kp_list[0][0], kp_list[0][1], kp_list[0][2],
                kp_list[1][0], kp_list[1][1], kp_list[1][2],
                kp_list[2][0], kp_list[2][1], kp_list[2][2],
                kp_list[3][0], kp_list[3][1], kp_list[3][2],
                kp_list[4][0], kp_list[4][1], kp_list[4][2],
            ])
            result['detected'] = True
            result['keypoints'] = keypoints
            
            # 4. 关键点过滤
            if self.keypoint_filter:
                kp_dict = {
                    'nose_x': keypoints[0], 'nose_y': keypoints[1], 'nose_conf': keypoints[2],
                    'right_shoulder_x': keypoints[3], 'right_shoulder_y': keypoints[4], 'right_shoulder_conf': keypoints[5],
                    'left_shoulder_x': keypoints[6], 'left_shoulder_y': keypoints[7], 'left_shoulder_conf': keypoints[8],
                    'right_hip_x': keypoints[9], 'right_hip_y': keypoints[10], 'right_hip_conf': keypoints[11],
                    'left_hip_x': keypoints[12], 'left_hip_y': keypoints[13], 'left_hip_conf': keypoints[14],
                }
                filter_result = self.keypoint_filter.filter_keypoints(kp_dict, self.frame_count)
                
                if not filter_result['valid'] and filter_result['reason'] != 'no_detection':
                    keypoints = np.zeros(15)
                    result['filtered'] = True
        else:
            keypoints = np.zeros(15)
        
        # 5. 帧率控制：确保缓冲区以 target_fps 填充（避免重复帧导致时间窗口过短）
        current_time = time.time()
        self.frame_buffer.append(depth_image.copy())  # 预览用的缓冲区不限制
        self.current_keypoints = keypoints  # 保存当前关键点供可视化使用
        
        # 只有达到时间间隔才添加到关键点缓冲区（用于RF分类）
        should_add_to_buffer = False
        if self.last_buffer_add_time == 0:
            # 第一帧
            should_add_to_buffer = True
            self.last_buffer_add_time = current_time
        elif current_time - self.last_buffer_add_time >= self.buffer_add_interval:
            should_add_to_buffer = True
            # 关键修复：使用累加而非赋值，避免时间误差累积
            self.last_buffer_add_time += self.buffer_add_interval
            # 防止严重滞后时跳帧（如果落后超过2帧，重置）
            if current_time - self.last_buffer_add_time > self.buffer_add_interval * 2:
                self.last_buffer_add_time = current_time
        
        if should_add_to_buffer:
            self.keypoints_buffer.append(keypoints)
            self.buffer_frame_count += 1
        
        # 6. 检查是否需要进行跌倒检测
        self.frames_since_last_check += 1
        if (len(self.keypoints_buffer) >= self.config.window_size and 
            self.frames_since_last_check >= self.config.window_stride):
            
            fall_result = self._check_fall()
            result['fall_detected'] = fall_result['detected']
            result['fall_confidence'] = fall_result['confidence']
            self.frames_since_last_check = 0
        
        return result
    
    def _check_fall(self) -> dict:
        """检查是否发生跌倒 (V5: 连续检测降低误报)"""
        # 提取窗口数据
        keypoints_window = list(self.keypoints_buffer)[-self.config.window_size:]
        keypoints_array = np.array(keypoints_window)
        
        # 预测 (返回 0=Normal, 1=Fall, 2=Backward)
        prediction = self.fall_classifier.predict(keypoints_array)
        
        result = {
            'detected': False,
            'confidence': 0.0,
            'fall_type': None
        }
        
        if not prediction['valid']:
            self.consecutive_fall_count = 0
            return result
        
        result['confidence'] = prediction['confidence']
        
        # 获取各类概率 [Normal, Fall, Backward]
        probs = prediction.get('probabilities', [0, 0, 0])
        pred_class = prediction['prediction']
        
        # 判断本次是否满足跌倒条件
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        
        # 跌倒判定条件：
        # 1. 预测类别是 Fall(1) 或 Backward(2)
        # 2. margin >= 阈值 (默认0.15)
        # 3. Fall/Backward 概率 >= 最低要求 (默认0.45)
        fall_prob = max(probs[1], probs[2]) if len(probs) >= 3 else 0
        is_fall_this_window = (
            pred_class in [1, 2] and 
            margin >= self.config.fall_margin_threshold and
            fall_prob >= self.config.fall_min_probability
        )
        
        if is_fall_this_window:
            self.consecutive_fall_count += 1
            self.consecutive_fall_type = prediction['label']
            self.consecutive_fall_confidence = max(self.consecutive_fall_confidence, prediction['confidence'])
            
            self.logger.debug(f"连续检测: {self.consecutive_fall_count}/{self.config.fall_consecutive_required} "
                            f"| {prediction['label']} margin={margin:.2f} prob={fall_prob:.2f}")
        else:
            # 正常帧，重置连续计数
            if self.consecutive_fall_count > 0:
                self.logger.info(f"连续检测中断 (count={self.consecutive_fall_count}), "
                               f"pred={prediction['label']}, probs=[N:{probs[0]:.2f}, F:{probs[1]:.2f}, B:{probs[2]:.2f}]")
            self.consecutive_fall_count = 0
            self.consecutive_fall_confidence = 0.0
            self.consecutive_fall_type = None
            return result
        
        # 检查是否达到连续检测要求
        if self.consecutive_fall_count < self.config.fall_consecutive_required:
            print(f"\r[连续检测] {self.consecutive_fall_count}/{self.config.fall_consecutive_required} "
                  f"类型:{self.consecutive_fall_type} 置信度:{self.consecutive_fall_confidence:.2f}    ", end="", flush=True)
            return result
        
        # ===== 达到连续检测要求，确认跌倒 =====
        current_time = time.time()
        
        # 冷却检查
        if current_time - self.last_fall_time < self.config.fall_cooldown_seconds:
            return result
        
        self.last_fall_time = current_time
        self.fall_count += 1
        result['detected'] = True
        result['fall_type'] = self.consecutive_fall_type
        result['confidence'] = self.consecutive_fall_confidence
        
        # 创建跌倒事件
        event = FallEvent(
            event_id=f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.fall_count}",
            timestamp=datetime.now(),
            confidence=self.consecutive_fall_confidence,
            keypoints_data=keypoints_window,
            frames=list(self.frame_buffer)[-self.config.window_size:] if self.config.save_fall_events else [],
            fall_type=self.consecutive_fall_type
        )
        
        self.logger.warning(f"⚠️ 确认跌倒！[{self.consecutive_fall_type}] "
                          f"连续{self.consecutive_fall_count}次检测 "
                          f"置信度: {self.consecutive_fall_confidence:.2f} | 事件ID: {event.event_id}")
        
        # 重置连续计数
        self.consecutive_fall_count = 0
        self.consecutive_fall_confidence = 0.0
        self.consecutive_fall_type = None
        
        # 保存事件
        if self.config.save_fall_events:
            try:
                event.save(self.config.output_dir)
                self.logger.info(f"跌倒事件已保存: {event.event_id}")
            except Exception as e:
                self.logger.error(f"保存跌倒事件失败: {e}")
        
        # 回调
        if self.on_fall_detected:
            try:
                self.on_fall_detected(event)
            except Exception as e:
                self.logger.error(f"回调执行失败: {e}")
        
        return result
    
    def run(self, camera: ToFCameraBase):
        """
        运行检测循环
        
        Args:
            camera: 相机对象
        """
        self.running = True
        self.start_time = time.time()
        
        self.logger.info("开始实时检测...")
        
        frame_interval = 1.0 / self.config.target_fps
        last_status_time = time.time()
        
        try:
            while self.running and camera.is_connected():
                frame_start = time.time()
                
                # 获取帧
                depth_data = camera.get_frame()
                if depth_data is None:
                    time.sleep(0.01)
                    continue
                
                # 处理帧
                result = self.process_frame(depth_data)
                
                # 更新跌倒置信度
                if result['fall_confidence'] > 0:
                    self.current_fall_confidence = result['fall_confidence']
                
                # 显示预览（使用可视化器）
                if self.visualizer and len(self.frame_buffer) > 0:
                    preview = self.frame_buffer[-1].copy()
                    
                    # 计算FPS
                    runtime = time.time() - self.start_time if self.start_time else 0
                    fps = self.frame_count / runtime if runtime > 0 else 0
                    
                    # 绘制骨架
                    if self.current_keypoints is not None and result['detected'] and not result['filtered']:
                        preview = self.visualizer.draw_skeleton(
                            preview, 
                            self.current_keypoints,
                            is_fall=result['fall_detected']
                        )
                    
                    # 绘制状态覆盖
                    status = "FALL DETECTED!" if result['fall_detected'] else "Monitoring..."
                    preview = self.visualizer.draw_status_overlay(
                        preview,
                        status_text=status,
                        fps=fps,
                        buffer_size=len(self.keypoints_buffer),
                        frame_count=self.frame_count,
                        fall_count=self.fall_count
                    )
                    
                    # 绘制跌倒警报
                    if result['fall_detected'] or self.current_fall_confidence > 0.5:
                        preview = self.visualizer.draw_fall_alert(
                            preview,
                            confidence=self.current_fall_confidence,
                            duration=time.time() - self.last_fall_time if self.last_fall_time > 0 else 0
                        )
                    
                    # 显示帧
                    if not self.visualizer.show_frame(preview, 'Fall Detection - ToF'):
                        self.running = False
                
                # 定期输出状态
                if time.time() - last_status_time > 10:
                    runtime = time.time() - self.start_time
                    fps = self.frame_count / runtime if runtime > 0 else 0
                    self.logger.info(f"状态: 帧数={self.frame_count}, FPS={fps:.1f}, 跌倒={self.fall_count}")
                    last_status_time = time.time()
                
                # 帧率控制
                elapsed = time.time() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止检测...")
        finally:
            self.running = False
            if self.visualizer:
                self.visualizer.close()
            
            runtime = time.time() - self.start_time if self.start_time else 0
            self.logger.info(f"检测结束 | 运行时间: {runtime/3600:.2f}小时 | "
                           f"处理帧数: {self.frame_count} | 检测跌倒: {self.fall_count}次")
    
    def get_status(self) -> dict:
        """获取检测器状态"""
        runtime = time.time() - self.start_time if self.start_time else 0
        return {
            'running': self.running,
            'runtime_hours': runtime / 3600,
            'frame_count': self.frame_count,
            'fall_count': self.fall_count,
            'buffer_size': len(self.frame_buffer),
            'fps': self.frame_count / runtime if runtime > 0 else 0
        }


# ======================= 主程序 =======================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='实时ToF相机跌倒检测')
    parser.add_argument('--json', type=str, help='JSON数据文件路径')
    parser.add_argument('--folder', type=str, help='图像文件夹路径')
    parser.add_argument('--fps', type=int, default=30, help='目标帧率')
    parser.add_argument('--window', type=int, default=150, help='检测窗口大小')
    parser.add_argument('--threshold', type=float, default=0.65, help='跌倒置信度阈值')
    parser.add_argument('--no-filter', action='store_true', help='禁用关键点过滤')
    parser.add_argument('--preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--output', type=str, default='fall_events', help='输出目录')
    parser.add_argument('--loop', action='store_true', help='循环播放（用于测试）')
    
    args = parser.parse_args()
    
    # 配置
    config = RealtimeConfig(
        window_size=args.window,
        fall_confidence_threshold=args.threshold,
        use_keypoint_filter=not args.no_filter,
        show_preview=args.preview,
        output_dir=args.output,
        target_fps=args.fps
    )
    
    # 回调函数
    def on_fall(event):
        print(f"\n🚨 警报！检测到跌倒！")
        print(f"   时间: {event.timestamp}")
        print(f"   置信度: {event.confidence:.2f}")
        print(f"   事件ID: {event.event_id}")
        # 这里可以添加其他通知：发送邮件、推送通知等
    
    # 创建相机
    if args.json:
        camera = JsonFileCamera(args.json, fps=args.fps, loop=args.loop)
    elif args.folder:
        camera = FolderCamera(args.folder, fps=args.fps, loop=args.loop)
    else:
        print("请指定 --json 或 --folder 参数")
        print("示例:")
        print("  python pipeline_realtime.py --json raw_data_1130/raw_data_20251130_123456.json --preview")
        print("  python pipeline_realtime.py --folder processed_images_320x240 --preview --loop")
        return
    
    # 连接相机
    if not camera.connect():
        print("相机连接失败")
        return
    
    # 创建检测器并运行
    detector = RealtimeFallDetector(config, on_fall_detected=on_fall)
    
    try:
        detector.run(camera)
    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()
