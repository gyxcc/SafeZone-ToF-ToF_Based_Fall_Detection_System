# -*- coding: utf-8 -*-
"""
真实 ToF 相机跌倒检测系统
连接串口 ToF 相机并进行实时跌倒检测

使用方法:
    python tof_fall_detection.py                    # 使用默认 COM3
    python tof_fall_detection.py --port COM4        # 指定串口
    python tof_fall_detection.py --preview          # 显示预览窗口
    python tof_fall_detection.py --duration 3600    # 运行1小时
"""

import os
import sys
import time
import json
import serial
import serial.tools.list_ports
from struct import unpack
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading
import queue

import numpy as np
import cv2
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import YOLO_MODEL_PATH
from depth_converter import raw_to_depth_image_fast
from pose_estimator import PoseEstimator
from fall_classifier_v6 import FallClassifierV6 as FallClassifier
from keypoint_filter import KeypointFilter, FilterConfig
from pipeline_realtime import RealtimeConfig, RealtimeFallDetector, FallEvent, ToFCameraBase


# ======================= 跌倒监控状态机 =======================

class MonitorState(Enum):
    """监控状态"""
    IDLE = "idle"                    # 空闲，正常监测
    MONITORING = "monitoring"        # 检测到跌倒，等待确认
    ALERTED = "alerted"              # 已触发报警


@dataclass
class FallMonitorEvent:
    """跌倒监控事件"""
    event_id: str
    fall_time: float                 # 检测到跌倒的时间戳
    rf_confidence: float             # RF 置信度
    fall_type: str = "Fall"          # Fall / Backward
    recovery_time: Optional[float] = None
    alert_triggered: bool = False
    status: str = "monitoring"       # monitoring / recovered / alerted


class FallMonitor:
    """
    跌倒监控器
    
    核心逻辑：
    1. RF 检测到跌倒 → 进入 MONITORING 状态
    2. 持续检查人是否恢复（重新检测到正常姿态）
    3. 超过 ALERT_DELAY 无恢复 → 触发真正报警
    4. 恢复正常 → 取消警报，回到 IDLE
    """
    
    def __init__(self, fps: float = 10.0):
        self.fps = fps
        
        # 状态
        self.state = MonitorState.IDLE
        self.current_event: Optional[FallMonitorEvent] = None
        
        # 参数配置
        self.ALERT_DELAY = 10.0           # 跌倒后多久无恢复则报警（秒）
        self.RECOVERY_CONFIRM_FRAMES = 5  # 需要连续N帧正常才算恢复
        self.QUICK_RECOVERY_TIME = 3.0    # 快速恢复时间（可能是误报）
        
        # 恢复检测缓冲
        self.normal_streak = 0            # 连续正常帧计数
        self.all_keypoints_streak = 0     # 连续检测到全部5个点的帧数
        self.RECOVERY_ALL_POINTS_REQUIRED = 5  # 需要连续N帧全部点可见才算恢复
        
        # 统计
        self.total_falls = 0
        self.total_recoveries = 0
        self.total_alerts = 0
        
        # 事件历史
        self.event_history: list = []
    
    def on_fall_detected(self, confidence: float, fall_type: str = "Fall") -> tuple:
        """
        RF 检测到跌倒时调用
        
        Returns:
            (是否记录为新事件, 消息)
        """
        now = time.time()
        
        # 如果已经在监测或已报警，不重复记录
        if self.state in [MonitorState.MONITORING, MonitorState.ALERTED]:
            return False, f"Already in {self.state.value} state"
        
        # 创建新事件
        self.total_falls += 1
        event_id = f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_falls}"
        
        self.current_event = FallMonitorEvent(
            event_id=event_id,
            fall_time=now,
            rf_confidence=confidence,
            fall_type=fall_type
        )
        
        self.state = MonitorState.MONITORING
        self.normal_streak = 0
        
        print(f"\n[Monitor] 📋 检测到跌倒，开始监控")
        print(f"          类型: {fall_type} | 置信度: {confidence:.1%}")
        print(f"          等待 {self.ALERT_DELAY}s 确认...")
        
        return True, event_id
    
    def on_normal_detected(self) -> tuple:
        """
        RF 检测到正常姿态时调用
        
        Returns:
            (是否恢复, 消息)
        """
        if self.state == MonitorState.IDLE:
            self.normal_streak = 0
            return False, "Already idle"
        
        self.normal_streak += 1
        
        # 需要连续多帧正常才算真正恢复
        if self.normal_streak >= self.RECOVERY_CONFIRM_FRAMES:
            return self._handle_recovery()
        
        return False, f"Waiting for recovery confirmation ({self.normal_streak}/{self.RECOVERY_CONFIRM_FRAMES})"
    
    def update(self, is_fall: bool, confidence: float = 0.0, fall_type: str = "Normal", 
                keypoints: np.ndarray = None) -> tuple:
        """
        每次RF预测后调用
        
        Args:
            is_fall: RF 是否检测到跌倒
            confidence: 置信度
            fall_type: 类型 (Normal/Fall/Backward)
            keypoints: 当前帧的关键点数据 (15元素: 5个点 x 3)
        
        Returns:
            (当前状态, 报警消息或None)
        """
        now = time.time()
        
        # 检查是否全部5个关键点都检测到（置信度 > 0.3）
        all_points_visible = False
        if keypoints is not None and len(keypoints) >= 15:
            # 5个点的置信度在 index 2, 5, 8, 11, 14
            conf_indices = [2, 5, 8, 11, 14]
            visible_count = sum(1 for idx in conf_indices if keypoints[idx] > 0.3)
            all_points_visible = (visible_count == 5)
        
        # 1. 处理检测结果
        if is_fall:
            self.normal_streak = 0
            self.all_keypoints_streak = 0
            if self.state == MonitorState.IDLE:
                self.on_fall_detected(confidence, fall_type)
        else:
            if self.state != MonitorState.IDLE:
                # 恢复检测：需要RF说Normal且全部5个点可见
                if all_points_visible:
                    self.all_keypoints_streak += 1
                    if self.all_keypoints_streak >= self.RECOVERY_ALL_POINTS_REQUIRED:
                        recovered, msg = self.on_normal_detected()
                        if recovered:
                            return self.state, None
                else:
                    self.all_keypoints_streak = 0
        
        # 2. 检查是否需要触发报警
        if self.state == MonitorState.MONITORING and self.current_event:
            elapsed = now - self.current_event.fall_time
            
            if elapsed >= self.ALERT_DELAY:
                alert_msg = self._trigger_alert()
                return self.state, alert_msg
            
            # 显示倒计时
            remaining = self.ALERT_DELAY - elapsed
            if int(elapsed * 2) % 2 == 0:  # 每0.5秒更新
                print(f"\r[Monitor] ⏱️  监控中 {elapsed:.1f}s / 报警倒计时: {remaining:.1f}s    ", end="", flush=True)
        
        return self.state, None
    
    def _handle_recovery(self) -> tuple:
        """处理恢复"""
        if self.current_event is None:
            self.state = MonitorState.IDLE
            return False, "No event"
        
        now = time.time()
        duration = now - self.current_event.fall_time
        
        self.current_event.recovery_time = now
        self.current_event.status = "recovered"
        self.total_recoveries += 1
        
        # 快速恢复提示
        if duration < self.QUICK_RECOVERY_TIME:
            print(f"\n[Monitor] ⚡ 快速恢复 ({duration:.1f}s) - 可能是误报")
        else:
            print(f"\n[Monitor] ✅ 恢复正常! 持续时间: {duration:.1f}s")
        
        self.event_history.append(self.current_event)
        self.current_event = None
        self.state = MonitorState.IDLE
        self.normal_streak = 0
        self.all_keypoints_streak = 0
        
        return True, f"Recovered after {duration:.1f}s"
    
    def _trigger_alert(self) -> str:
        """触发真正的报警"""
        if self.current_event is None:
            return ""
        
        now = time.time()
        duration = now - self.current_event.fall_time
        
        self.current_event.alert_triggered = True
        self.current_event.status = "alerted"
        self.total_alerts += 1
        self.state = MonitorState.ALERTED
        
        self.event_history.append(self.current_event)
        
        alert_msg = (
            f"\n{'='*60}\n"
            f"🚨🚨🚨 紧急报警! FALL ALERT! 🚨🚨🚨\n"
            f"{'='*60}\n"
            f"事件ID: {self.current_event.event_id}\n"
            f"跌倒类型: {self.current_event.fall_type}\n"
            f"无响应时长: {duration:.1f} 秒\n"
            f"置信度: {self.current_event.rf_confidence:.1%}\n"
            f"{'='*60}\n"
        )
        
        print(alert_msg)
        
        return alert_msg
    
    def reset(self):
        """手动重置（报警确认后）"""
        print(f"[Monitor] 🔄 手动重置")
        self.current_event = None
        self.state = MonitorState.IDLE
        self.normal_streak = 0
        self.all_keypoints_streak = 0
    
    def get_status(self) -> dict:
        """获取状态"""
        return {
            "state": self.state.value,
            "current_event": self.current_event.event_id if self.current_event else None,
            "normal_streak": self.normal_streak,
            "all_keypoints_streak": self.all_keypoints_streak,
            "total_falls": self.total_falls,
            "total_recoveries": self.total_recoveries,
            "total_alerts": self.total_alerts
        }


# ======================= ToF 相机配置 =======================

class ToFConfig:
    """ToF 相机配置"""
    # 串口设置
    DEFAULT_PORT = "COM3"
    INIT_BAUDRATE = 115200
    WORK_BAUDRATE = 1000000  # 与 ToF_simple_gemini.py 一致
    
    # 帧格式
    FRAME_HEAD = b"\x00\xFF"
    FRAME_TAIL = b"\xDD"
    FRAME_WIDTH = 100
    FRAME_HEIGHT = 100
    DATA_LENGTH = FRAME_WIDTH * FRAME_HEIGHT  # 10000 bytes
    
    # AT 命令
    AT_COMMANDS = [
        b"AT+BINN=1\r",
        b"AT+UNIT=0\r",
        b"AT+DISP=2\r",
        b"AT+FPS=15\r",  # 15 FPS (提高帧率)
        b"AT+BAUD=8\r",
        b"AT+SAVE\r",
    ]


# ======================= 真实 ToF 相机类 =======================

class RealToFCamera(ToFCameraBase):
    """
    真实 ToF 相机接口
    通过串口连接 100x100 深度传感器
    """
    
    def __init__(self, port: str = None, timeout: float = 1.0):
        """
        初始化 ToF 相机
        
        Args:
            port: 串口号 (如 "COM3")，None 则自动检测
            timeout: 串口超时时间
        """
        self.port = port or ToFConfig.DEFAULT_PORT
        self.timeout = timeout
        self.serial = serial.Serial()
        self._connected = False
        self._running = False
        
        # 数据缓冲
        self.raw_buffer = b''
        self.frame_queue = queue.Queue(maxsize=10)
        self.frame_count = 0
        
        # 读取线程
        self.read_thread = None
    
    def list_ports(self):
        """列出可用串口"""
        ports = serial.tools.list_ports.comports()
        print("可用串口:")
        for p in ports:
            print(f"  {p.device}: {p.description}")
        return [p.device for p in ports]
    
    def connect(self) -> bool:
        """连接 ToF 相机"""
        try:
            # 初始连接（低波特率发送AT命令）
            print(f"[ToF] 连接串口 {self.port}...")
            self.serial.baudrate = ToFConfig.INIT_BAUDRATE
            self.serial.port = self.port
            self.serial.bytesize = serial.EIGHTBITS
            self.serial.parity = serial.PARITY_NONE
            self.serial.stopbits = serial.STOPBITS_ONE
            self.serial.timeout = self.timeout
            self.serial.open()
            
            # 发送初始化命令（完全复制 ToF_simple_gemini.py 的逻辑）
            print("[ToF] 发送初始化命令...")
            self.serial.write(b"AT+BINN=1\r")
            self.serial.write(b"AT+UNIT=0\r")
            self.serial.write(b"AT+DISP=2\r")
            self.serial.write(b"AT+FPS=10\r")
            self.serial.write(b"AT+BAUD=8\r")
            self.serial.write(b"AT+SAVE\r")
            # 关键：不加任何延时，立即切换波特率
            
            # 立即切换到高波特率
            print(f"[ToF] 切换到 {ToFConfig.WORK_BAUDRATE} 波特率...")
            self.serial.close()
            self.serial.baudrate = ToFConfig.WORK_BAUDRATE
            self.serial.open()
            
            self._connected = True
            self._running = True
            
            # 启动读取线程
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            print("[ToF] 相机连接成功！")
            return True
            
        except serial.SerialException as e:
            print(f"[ToF] 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        self._connected = False
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2)
        
        if self.serial.is_open:
            self.serial.close()
        
        print("[ToF] 相机已断开")
    
    def _read_loop(self):
        """后台读取线程"""
        while self._running:
            try:
                if self.serial.in_waiting > 0:
                    self.raw_buffer += self.serial.read(self.serial.in_waiting)
                    self._parse_frames()
                else:
                    time.sleep(0.001)
            except Exception as e:
                if self._running:
                    print(f"[ToF] 读取错误: {e}")
                break
    
    def _parse_frames(self):
        """解析帧数据"""
        while True:
            # 寻找帧头
            head_idx = self.raw_buffer.find(ToFConfig.FRAME_HEAD)
            if head_idx < 0:
                # 防止内存溢出
                if len(self.raw_buffer) > 50000:
                    self.raw_buffer = b''
                return
            
            # 截断到帧头
            self.raw_buffer = self.raw_buffer[head_idx:]
            
            # 检查长度字段
            if len(self.raw_buffer) < 4:
                return
            
            # 解析数据长度
            data_len = unpack("H", self.raw_buffer[2:4])[0]
            frame_len = 2 + 2 + data_len + 2  # 帧头 + 长度 + 数据 + 校验 + 帧尾
            
            if len(self.raw_buffer) < frame_len:
                return
            
            # 提取帧
            frame = self.raw_buffer[:frame_len]
            self.raw_buffer = self.raw_buffer[frame_len:]
            
            # 校验
            frame_tail = frame[-1:]
            checksum = frame[-2]
            calculated = sum(frame[:-2]) % 256
            
            if frame_tail != ToFConfig.FRAME_TAIL or checksum != calculated:
                continue
            
            # 提取深度数据 (从第20字节开始)
            depth_data = frame[20:-2]
            
            if len(depth_data) == ToFConfig.DATA_LENGTH:
                depth_array = np.array([unpack("B", bytes([v]))[0] for v in depth_data], dtype=np.uint8)
                
                # 放入队列
                try:
                    self.frame_queue.put_nowait(depth_array)
                    self.frame_count += 1
                except queue.Full:
                    # 丢弃旧帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(depth_array)
                    except:
                        pass
    
    def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧深度数据"""
        if not self._connected:
            return None
        
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def is_connected(self) -> bool:
        return self._connected and self.serial.is_open
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'connected': self._connected,
            'port': self.port,
            'frame_count': self.frame_count,
            'queue_size': self.frame_queue.qsize()
        }


# ======================= 跌倒报警 =======================

def default_fall_alert(event: FallEvent):
    """默认跌倒报警 (V4: 支持三分类显示)"""
    # 获取跌倒类型 (如果有)
    fall_type = getattr(event, 'fall_type', None) or 'Fall'
    
    print("\n" + "=" * 50)
    if fall_type == 'Backward':
        print("🚨🚨🚨 警报！检测到向后跌倒！🚨🚨🚨")
    else:
        print("🚨🚨🚨 警报！检测到跌倒！🚨🚨🚨")
    print("=" * 50)
    print(f"   时间: {event.timestamp}")
    print(f"   类型: {fall_type}")
    print(f"   置信度: {event.confidence:.1%}")
    print(f"   事件ID: {event.event_id}")
    print("=" * 50 + "\n")
    
    # 可以在这里添加更多报警方式:
    # - 发送邮件
    # - 推送通知
    # - 触发警报器
    # - 拨打紧急电话


def default_fall_alert_simple(confidence: float, fall_type: str = "Fall"):
    """简化版立即报警 (用于禁用监控时)"""
    print("\n" + "=" * 50)
    if fall_type == 'Backward':
        print("🚨🚨🚨 警报！检测到向后跌倒！🚨🚨🚨")
    else:
        print("🚨🚨🚨 警报！检测到跌倒！🚨🚨🚨")
    print("=" * 50)
    print(f"   时间: {datetime.now()}")
    print(f"   类型: {fall_type}")
    print(f"   置信度: {confidence:.1%}")
    print("=" * 50 + "\n")


# ======================= 主程序 =======================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ToF 相机实时跌倒检测')
    parser.add_argument('--port', type=str, default='COM3', help='串口号 (如 COM3)')
    parser.add_argument('--list-ports', action='store_true', help='列出可用串口')
    parser.add_argument('--preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--duration', type=int, default=0, help='运行时长(秒)，0=无限')
    parser.add_argument('--window', type=int, default=50, help='检测窗口大小(帧数) V5: 50帧@10FPS=5秒')
    parser.add_argument('--threshold', type=float, default=0.60, help='跌倒置信度阈值')
    parser.add_argument('--consecutive', type=int, default=3, help='连续检测次数要求（阻止误报）')
    parser.add_argument('--margin', type=float, default=0.15, help='概率margin阈值')
    parser.add_argument('--min-prob', type=float, default=0.45, help='Fall/Backward最低概率要求')
    parser.add_argument('--output', type=str, default='fall_events', help='输出目录')
    parser.add_argument('--no-filter', action='store_true', help='禁用关键点过滤')
    parser.add_argument('--no-save', action='store_true', help='不保存跌倒事件')
    parser.add_argument('--record', action='store_true', help='录制所有深度图到文件夹')
    parser.add_argument('--record-dir', type=str, default=None, help='录制目录（默认自动生成带日期的目录）')
    parser.add_argument('--save-vis', action='store_true', help='保存深度图和带姿态点的可视化图')
    parser.add_argument('--save-vis-dir', type=str, default=None, help='可视化保存目录（默认自动生成）')
    parser.add_argument('--alert-delay', type=float, default=10.0, help='报警延迟时间（秒），超过此时间无恢复才触发真正报警')
    parser.add_argument('--no-monitor', action='store_true', help='禁用延迟监控，检测到跌倒立即报警')
    
    args = parser.parse_args()
    
    # 列出串口
    if args.list_ports:
        camera = RealToFCamera()
        camera.list_ports()
        return
    
    print("=" * 60)
    print("ToF 相机实时跌倒检测系统")
    print("=" * 60)
    
    # 录制设置
    record_dir = None
    record_frame_count = 0
    if args.record:
        if args.record_dir:
            record_dir = Path(args.record_dir)
    
    # 可视化保存设置
    vis_dir_raw = None
    vis_dir_skeleton = None
    vis_frame_count = 0
    if args.save_vis:
        if args.save_vis_dir:
            vis_base_dir = Path(args.save_vis_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_base_dir = Path(__file__).parent / f"vis_output_{timestamp}"
        vis_dir_raw = vis_base_dir / "depth_raw"
        vis_dir_skeleton = vis_base_dir / "depth_skeleton"
        vis_dir_raw.mkdir(parents=True, exist_ok=True)
        vis_dir_skeleton.mkdir(parents=True, exist_ok=True)
        print(f"\n📷 可视化保存已启用")
        print(f"   原始深度图: {vis_dir_raw}")
        print(f"   带姿态点: {vis_dir_skeleton}")
    
    if args.record:
        if args.record_dir:
            record_dir = Path(args.record_dir)
        else:
            # 自动生成带日期的目录
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            record_dir = Path(__file__).parent / f"recorded_depth_2026-1-20_{timestamp}"
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📹 录制模式已启用")
        print(f"   保存目录: {record_dir}")
    
    # 创建相机
    camera = RealToFCamera(port=args.port)
    
    # 连接相机
    print(f"\n正在连接 ToF 相机 ({args.port})...")
    if not camera.connect():
        print("\n相机连接失败！请检查:")
        print("  1. ToF 相机是否已连接")
        print("  2. 串口号是否正确")
        print("  3. 是否有其他程序占用串口")
        camera.list_ports()
        return
    
    # 等待相机稳定
    print("等待相机稳定 (3秒)...")
    time.sleep(3)
    
    # 配置检测器
    config = RealtimeConfig(
        window_size=args.window,
        fall_confidence_threshold=args.threshold,
        fall_consecutive_required=args.consecutive,
        fall_margin_threshold=args.margin,
        fall_min_probability=args.min_prob,
        use_keypoint_filter=not args.no_filter,
        show_preview=args.preview,
        output_dir=args.output,
        save_fall_events=not args.no_save,
        target_fps=10  # ToF 相机通常 10 FPS
    )
    
    # 创建检测器 (不传回调，由 FallMonitor 管理报警)
    detector = RealtimeFallDetector(config, on_fall_detected=None)
    
    # 创建跌倒监控器
    fall_monitor = None
    if not args.no_monitor:
        fall_monitor = FallMonitor(fps=10.0)
        fall_monitor.ALERT_DELAY = args.alert_delay
    
    print("\n" + "=" * 60)
    print("开始实时跌倒检测")
    print(f"  - 检测窗口: {args.window} 帧")
    print(f"  - 置信度阈值: {args.threshold:.0%}")
    print(f"  - 连续检测要求: {args.consecutive} 次")
    print(f"  - 概率margin: {args.margin}")
    print(f"  - 最低跌倒概率: {args.min_prob}")
    print(f"  - 关键点过滤: {'启用' if not args.no_filter else '禁用'}")
    print(f"  - 延迟监控: {'禁用' if args.no_monitor else f'启用 ({args.alert_delay}秒)'}")
    print(f"  - 保存事件: {'是' if not args.no_save else '否'}")
    print(f"  - 录制深度图: {'是' if args.record else '否'}")
    print(f"  - 运行时长: {'无限' if args.duration == 0 else f'{args.duration}秒'}")
    print("=" * 60)
    print("\n按 Ctrl+C 停止检测\n")
    
    start_time = time.time()
    
    # 帧去重
    last_frame = None
    duplicate_count = 0
    
    try:
        while True:
            # 检查时长限制
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                print(f"\n已达到 {args.duration} 秒运行时长，自动停止")
                break
            
            # 获取帧
            depth_data = camera.get_frame()
            if depth_data is None:
                if not camera.is_connected():
                    print("\n相机断开连接！")
                    break
                continue
            
            # 帧去重：跳过与上一帧完全相同的帧
            if last_frame is not None and np.array_equal(depth_data, last_frame):
                duplicate_count += 1
                continue
            last_frame = depth_data.copy()
            
            # 录制深度图
            if args.record and record_dir is not None:
                if len(depth_data) == 10000:
                    depth_2d = depth_data.reshape(100, 100)
                    frame_path = record_dir / f"depth_{record_frame_count:06d}.png"
                    cv2.imwrite(str(frame_path), depth_2d)
                    record_frame_count += 1
            
            # 处理帧
            result = detector.process_frame(depth_data)
            
            # 跌倒监控逻辑
            monitor_state = MonitorState.IDLE
            if fall_monitor is not None:
                # 获取RF分类结果
                is_fall = result.get('fall_detected', False)
                confidence = result.get('fall_confidence', 0.0)
                
                # 从 detector 获取跌倒类型
                fall_type = "Normal"
                if is_fall:
                    # 获取分类器的预测标签
                    fall_type = getattr(detector, '_last_fall_type', 'Fall')
                
                # 获取当前关键点
                current_kp = detector.current_keypoints
                
                # 更新监控器 (传入关键点用于恢复检测)
                monitor_state, alert_msg = fall_monitor.update(is_fall, confidence, fall_type, current_kp)
                
                # 如果触发了报警，可以在这里添加额外操作
                # (FallMonitor._trigger_alert 已经打印报警信息)
            else:
                # 没有监控器，立即报警
                if result.get('fall_detected', False):
                    default_fall_alert_simple(
                        confidence=result.get('fall_confidence', 0.0),
                        fall_type='Fall'
                    )
            
            # 显示预览
            if args.preview:
                # 统一用 raw_to_depth_image_fast 处理预览图
                if len(depth_data) == 10000:
                    preview_pil = raw_to_depth_image_fast(depth_data)
                    preview = np.array(preview_pil)
                    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                elif len(detector.frame_buffer) > 0:
                    preview = detector.frame_buffer[-1].copy()
                else:
                    continue
                
                # 保存原始深度图（骨架绘制之前）
                if args.save_vis and vis_dir_raw is not None:
                    raw_filename = vis_dir_raw / f"frame_{vis_frame_count:06d}.png"
                    cv2.imwrite(str(raw_filename), preview)
                
                # ========== 窗口1: ToF 深度图 + 骨架 (纯净画面) ==========
                skeleton_color = (0, 255, 0)  # 默认绿色
                
                # 绘制骨架（如果检测到人体）
                if result['detected'] and not result['filtered']:
                    kp = detector.current_keypoints
                    if kp is not None and len(kp) >= 15:
                        h, w = preview.shape[:2]
                        # 提取关键点坐标
                        points = []
                        for i in range(5):
                            x = int(kp[i*3] * w / 100)
                            y = int(kp[i*3 + 1] * h / 100)
                            conf = kp[i*3 + 2]
                            points.append((x, y, conf))
                        
                        # 骨架连接: nose-rshoulder, nose-lshoulder, rshoulder-lshoulder, rshoulder-rhip, lshoulder-lhip, rhip-lhip
                        connections = [(0,1), (0,2), (1,2), (1,3), (2,4), (3,4)]
                        
                        # 根据状态选择颜色
                        if monitor_state == MonitorState.ALERTED:
                            skeleton_color = (0, 0, 255)  # 红色 - 已报警
                        elif monitor_state == MonitorState.MONITORING:
                            skeleton_color = (0, 165, 255)  # 橙色 - 监控中
                        elif result['fall_detected']:
                            skeleton_color = (0, 255, 255)  # 黄色 - 疑似跌倒
                        else:
                            skeleton_color = (0, 255, 0)  # 绿色 - 正常
                        
                        # 绘制连接线
                        for i, j in connections:
                            if points[i][2] > 0.3 and points[j][2] > 0.3:
                                cv2.line(preview, (points[i][0], points[i][1]), 
                                        (points[j][0], points[j][1]), skeleton_color, 2)
                        
                        # 绘制关键点
                        for i, (x, y, conf) in enumerate(points):
                            if conf > 0.3:
                                cv2.circle(preview, (x, y), 3, (0, 255, 255), -1)
                
                # 保存带骨架的深度图
                if args.save_vis and vis_dir_skeleton is not None:
                    skel_filename = vis_dir_skeleton / f"frame_{vis_frame_count:06d}.png"
                    cv2.imwrite(str(skel_filename), preview)
                    vis_frame_count += 1
                
                # 放大显示 ToF 画面
                preview_large = cv2.resize(preview, (400, 400), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('ToF View', preview_large)
                
                # ========== 窗口2: 信息面板 ==========
                info_panel = np.zeros((400, 300, 3), dtype=np.uint8)
                info_panel[:] = (40, 40, 40)  # 深灰色背景
                
                # 计算运行时间和FPS
                runtime = time.time() - start_time
                current_fps = detector.frame_count / runtime if runtime > 0 else 0
                
                # 状态颜色
                if monitor_state == MonitorState.ALERTED:
                    status_text = "ALERT!"
                    status_color = (0, 0, 255)
                elif monitor_state == MonitorState.MONITORING:
                    status_text = "MONITORING"
                    status_color = (0, 165, 255)
                else:
                    status_text = "NORMAL"
                    status_color = (0, 255, 0)
                
                # 绘制标题
                cv2.putText(info_panel, "Fall Detection System", (15, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.line(info_panel, (15, 45), (285, 45), (100, 100, 100), 1)
                
                # 基本信息
                y_offset = 80
                line_height = 28
                
                # 计算 Buffer FPS (关键点缓冲区实际填充速度)
                buffer_fps = detector.buffer_frame_count / runtime if runtime > 0 else 0
                
                info_items = [
                    ("Frame:", f"{detector.frame_count}"),
                    ("UI FPS:", f"{current_fps:.1f}"),
                    ("Buffer FPS:", f"{buffer_fps:.1f} (target: {config.target_fps})"),
                    ("Buffer:", f"{len(detector.keypoints_buffer)}/{config.window_size}"),
                    ("RF Falls:", f"{detector.fall_count}"),
                    ("Consec:", f"{detector.consecutive_fall_count}/{config.fall_consecutive_required}"),
                ]
                
                for label, value in info_items:
                    cv2.putText(info_panel, label, (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                    cv2.putText(info_panel, value, (115, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    y_offset += line_height
                
                # 分隔线
                cv2.line(info_panel, (15, y_offset), (285, y_offset), (100, 100, 100), 1)
                y_offset += 20
                
                # 监控状态
                cv2.putText(info_panel, "Monitor Status:", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y_offset += line_height
                
                cv2.putText(info_panel, status_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                y_offset += line_height + 10
                
                # 监控倒计时
                if fall_monitor and fall_monitor.current_event:
                    elapsed = time.time() - fall_monitor.current_event.fall_time
                    remaining = max(0, fall_monitor.ALERT_DELAY - elapsed)
                    
                    cv2.putText(info_panel, f"Elapsed: {elapsed:.1f}s", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += line_height
                    
                    if monitor_state == MonitorState.MONITORING:
                        cv2.putText(info_panel, f"Alert in: {remaining:.1f}s", (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                        y_offset += line_height
                        
                        # 进度条
                        progress = min(1.0, elapsed / fall_monitor.ALERT_DELAY)
                        bar_width = 260
                        bar_height = 15
                        cv2.rectangle(info_panel, (20, y_offset), (20 + bar_width, y_offset + bar_height), 
                                     (80, 80, 80), -1)
                        cv2.rectangle(info_panel, (20, y_offset), (20 + int(bar_width * progress), y_offset + bar_height), 
                                     (0, 165, 255), -1)
                        y_offset += bar_height + 15
                
                # 监控统计
                if fall_monitor:
                    stats = fall_monitor.get_status()
                    cv2.line(info_panel, (15, y_offset), (285, y_offset), (100, 100, 100), 1)
                    y_offset += 20
                    
                    cv2.putText(info_panel, "Statistics:", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    y_offset += line_height
                    
                    stat_items = [
                        ("Falls:", str(stats['total_falls'])),
                        ("Recoveries:", str(stats['total_recoveries'])),
                        ("Alerts:", str(stats['total_alerts'])),
                    ]
                    
                    for label, value in stat_items:
                        cv2.putText(info_panel, label, (30, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
                        cv2.putText(info_panel, value, (130, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        y_offset += 22
                
                cv2.imshow('Info Panel', info_panel)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    print("\n用户按下 ESC，停止检测")
                    break
            
            # 定期状态报告
            if detector.frame_count % 100 == 0 and detector.frame_count > 0:
                runtime = time.time() - start_time
                fps = detector.frame_count / runtime
                print(f"[状态] 帧: {detector.frame_count} | FPS: {fps:.1f} | 跌倒: {detector.fall_count}")
    
    except KeyboardInterrupt:
        print("\n\n收到中断信号，停止检测...")
    
    finally:
        # 清理
        if args.preview:
            cv2.destroyAllWindows()
        
        camera.disconnect()
        
        # 显示最终统计
        runtime = time.time() - start_time
        total_frames = detector.frame_count + duplicate_count
        print("\n" + "=" * 60)
        print("检测结束")
        print(f"  - 运行时间: {runtime/60:.1f} 分钟")
        print(f"  - 相机帧数: {total_frames} (去重: {duplicate_count}, 处理: {detector.frame_count})")
        print(f"  - RF检测跌倒: {detector.fall_count} 次")
        if fall_monitor:
            status = fall_monitor.get_status()
            print(f"  - 监控记录: {status['total_falls']} 次跌倒")
            print(f"  - 恢复次数: {status['total_recoveries']} 次")
            print(f"  - 真实报警: {status['total_alerts']} 次")
        if args.record:
            print(f"  - 录制帧数: {record_frame_count}")
            print(f"  - 保存目录: {record_dir}")
        if runtime > 0:
            print(f"  - 有效 FPS: {detector.frame_count/runtime:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
