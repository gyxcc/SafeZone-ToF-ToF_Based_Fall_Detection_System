# -*- coding: utf-8 -*-
"""
实时可视化模块
绘制深度图、骨架、状态信息
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from datetime import datetime


# 骨架连接定义 (关键点索引对)
# 0: nose, 1: right_shoulder, 2: left_shoulder, 3: right_hip, 4: left_hip
SKELETON_CONNECTIONS = [
    (0, 1),  # nose -> right_shoulder
    (0, 2),  # nose -> left_shoulder
    (1, 2),  # right_shoulder -> left_shoulder
    (1, 3),  # right_shoulder -> right_hip
    (2, 4),  # left_shoulder -> left_hip
    (3, 4),  # right_hip -> left_hip
]

# 关键点颜色
KEYPOINT_COLORS = {
    0: (0, 255, 255),    # nose - 黄色
    1: (0, 255, 0),      # right_shoulder - 绿色
    2: (0, 255, 0),      # left_shoulder - 绿色
    3: (255, 0, 0),      # right_hip - 蓝色
    4: (255, 0, 0),      # left_hip - 蓝色
}

# 骨架颜色
SKELETON_COLOR = (255, 255, 255)  # 白色


class RealtimeVisualizer:
    """实时可视化器"""
    
    def __init__(self, 
                 display_size: Tuple[int, int] = (800, 600),
                 show_skeleton: bool = True,
                 show_stats: bool = True,
                 show_history: bool = True,
                 history_length: int = 100):
        """
        初始化可视化器
        
        Args:
            display_size: 显示窗口大小 (宽, 高)
            show_skeleton: 是否显示骨架
            show_stats: 是否显示统计信息
            show_history: 是否显示历史曲线
            history_length: 历史数据长度
        """
        self.display_size = display_size
        self.show_skeleton = show_skeleton
        self.show_stats = show_stats
        self.show_history = show_history
        
        # 历史数据（用于绘制曲线）
        self.confidence_history = deque(maxlen=history_length)
        self.height_history = deque(maxlen=history_length)
        self.fall_events = []  # 跌倒事件时间戳
        
        # 状态
        self.start_time = datetime.now()
        self.frame_count = 0
        self.fall_count = 0
        self.last_fall_confidence = 0
        self.last_status = "正常"
        self.status_color = (0, 255, 0)  # 绿色
        
        # 窗口
        self.window_name = "Fall Detection - Real-time Monitor"
        
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                      conf_threshold: float = 0.3,
                      is_fall: bool = False) -> np.ndarray:
        """
        在图像上绘制骨架
        
        Args:
            image: BGR图像
            keypoints: 关键点数组 [x0,y0,c0, x1,y1,c1, ...]
            conf_threshold: 置信度阈值
            is_fall: 是否为跌倒状态
        
        Returns:
            绘制后的图像
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # 跌倒状态使用红色
        skeleton_color = (0, 0, 255) if is_fall else SKELETON_COLOR
        
        # 解析关键点
        points = []
        for i in range(5):
            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            conf = keypoints[i * 3 + 2]
            
            if conf > conf_threshold:
                # 缩放到图像尺寸
                px = int(x * w / 100)
                py = int(y * h / 100)
                points.append((px, py, conf))
            else:
                points.append(None)
        
        # 绘制骨架连接
        for i, j in SKELETON_CONNECTIONS:
            if points[i] is not None and points[j] is not None:
                pt1 = (points[i][0], points[i][1])
                pt2 = (points[j][0], points[j][1])
                cv2.line(img, pt1, pt2, skeleton_color, 2)
        
        # 绘制关键点
        for i, pt in enumerate(points):
            if pt is not None:
                color = (0, 0, 255) if is_fall else KEYPOINT_COLORS.get(i, (255, 255, 255))
                cv2.circle(img, (pt[0], pt[1]), 4, color, -1)
                cv2.circle(img, (pt[0], pt[1]), 6, color, 1)
        
        return img
    
    def draw_status_overlay(self, image: np.ndarray, 
                            status_text: str = "Monitoring...",
                            fps: float = 0.0,
                            buffer_size: int = 0,
                            frame_count: int = 0,
                            fall_count: int = 0) -> np.ndarray:
        """
        绘制状态覆盖层
        
        Args:
            image: BGR图像
            status_text: 状态文本
            fps: 当前帧率
            buffer_size: 缓冲区大小
            frame_count: 总帧数
            fall_count: 跌倒次数
        
        Returns:
            绘制后的图像
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # 绘制半透明状态栏背景
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # 状态文本
        is_fall = "FALL" in status_text.upper()
        status_color = (0, 0, 255) if is_fall else (0, 255, 0)
        cv2.putText(img, status_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # 统计信息
        stats = f"Frame:{frame_count} | Buf:{buffer_size} | FPS:{fps:.1f} | Falls:{fall_count}"
        cv2.putText(img, stats, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 时间戳
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(img, time_str, (w - text_size[0] - 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return img
    
    def draw_fall_alert(self, image: np.ndarray, 
                        confidence: float = 0.0,
                        duration: float = 0.0) -> np.ndarray:
        """
        绘制跌倒警报
        
        Args:
            image: BGR图像
            confidence: 跌倒置信度
            duration: 距上次跌倒的时间
        
        Returns:
            绘制后的图像
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        if confidence > 0.65:
            # 红色边框警报
            border_thickness = 8
            cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 0, 255), border_thickness)
            
            # 警报文本背景
            alert_h = 60
            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - alert_h), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # 警报文本
            cv2.putText(img, "!! FALL ALERT !!", (w//2 - 80, h - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"Confidence: {confidence:.1%}", (w//2 - 70, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def show_frame(self, image: np.ndarray, window_name: str = "Fall Detection") -> bool:
        """
        显示帧
        
        Args:
            image: 要显示的图像
            window_name: 窗口名称
        
        Returns:
            bool: True 继续运行, False 退出 (按 ESC)
        """
        # 放大显示（如果图像太小）
        h, w = image.shape[:2]
        if w < 400 or h < 400:
            scale = max(400 / w, 400 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        
        return key != 27  # ESC 退出
    
    def draw_stats_panel(self, width: int = 300, height: int = 600) -> np.ndarray:
        """绘制统计信息面板"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # 深灰色背景
        
        y = 30
        line_height = 30
        
        # 标题
        cv2.putText(panel, "Fall Detection Monitor", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += line_height + 10
        
        # 分隔线
        cv2.line(panel, (10, y), (width - 10, y), (100, 100, 100), 1)
        y += 20
        
        # 状态
        cv2.putText(panel, "Status:", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(panel, self.last_status, (100, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.status_color, 2)
        y += line_height
        
        # 运行时间
        runtime = datetime.now() - self.start_time
        runtime_str = str(runtime).split('.')[0]
        cv2.putText(panel, f"Runtime: {runtime_str}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height
        
        # 帧数
        cv2.putText(panel, f"Frames: {self.frame_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height
        
        # FPS
        elapsed = runtime.total_seconds()
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height
        
        # 分隔线
        cv2.line(panel, (10, y), (width - 10, y), (100, 100, 100), 1)
        y += 20
        
        # 跌倒统计
        cv2.putText(panel, f"Fall Events: {self.fall_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        y += line_height
        
        cv2.putText(panel, f"Last Confidence: {self.last_fall_confidence:.1%}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height + 10
        
        # 分隔线
        cv2.line(panel, (10, y), (width - 10, y), (100, 100, 100), 1)
        y += 20
        
        # 历史曲线标题
        cv2.putText(panel, "Body Height History", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 10
        
        # 绘制历史曲线
        if len(self.height_history) > 1:
            chart_height = 100
            chart_width = width - 40
            chart_x = 20
            chart_y = y
            
            # 绘制背景
            cv2.rectangle(panel, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height),
                         (60, 60, 60), -1)
            
            # 绘制曲线
            heights = list(self.height_history)
            if len(heights) > 1:
                min_h = min(heights) - 5
                max_h = max(heights) + 5
                if max_h == min_h:
                    max_h = min_h + 10
                
                points = []
                for i, h in enumerate(heights):
                    px = chart_x + int(i * chart_width / len(heights))
                    py = chart_y + chart_height - int((h - min_h) * chart_height / (max_h - min_h))
                    points.append((px, py))
                
                for i in range(len(points) - 1):
                    cv2.line(panel, points[i], points[i+1], (0, 255, 255), 1)
            
            y = chart_y + chart_height + 20
        
        # 置信度曲线标题
        cv2.putText(panel, "Detection Confidence", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 10
        
        # 绘制置信度曲线
        if len(self.confidence_history) > 1:
            chart_height = 80
            chart_width = width - 40
            chart_x = 20
            chart_y = y
            
            cv2.rectangle(panel, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height),
                         (60, 60, 60), -1)
            
            # 绘制阈值线
            threshold_y = chart_y + chart_height - int(0.65 * chart_height)
            cv2.line(panel, (chart_x, threshold_y), 
                    (chart_x + chart_width, threshold_y), (0, 0, 150), 1)
            
            confs = list(self.confidence_history)
            points = []
            for i, c in enumerate(confs):
                px = chart_x + int(i * chart_width / len(confs))
                py = chart_y + chart_height - int(c * chart_height)
                points.append((px, py))
            
            for i in range(len(points) - 1):
                color = (0, 255, 0) if confs[i] < 0.65 else (0, 0, 255)
                cv2.line(panel, points[i], points[i+1], color, 1)
        
        return panel
    
    def update(self, depth_image: np.ndarray, keypoints: np.ndarray,
               fall_detected: bool = False, fall_confidence: float = 0.0,
               is_person_present: bool = True) -> np.ndarray:
        """
        更新可视化
        
        Args:
            depth_image: 深度图像 (BGR)
            keypoints: 关键点数组
            fall_detected: 是否检测到跌倒
            fall_confidence: 跌倒置信度
            is_person_present: 是否检测到人
        
        Returns:
            合成的显示图像
        """
        self.frame_count += 1
        
        # 更新历史
        if is_person_present and keypoints is not None:
            # 计算身体平均高度
            valid_y = []
            for i in range(5):
                if keypoints[i * 3 + 2] > 0.3:
                    valid_y.append(keypoints[i * 3 + 1])
            if valid_y:
                self.height_history.append(np.mean(valid_y))
        
        self.confidence_history.append(fall_confidence)
        self.last_fall_confidence = fall_confidence
        
        # 更新状态
        if fall_detected:
            self.fall_count += 1
            self.last_status = "FALL DETECTED!"
            self.status_color = (0, 0, 255)  # 红色
            self.fall_events.append(datetime.now())
        elif not is_person_present:
            self.last_status = "No Person"
            self.status_color = (128, 128, 128)  # 灰色
        else:
            self.last_status = "Normal"
            self.status_color = (0, 255, 0)  # 绿色
        
        # 准备深度图显示
        if len(depth_image.shape) == 2:
            depth_display = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
        else:
            depth_display = depth_image.copy()
        
        # 放大深度图
        main_size = (self.display_size[0] - 300, self.display_size[1])
        depth_large = cv2.resize(depth_display, main_size, interpolation=cv2.INTER_NEAREST)
        
        # 绘制骨架
        if self.show_skeleton and keypoints is not None:
            depth_large = self.draw_skeleton(depth_large, keypoints)
        
        # 添加状态文字到深度图
        if fall_detected:
            # 跌倒警告
            cv2.rectangle(depth_large, (0, 0), (main_size[0], 60), (0, 0, 200), -1)
            cv2.putText(depth_large, "!! FALL DETECTED !!", (main_size[0]//2 - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # 创建统计面板
        stats_panel = self.draw_stats_panel(300, self.display_size[1])
        
        # 合成最终图像
        display = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        display[:, :main_size[0]] = depth_large
        display[:, main_size[0]:] = stats_panel
        
        return display
    
    def show(self, display: np.ndarray) -> int:
        """
        显示图像
        
        Returns:
            按键值 (27=ESC, -1=无按键)
        """
        cv2.imshow(self.window_name, display)
        return cv2.waitKey(1)
    
    def close(self):
        """关闭窗口"""
        cv2.destroyAllWindows()
    
    def reset(self):
        """重置状态"""
        self.start_time = datetime.now()
        self.frame_count = 0
        self.fall_count = 0
        self.confidence_history.clear()
        self.height_history.clear()
        self.fall_events.clear()


def demo_visualizer():
    """演示可视化器"""
    import numpy as np
    
    viz = RealtimeVisualizer(display_size=(800, 500))
    
    print("可视化演示 - 按 ESC 退出")
    
    frame_idx = 0
    while True:
        # 模拟深度图
        depth = np.random.randint(100, 200, (100, 100), dtype=np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        
        # 模拟关键点 (正弦运动)
        t = frame_idx * 0.1
        keypoints = np.array([
            50 + 10*np.sin(t), 30, 0.9,  # nose
            40, 45 + 5*np.sin(t), 0.85,  # right_shoulder
            60, 45 + 5*np.sin(t), 0.85,  # left_shoulder
            42, 70, 0.8,  # right_hip
            58, 70, 0.8,  # left_hip
        ])
        
        # 模拟跌倒检测
        fall_detected = frame_idx == 100
        fall_conf = np.random.uniform(0.3, 0.5) if not fall_detected else 0.85
        
        display = viz.update(depth_color, keypoints, fall_detected, fall_conf)
        key = viz.show(display)
        
        if key == 27:
            break
        
        frame_idx += 1
        
    viz.close()


if __name__ == "__main__":
    demo_visualizer()
