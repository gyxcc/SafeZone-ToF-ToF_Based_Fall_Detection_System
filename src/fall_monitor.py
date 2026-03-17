"""
fall_monitor.py
跌倒监测器 - 核心检测与报警逻辑分离

逻辑：
1. RF 检测到跌倒 → 仅记录，进入监测状态
2. 监测后续变化（人是否重新出现）
3. 如果持续无变化超过阈值 → 触发报警
4. 如果人重新出现 → 取消警报，记录为"已恢复"
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import time
from datetime import datetime


# ==================== 状态定义 ====================

class MonitorState(Enum):
    """监测状态"""
    IDLE = "idle"                    # 空闲，无人或正常
    MONITORING = "monitoring"        # 检测到跌倒，正在监测
    ALERT_PENDING = "alert_pending"  # 即将报警（等待最后确认）
    ALERTED = "alerted"              # 已报警


@dataclass
class FallEvent:
    """跌倒事件记录"""
    event_id: str
    fall_time: float                          # 检测到跌倒的时间
    rf_confidence: float                      # RF 置信度
    last_keypoint_y: float = -1               # 消失前最后的 Y 坐标
    disappear_time: Optional[float] = None    # 人消失的时间
    recovery_time: Optional[float] = None     # 恢复的时间
    alert_triggered: bool = False             # 是否触发了报警
    alert_time: Optional[float] = None        # 报警时间
    status: str = "monitoring"                # monitoring / recovered / alerted


@dataclass
class KeypointFrame:
    """单帧关键点数据"""
    timestamp: float
    keypoints: np.ndarray  # shape: (5, 3) - [x, y, conf]
    
    @property
    def valid_count(self) -> int:
        return int(np.sum(self.keypoints[:, 2] > 0.3))
    
    @property
    def is_valid(self) -> bool:
        return self.valid_count >= 3
    
    @property
    def center_y(self) -> float:
        valid_mask = self.keypoints[:, 2] > 0.3
        if not np.any(valid_mask):
            return -1
        return np.mean(self.keypoints[valid_mask, 1])
    
    @property
    def avg_confidence(self) -> float:
        return np.mean(self.keypoints[:, 2])


# ==================== 核心监测器 ====================

class FallMonitor:
    """
    跌倒监测器
    
    职责：
    1. 接收 RF 的跌倒检测结果
    2. 记录跌倒事件
    3. 监测后续变化
    4. 决定是否需要报警
    """
    
    def __init__(self, fps: float = 10.0):
        self.fps = fps
        
        # 当前状态
        self.state = MonitorState.IDLE
        self.current_event: Optional[FallEvent] = None
        
        # 关键点历史
        self.history: deque[KeypointFrame] = deque(maxlen=int(fps * 30))  # 30秒历史
        
        # 事件历史
        self.event_history: List[FallEvent] = []
        
        # 时间阈值（秒）
        self.ALERT_DELAY = 10.0           # 跌倒后多久无恢复则报警
        self.QUICK_RECOVERY_TIME = 3.0    # 快速恢复时间（可能是误报）
        self.REAPPEAR_CONFIRM_FRAMES = 5  # 重新出现确认帧数
        self.DISAPPEAR_CONFIRM_FRAMES = 5 # 消失确认帧数
        
        # 统计
        self.total_falls = 0
        self.total_recoveries = 0
        self.total_alerts = 0
        
    def add_keypoints(self, timestamp: float, keypoints: np.ndarray):
        """添加新的关键点帧"""
        frame = KeypointFrame(timestamp=timestamp, keypoints=keypoints.copy())
        self.history.append(frame)
    
    def on_rf_fall_detected(self, timestamp: float, confidence: float) -> Tuple[bool, str]:
        """
        RF 检测到跌倒时调用
        
        Returns:
            (是否记录为新事件, 消息)
        """
        # 如果已经在监测中，不重复记录
        if self.state == MonitorState.MONITORING:
            return False, "Already monitoring a fall event"
        
        # 如果已经报警，不重复处理
        if self.state == MonitorState.ALERTED:
            return False, "Already alerted"
        
        # 创建新的跌倒事件
        self.total_falls += 1
        event_id = f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_falls}"
        
        # 获取消失前的最后 Y 坐标
        last_y = -1
        if len(self.history) > 0:
            for frame in reversed(list(self.history)[-10:]):
                if frame.is_valid:
                    last_y = frame.center_y
                    break
        
        self.current_event = FallEvent(
            event_id=event_id,
            fall_time=timestamp,
            rf_confidence=confidence,
            last_keypoint_y=last_y
        )
        
        self.state = MonitorState.MONITORING
        
        print(f"\n[FallMonitor] 📋 记录跌倒事件: {event_id}")
        print(f"             RF置信度: {confidence:.2%}")
        print(f"             进入监测状态，等待后续变化...")
        
        return True, f"Fall event recorded: {event_id}"
    
    def on_rf_normal_detected(self, timestamp: float) -> Tuple[bool, str]:
        """
        RF 检测到正常时调用
        
        如果在监测状态，检查是否真的恢复
        """
        if self.state != MonitorState.MONITORING:
            return False, "Not in monitoring state"
        
        # 检查人是否重新出现并稳定
        recovered, msg = self._check_person_recovered()
        
        if recovered:
            return self._handle_recovery(timestamp)
        
        return False, msg
    
    def update(self, timestamp: float) -> Tuple[MonitorState, Optional[str]]:
        """
        每帧调用，更新监测状态
        
        Returns:
            (当前状态, 如果需要报警则返回报警消息)
        """
        if self.state == MonitorState.IDLE:
            return self.state, None
        
        if self.state == MonitorState.ALERTED:
            # 已经报警，检查是否恢复
            recovered, _ = self._check_person_recovered()
            if recovered:
                self._handle_recovery(timestamp)
                return self.state, "Person recovered after alert"
            return self.state, None
        
        if self.state == MonitorState.MONITORING:
            # 检查是否需要触发报警
            if self.current_event:
                elapsed = timestamp - self.current_event.fall_time
                
                # 检查人是否重新出现
                recovered, msg = self._check_person_recovered()
                if recovered:
                    # 检查是否是快速恢复（可能误报）
                    if elapsed < self.QUICK_RECOVERY_TIME:
                        print(f"\n[FallMonitor] ⚡ 快速恢复 ({elapsed:.1f}s) - 可能是误报")
                    self._handle_recovery(timestamp)
                    return self.state, None
                
                # 检查是否超时需要报警
                if elapsed >= self.ALERT_DELAY:
                    alert_msg = self._trigger_alert(timestamp)
                    return self.state, alert_msg
                
                # 还在监测中
                remaining = self.ALERT_DELAY - elapsed
                if int(elapsed) % 2 == 0 and elapsed > 0:  # 每2秒提示一次
                    print(f"\r[FallMonitor] 监测中... {elapsed:.0f}s / 报警倒计时: {remaining:.0f}s", end="")
        
        return self.state, None
    
    def _check_person_recovered(self) -> Tuple[bool, str]:
        """检查人是否重新出现并稳定"""
        if len(self.history) < self.REAPPEAR_CONFIRM_FRAMES:
            return False, "Insufficient data"
        
        recent = list(self.history)[-self.REAPPEAR_CONFIRM_FRAMES:]
        visible_count = sum(1 for f in recent if f.is_valid)
        
        # 需要大部分帧都能看到人
        if visible_count >= self.REAPPEAR_CONFIRM_FRAMES * 0.8:
            return True, "Person reappeared and stable"
        
        return False, f"Person not stable ({visible_count}/{self.REAPPEAR_CONFIRM_FRAMES} frames visible)"
    
    def _check_person_disappeared(self) -> bool:
        """检查人是否消失"""
        if len(self.history) < self.DISAPPEAR_CONFIRM_FRAMES:
            return False
        
        recent = list(self.history)[-self.DISAPPEAR_CONFIRM_FRAMES:]
        invisible_count = sum(1 for f in recent if not f.is_valid)
        
        return invisible_count >= self.DISAPPEAR_CONFIRM_FRAMES * 0.8
    
    def _handle_recovery(self, timestamp: float) -> Tuple[bool, str]:
        """处理恢复事件"""
        if self.current_event is None:
            return False, "No event to recover"
        
        self.current_event.recovery_time = timestamp
        self.current_event.status = "recovered"
        
        duration = timestamp - self.current_event.fall_time
        self.total_recoveries += 1
        
        print(f"\n[FallMonitor] ✅ 恢复! 事件: {self.current_event.event_id}")
        print(f"             跌倒持续时间: {duration:.1f}s")
        
        # 保存事件到历史
        self.event_history.append(self.current_event)
        
        # 重置状态
        self.current_event = None
        self.state = MonitorState.IDLE
        
        return True, f"Recovered after {duration:.1f}s"
    
    def _trigger_alert(self, timestamp: float) -> str:
        """
        触发报警
        
        ⚠️ 报警逻辑留白，仅返回消息
        """
        if self.current_event is None:
            return ""
        
        self.current_event.alert_triggered = True
        self.current_event.alert_time = timestamp
        self.current_event.status = "alerted"
        
        duration = timestamp - self.current_event.fall_time
        self.total_alerts += 1
        self.state = MonitorState.ALERTED
        
        # 保存事件到历史
        self.event_history.append(self.current_event)
        
        # ========================================
        # ⚠️ 报警逻辑留白 - 在这里添加实际报警代码
        # ========================================
        alert_message = (
            f"\n{'='*60}\n"
            f"🚨🚨🚨 需要报警! ALERT REQUIRED! 🚨🚨🚨\n"
            f"{'='*60}\n"
            f"事件ID: {self.current_event.event_id}\n"
            f"跌倒时间: {datetime.fromtimestamp(self.current_event.fall_time).strftime('%H:%M:%S')}\n"
            f"无响应时长: {duration:.1f} 秒\n"
            f"RF置信度: {self.current_event.rf_confidence:.2%}\n"
            f"{'='*60}\n"
        )
        
        print(alert_message)
        
        return alert_message
    
    def reset(self):
        """手动重置（报警后确认）"""
        print(f"\n[FallMonitor] 🔄 手动重置")
        self.current_event = None
        self.state = MonitorState.IDLE
    
    def get_status(self) -> dict:
        """获取当前状态摘要"""
        return {
            "state": self.state.value,
            "current_event": self.current_event.event_id if self.current_event else None,
            "total_falls": self.total_falls,
            "total_recoveries": self.total_recoveries,
            "total_alerts": self.total_alerts,
            "history_length": len(self.history)
        }


# ==================== 使用示例 ====================

def demo():
    """演示用法"""
    import random
    
    print("="*60)
    print("FallMonitor 演示")
    print("="*60)
    
    monitor = FallMonitor(fps=10.0)
    
    # 模拟帧数据
    def generate_keypoints(visible=True):
        if visible:
            # 生成有效关键点
            kpts = np.random.rand(5, 3)
            kpts[:, 2] = np.random.uniform(0.5, 0.9, 5)  # 置信度
        else:
            # 生成无效关键点（人消失）
            kpts = np.zeros((5, 3))
            kpts[:, 2] = np.random.uniform(0.0, 0.2, 5)  # 低置信度
        return kpts
    
    start_time = time.time()
    
    # 场景1: 正常状态 (5秒)
    print("\n[场景1] 正常状态...")
    for i in range(50):
        t = start_time + i * 0.1
        kpts = generate_keypoints(visible=True)
        monitor.add_keypoints(t, kpts)
        state, alert = monitor.update(t)
        time.sleep(0.02)  # 加速演示
    
    # 场景2: RF 检测到跌倒
    print("\n[场景2] RF 检测到跌倒...")
    fall_time = start_time + 5.0
    monitor.on_rf_fall_detected(fall_time, confidence=0.75)
    
    # 场景3: 人消失，监测中 (8秒，未超过报警阈值)
    print("\n[场景3] 人消失，监测中...")
    for i in range(80):
        t = fall_time + i * 0.1
        kpts = generate_keypoints(visible=False)  # 人消失
        monitor.add_keypoints(t, kpts)
        state, alert = monitor.update(t)
        if alert:
            print(f"Alert: {alert}")
            break
        time.sleep(0.02)
    
    # 场景4: 人恢复（如果还没报警）
    if monitor.state == MonitorState.MONITORING:
        print("\n[场景4] 人重新出现...")
        for i in range(20):
            t = fall_time + 8.0 + i * 0.1
            kpts = generate_keypoints(visible=True)  # 人出现
            monitor.add_keypoints(t, kpts)
            monitor.on_rf_normal_detected(t)
            state, alert = monitor.update(t)
            time.sleep(0.02)
    
    # 打印最终状态
    print("\n" + "="*60)
    print("最终状态:")
    print(monitor.get_status())
    print("="*60)


if __name__ == "__main__":
    demo()
