# -*- coding: utf-8 -*-
"""
关键点后处理过滤器
过滤YOLO的误检测（边缘假阳性、不合理的人体姿态等）
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class FilterConfig:
    """过滤器配置"""
    # 图像尺寸
    img_width: int = 100
    img_height: int = 100
    
    # 边缘区域定义（像素）- 减小边缘区域
    edge_margin: int = 8  # 从 12 减少到 8
    
    # 最小置信度（低于此值直接过滤）- 大幅降低以匹配YOLO阈值
    min_confidence: float = 0.25  # 从 0.65 降低到 0.25（略低于YOLO的0.3）
    
    # 边缘区域的置信度阈值（边缘需要更高置信度）
    edge_confidence: float = 0.50  # 从 0.90 降低到 0.50
    
    # 人体尺寸约束 - 放宽以适应摔倒姿势
    min_body_height: float = 8.0   # 从 18 降低到 8（摔倒时躺着身体高度会很小）
    max_body_height: float = 70.0  # 从 55 增加到 70
    min_body_width: float = 5.0    # 从 10 降低到 5
    
    # 关键点一致性检查 - 放宽
    max_shoulder_asymmetry: float = 35.0  # 从 25 增加到 35
    max_hip_asymmetry: float = 35.0       # 从 25 增加到 35
    
    # 时间一致性（帧间跳变检测）- 放宽
    max_position_jump: float = 35.0  # 从 20 增加到 35（摔倒时位移大）
    
    # 持续检测要求 - 放宽
    min_consecutive_detections: int = 2  # 从 5 减少到 2


class KeypointFilter:
    """关键点后处理过滤器"""
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self.history: List[dict] = []  # 历史检测记录
        self.consecutive_valid = 0  # 连续有效检测计数
        self.last_valid_position = None  # 上一个有效位置
    
    def is_in_edge_region(self, x: float, y: float) -> bool:
        """检查坐标是否在边缘区域"""
        margin = self.config.edge_margin
        return (x < margin or 
                x > self.config.img_width - margin or 
                y < margin or 
                y > self.config.img_height - margin)
    
    def check_single_keypoint(self, x: float, y: float, conf: float) -> Tuple[bool, str]:
        """
        检查单个关键点是否有效
        
        Returns:
            (is_valid, reason)
        """
        # 零值检测
        if x == 0 and y == 0 and conf == 0:
            return True, "no_detection"  # 没有检测到是正常的
        
        # 置信度检查
        if conf < self.config.min_confidence:
            return False, f"low_conf_{conf:.2f}"
        
        # 边缘区域需要更高置信度
        if self.is_in_edge_region(x, y):
            if conf < self.config.edge_confidence:
                return False, f"edge_low_conf_{conf:.2f}"
        
        return True, "valid"
    
    def check_body_geometry(self, keypoints: dict) -> Tuple[bool, str]:
        """
        检查人体几何合理性
        
        Args:
            keypoints: {
                'nose': (x, y, conf),
                'right_shoulder': (x, y, conf),
                'left_shoulder': (x, y, conf),
                'right_hip': (x, y, conf),
                'left_hip': (x, y, conf)
            }
        """
        # 提取高置信度关键点
        valid_kps = {}
        for name, (x, y, conf) in keypoints.items():
            if conf > self.config.min_confidence:
                valid_kps[name] = (x, y, conf)
        
        # 至少需要3个有效关键点
        if len(valid_kps) < 3:
            return False, f"too_few_keypoints_{len(valid_kps)}"
        
        # 检查肩膀对称性
        if 'right_shoulder' in valid_kps and 'left_shoulder' in valid_kps:
            rs_y = valid_kps['right_shoulder'][1]
            ls_y = valid_kps['left_shoulder'][1]
            if abs(rs_y - ls_y) > self.config.max_shoulder_asymmetry:
                return False, f"shoulder_asymmetry_{abs(rs_y - ls_y):.1f}"
        
        # 检查髋部对称性
        if 'right_hip' in valid_kps and 'left_hip' in valid_kps:
            rh_y = valid_kps['right_hip'][1]
            lh_y = valid_kps['left_hip'][1]
            if abs(rh_y - lh_y) > self.config.max_hip_asymmetry:
                return False, f"hip_asymmetry_{abs(rh_y - lh_y):.1f}"
        
        # 检查身体高度（肩到髋）
        shoulder_y = None
        hip_y = None
        
        if 'right_shoulder' in valid_kps or 'left_shoulder' in valid_kps:
            shoulder_ys = []
            if 'right_shoulder' in valid_kps:
                shoulder_ys.append(valid_kps['right_shoulder'][1])
            if 'left_shoulder' in valid_kps:
                shoulder_ys.append(valid_kps['left_shoulder'][1])
            shoulder_y = np.mean(shoulder_ys)
        
        if 'right_hip' in valid_kps or 'left_hip' in valid_kps:
            hip_ys = []
            if 'right_hip' in valid_kps:
                hip_ys.append(valid_kps['right_hip'][1])
            if 'left_hip' in valid_kps:
                hip_ys.append(valid_kps['left_hip'][1])
            hip_y = np.mean(hip_ys)
        
        if shoulder_y is not None and hip_y is not None:
            body_height = abs(hip_y - shoulder_y)
            if body_height < self.config.min_body_height:
                return False, f"body_too_small_{body_height:.1f}"
            if body_height > self.config.max_body_height:
                return False, f"body_too_large_{body_height:.1f}"
        
        # 检查身体宽度（肩宽）
        if 'right_shoulder' in valid_kps and 'left_shoulder' in valid_kps:
            rs_x = valid_kps['right_shoulder'][0]
            ls_x = valid_kps['left_shoulder'][0]
            body_width = abs(rs_x - ls_x)
            if body_width < self.config.min_body_width:
                return False, f"body_too_narrow_{body_width:.1f}"
        
        # 检查所有有效关键点是否都在边缘区域
        all_in_edge = all(
            self.is_in_edge_region(x, y) 
            for name, (x, y, conf) in valid_kps.items()
        )
        if all_in_edge and len(valid_kps) >= 3:
            return False, "all_keypoints_in_edge"
        
        return True, "valid_geometry"
    
    def check_temporal_consistency(self, keypoints: dict) -> Tuple[bool, str]:
        """
        检查时间一致性（帧间跳变）
        """
        if self.last_valid_position is None:
            return True, "first_detection"
        
        # 计算当前位置（使用鼻子或肩膀中心）
        current_pos = None
        
        if 'nose' in keypoints and keypoints['nose'][2] > self.config.min_confidence:
            current_pos = (keypoints['nose'][0], keypoints['nose'][1])
        elif ('right_shoulder' in keypoints and 'left_shoulder' in keypoints and
              keypoints['right_shoulder'][2] > self.config.min_confidence and
              keypoints['left_shoulder'][2] > self.config.min_confidence):
            rs = keypoints['right_shoulder']
            ls = keypoints['left_shoulder']
            current_pos = ((rs[0] + ls[0]) / 2, (rs[1] + ls[1]) / 2)
        
        if current_pos is None:
            return True, "no_position"
        
        # 计算位移
        dx = current_pos[0] - self.last_valid_position[0]
        dy = current_pos[1] - self.last_valid_position[1]
        jump = np.sqrt(dx**2 + dy**2)
        
        if jump > self.config.max_position_jump:
            return False, f"position_jump_{jump:.1f}"
        
        return True, "consistent"
    
    def filter_keypoints(self, keypoints: dict, frame_index: int = -1) -> dict:
        """
        过滤关键点检测结果
        
        Args:
            keypoints: 原始关键点字典
            frame_index: 帧索引（用于调试）
        
        Returns:
            dict: {
                'valid': bool,
                'keypoints': dict (过滤后的关键点，无效的设为0),
                'reason': str,
                'confidence_score': float
            }
        """
        # 构建关键点字典
        kp_dict = {
            'nose': (keypoints.get('nose_x', 0), keypoints.get('nose_y', 0), 
                     keypoints.get('nose_conf', 0)),
            'right_shoulder': (keypoints.get('right_shoulder_x', 0), 
                               keypoints.get('right_shoulder_y', 0),
                               keypoints.get('right_shoulder_conf', 0)),
            'left_shoulder': (keypoints.get('left_shoulder_x', 0), 
                              keypoints.get('left_shoulder_y', 0),
                              keypoints.get('left_shoulder_conf', 0)),
            'right_hip': (keypoints.get('right_hip_x', 0), 
                          keypoints.get('right_hip_y', 0),
                          keypoints.get('right_hip_conf', 0)),
            'left_hip': (keypoints.get('left_hip_x', 0), 
                         keypoints.get('left_hip_y', 0),
                         keypoints.get('left_hip_conf', 0)),
        }
        
        # 检查是否全为0（无检测）
        all_zero = all(
            x == 0 and y == 0 and conf == 0 
            for x, y, conf in kp_dict.values()
        )
        if all_zero:
            return {
                'valid': True,
                'keypoints': keypoints,
                'reason': 'no_detection',
                'confidence_score': 0.0,
                'is_person_present': False
            }
        
        # 1. 检查几何合理性
        geom_valid, geom_reason = self.check_body_geometry(kp_dict)
        if not geom_valid:
            # 将无效检测清零
            filtered_kp = {k: 0 for k in keypoints}
            filtered_kp['frame_index'] = keypoints.get('frame_index', frame_index)
            return {
                'valid': False,
                'keypoints': filtered_kp,
                'reason': geom_reason,
                'confidence_score': 0.0,
                'is_person_present': False
            }
        
        # 2. 检查时间一致性
        temp_valid, temp_reason = self.check_temporal_consistency(kp_dict)
        
        # 计算平均置信度
        confs = [conf for x, y, conf in kp_dict.values() if conf > 0]
        avg_conf = np.mean(confs) if confs else 0
        
        # 更新状态
        if geom_valid and temp_valid:
            self.consecutive_valid += 1
            # 更新最后有效位置
            if kp_dict['nose'][2] > self.config.min_confidence:
                self.last_valid_position = (kp_dict['nose'][0], kp_dict['nose'][1])
            elif (kp_dict['right_shoulder'][2] > self.config.min_confidence and
                  kp_dict['left_shoulder'][2] > self.config.min_confidence):
                rs = kp_dict['right_shoulder']
                ls = kp_dict['left_shoulder']
                self.last_valid_position = ((rs[0] + ls[0]) / 2, (rs[1] + ls[1]) / 2)
        else:
            self.consecutive_valid = 0
        
        # 需要连续检测才认为有效
        is_stable = self.consecutive_valid >= self.config.min_consecutive_detections
        
        return {
            'valid': geom_valid and temp_valid,
            'keypoints': keypoints,
            'reason': f"{geom_reason}|{temp_reason}",
            'confidence_score': avg_conf,
            'is_person_present': geom_valid and avg_conf > self.config.min_confidence,
            'is_stable': is_stable,
            'consecutive_count': self.consecutive_valid
        }
    
    def reset(self):
        """重置过滤器状态"""
        self.history.clear()
        self.consecutive_valid = 0
        self.last_valid_position = None


def filter_keypoints_dataframe(df, config: FilterConfig = None):
    """
    过滤整个DataFrame的关键点
    
    Returns:
        filtered_df: 过滤后的DataFrame
        stats: 过滤统计信息
    """
    import pandas as pd
    
    filter = KeypointFilter(config)
    
    filtered_rows = []
    stats = {
        'total': len(df),
        'valid': 0,
        'filtered': 0,
        'no_detection': 0,
        'reasons': {}
    }
    
    for idx, row in df.iterrows():
        keypoints = row.to_dict()
        result = filter.filter_keypoints(keypoints, frame_index=idx)
        
        if result['reason'] == 'no_detection':
            stats['no_detection'] += 1
            filtered_rows.append(row.to_dict())
        elif result['valid']:
            stats['valid'] += 1
            filtered_rows.append(row.to_dict())
        else:
            stats['filtered'] += 1
            # 记录过滤原因
            reason = result['reason'].split('|')[0]
            stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
            # 添加清零的行
            filtered_rows.append(result['keypoints'])
    
    filtered_df = pd.DataFrame(filtered_rows)
    return filtered_df, stats


if __name__ == "__main__":
    import pandas as pd
    
    # 测试过滤器
    csv_path = r'C:\Users\LTTC\Desktop\fyp\fall_detection_pipeline_1130\output\raw_data_20251130_123456\keypoints.csv'
    df = pd.read_csv(csv_path)
    
    print("=== 关键点过滤器测试 ===")
    print(f"原始帧数: {len(df)}")
    
    filtered_df, stats = filter_keypoints_dataframe(df)
    
    print(f"\n过滤统计:")
    print(f"  有效帧: {stats['valid']}")
    print(f"  无检测帧: {stats['no_detection']}")
    print(f"  被过滤帧: {stats['filtered']}")
    print(f"\n过滤原因分布:")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
