# -*- coding: utf-8 -*-
"""
Fall Classifier V6 - 去除绝对位置特征，增加动态特征

改进:
  - 去除 mean_height, min_height, mean/min/max_body_length, total_frames, valid_frames
    这些绝对位置特征是误报的主要来源（站远/站近/身高不同就变化）
  - 新增加速度特征（二阶导）：跌倒的物理本质是"突然加速下落"
  - 新增相对比例特征：height_ratio, body_length_ratio
  - 新增横向运动特征：x_displacement, x_velocity_std
  - 新增体态稳定性特征：shoulder_hip_ratio_change, pose_angle_std

三分类: Normal(0), Fall(1), Backward(2)
窗口大小: 50帧 (5秒 @ 10 FPS)
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from config import RF_CONFIG


# Default model path
DEFAULT_MODEL_PATH = str(Path(__file__).parent / 'fall_classifier_v6.joblib')

# Class names
CLASS_NAMES = ['Normal', 'Fall', 'Backward']

# V6 配置
TARGET_FPS = 10
WINDOW_SIZE = 50  # 5秒 @ 10 FPS


def calculate_body_length(row) -> float:
    """Calculate body length: distance from nose to hip center"""
    nose_conf = row.get('nose_conf', 0)
    left_hip_conf = row.get('left_hip_conf', 0)
    right_hip_conf = row.get('right_hip_conf', 0)
    
    if nose_conf < 0.3:
        return -1
    
    nose_x = row['nose_x']
    nose_y = row['nose_y']
    
    if left_hip_conf > 0.3 and right_hip_conf > 0.3:
        hip_x = (row['left_hip_x'] + row['right_hip_x']) / 2
        hip_y = (row['left_hip_y'] + row['right_hip_y']) / 2
    elif left_hip_conf > 0.3:
        hip_x = row['left_hip_x']
        hip_y = row['left_hip_y']
    elif right_hip_conf > 0.3:
        hip_x = row['right_hip_x']
        hip_y = row['right_hip_y']
    else:
        return -1
    
    length = np.sqrt((nose_x - hip_x)**2 + (nose_y - hip_y)**2)
    return length


class FallClassifierV6:
    """
    Fall Classifier V6 - 去除绝对位置，增强动态特征
    
    Classes:
    - 0: Normal
    - 1: Fall (forward/side)
    - 2: Backward Fall
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading V6 classifier (10 FPS): {model_path}")
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            self.model = model_data['classifier']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data.get('class_names', CLASS_NAMES)
            self.n_classes = model_data.get('n_classes', 3)
            self.window_size = model_data.get('window_size', WINDOW_SIZE)
            self.target_fps = model_data.get('target_fps', TARGET_FPS)
        else:
            raise ValueError("Invalid model format")
        
        self.min_valid_frames = 10
    
    def extract_features_from_df(self, df) -> dict:
        """Extract features from DataFrame (V6 - 去除绝对位置，增强动态)"""
        features = {}
        
        total_frames = len(df)
        
        # ===== 1. 置信度特征 (保留, 4个) =====
        nose_conf = df['nose_conf'].values
        
        half = total_frames // 2
        first_half_conf = nose_conf[:half].mean() if half > 0 else 0
        second_half_conf = nose_conf[half:].mean() if half > 0 else 0
        
        features['conf_drop'] = first_half_conf - second_half_conf
        features['first_half_conf'] = first_half_conf
        features['second_half_conf'] = second_half_conf
        
        second_half_low_conf = (nose_conf[half:] < 0.3).sum()
        features['disappear_ratio'] = second_half_low_conf / max(total_frames - half, 1)
        
        # ===== 2. 有效帧比例 (只保留比率, 1个) =====
        valid_mask = df['nose_conf'] > 0.3
        valid_df = df[valid_mask].copy()
        
        valid_frames = len(valid_df)
        valid_ratio = valid_frames / total_frames if total_frames > 0 else 0
        
        features['valid_frame_ratio'] = valid_ratio
        
        # 默认值列表（V6 新特征集）
        default_keys = [
            # Y速度 (保留, 4个)
            'mean_vy', 'max_vy', 'min_vy', 'std_vy',
            # Y加速度 (新增, 3个) ★核心改进
            'max_acceleration', 'mean_abs_acceleration', 'acceleration_std',
            # 高度变化 (只保留相对变化, 3个) ★去掉mean_height, min_height
            'height_change', 'abs_height_change', 'height_ratio',
            # 肩髋比 (保留+新增变化率, 3个)
            'mean_shoulder_hip_ratio', 'min_shoulder_hip_ratio', 'shoulder_hip_ratio_change',
            # 姿态角度 (保留+新增std, 3个)
            'mean_pose_angle', 'max_pose_angle_change', 'pose_angle_std',
            # 帧间隔 (保留, 2个)
            'frame_gap_std', 'max_frame_gap',
            # 跌倒动态 (保留+新增, 3个)
            'y_velocity_increase', 'sudden_drop', 'velocity_spike_count',
            # 体长 (只保留相对变化, 3个) ★去掉mean/min/max绝对值
            'body_length_ratio', 'body_length_change', 'body_length_shrink_rate',
            # 横向运动 (新增, 2个) ★
            'x_displacement', 'x_velocity_std',
        ]
        
        if valid_frames < self.min_valid_frames:
            for key in default_keys:
                features[key] = 0
            return features
        
        # ===== 3. 身体Y坐标 (加权平均) =====
        y_cols = [col for col in valid_df.columns if col.endswith('_y')]
        conf_cols = [col.replace('_y', '_conf') for col in y_cols]
        
        body_y = []
        for idx, row in valid_df.iterrows():
            weights = []
            y_values = []
            for y_col, conf_col in zip(y_cols, conf_cols):
                if conf_col in row and row[conf_col] > 0.3:
                    weights.append(row[conf_col])
                    y_values.append(row[y_col])
            if weights:
                body_y.append(np.average(y_values, weights=weights))
            else:
                body_y.append(np.nan)
        
        body_y = np.array(body_y)
        body_y = body_y[~np.isnan(body_y)]
        
        if len(body_y) < self.min_valid_frames:
            for key in default_keys:
                features[key] = 0
            return features
        
        # ===== 4. Y速度 (一阶导, 保留) =====
        vy = np.diff(body_y)
        features['mean_vy'] = np.mean(vy)
        features['max_vy'] = np.max(vy)
        features['min_vy'] = np.min(vy)
        features['std_vy'] = np.std(vy)
        
        # ===== 5. Y加速度 (二阶导, 新增) ★ =====
        if len(vy) >= 2:
            ay = np.diff(vy)  # 加速度 = 速度的变化率
            features['max_acceleration'] = np.max(ay)
            features['mean_abs_acceleration'] = np.mean(np.abs(ay))
            features['acceleration_std'] = np.std(ay)
        else:
            features['max_acceleration'] = 0
            features['mean_abs_acceleration'] = 0
            features['acceleration_std'] = 0
        
        # ===== 6. 高度变化 (只保留相对量) ★ =====
        features['height_change'] = body_y[-1] - body_y[0]
        features['abs_height_change'] = abs(body_y[-1] - body_y[0])
        # 相对比例：最低点/最高点 (跌倒时会显著偏离1)
        max_h = np.max(body_y)
        min_h = np.min(body_y)
        features['height_ratio'] = min_h / max_h if max_h > 0 else 1.0
        
        # ===== 7. 肩髋比 (保留 + 新增变化率) =====
        shoulder_hip_ratios = []
        for idx, row in valid_df.iterrows():
            if row.get('right_shoulder_conf', 0) > 0.3 and row.get('left_shoulder_conf', 0) > 0.3:
                if row.get('right_hip_conf', 0) > 0.3 and row.get('left_hip_conf', 0) > 0.3:
                    shoulder_y = (row['right_shoulder_y'] + row['left_shoulder_y']) / 2
                    hip_y = (row['right_hip_y'] + row['left_hip_y']) / 2
                    ratio = hip_y - shoulder_y if hip_y != 0 else 0
                    shoulder_hip_ratios.append(ratio)
        
        if shoulder_hip_ratios:
            features['mean_shoulder_hip_ratio'] = np.mean(shoulder_hip_ratios)
            features['min_shoulder_hip_ratio'] = np.min(shoulder_hip_ratios)
            # 新增：肩髋比变化率（体态变化速度）
            if len(shoulder_hip_ratios) > 1:
                features['shoulder_hip_ratio_change'] = shoulder_hip_ratios[-1] - shoulder_hip_ratios[0]
            else:
                features['shoulder_hip_ratio_change'] = 0
        else:
            features['mean_shoulder_hip_ratio'] = 0
            features['min_shoulder_hip_ratio'] = 0
            features['shoulder_hip_ratio_change'] = 0
        
        # ===== 8. 姿态角度 (保留 + 新增std) =====
        pose_angles = []
        for idx, row in valid_df.iterrows():
            if row.get('right_shoulder_conf', 0) > 0.3 and row.get('left_shoulder_conf', 0) > 0.3:
                dx = row['right_shoulder_x'] - row['left_shoulder_x']
                dy = row['right_shoulder_y'] - row['left_shoulder_y']
                if dx != 0:
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    pose_angles.append(abs(angle))
        
        if pose_angles:
            features['mean_pose_angle'] = np.mean(pose_angles)
            pose_angle_changes = np.abs(np.diff(pose_angles)) if len(pose_angles) > 1 else [0]
            features['max_pose_angle_change'] = np.max(pose_angle_changes)
            features['pose_angle_std'] = np.std(pose_angles)  # 新增
        else:
            features['mean_pose_angle'] = 0
            features['max_pose_angle_change'] = 0
            features['pose_angle_std'] = 0
        
        # ===== 9. 帧间隔 (保留) =====
        if 'frame_index' in valid_df.columns:
            frame_indices = valid_df['frame_index'].values
            frame_gaps = np.diff(frame_indices)
            features['frame_gap_std'] = np.std(frame_gaps) if len(frame_gaps) > 0 else 0
            features['max_frame_gap'] = np.max(frame_gaps) if len(frame_gaps) > 0 else 0
        else:
            features['frame_gap_std'] = 0
            features['max_frame_gap'] = 0
        
        # ===== 10. 跌倒动态 (保留 + 新增velocity_spike) =====
        if len(vy) >= 4:
            first_half_vy = vy[:len(vy)//2]
            second_half_vy = vy[len(vy)//2:]
            features['y_velocity_increase'] = np.mean(second_half_vy) - np.mean(first_half_vy)
        else:
            features['y_velocity_increase'] = 0
        
        # 突然下降（5帧窗口 @ 10 FPS = 0.5秒）
        window = 5
        sudden_drops = []
        for i in range(len(body_y) - window):
            drop = body_y[i + window] - body_y[i]
            sudden_drops.append(drop)
        features['sudden_drop'] = np.max(sudden_drops) if sudden_drops else 0
        
        # 速度突变帧计数：|vy| 超过均值+2*std的帧数比例
        vy_threshold = np.mean(np.abs(vy)) + 2 * np.std(vy) if np.std(vy) > 0 else 999
        features['velocity_spike_count'] = np.sum(np.abs(vy) > vy_threshold) / len(vy)
        
        # ===== 11. 体长特征 (只保留相对变化) ★ =====
        body_lengths = []
        for idx, row in valid_df.iterrows():
            length = calculate_body_length(row)
            if length > 0:
                body_lengths.append(length)
        
        if len(body_lengths) >= 3:
            max_bl = np.max(body_lengths)
            min_bl = np.min(body_lengths)
            features['body_length_ratio'] = min_bl / max_bl if max_bl > 0 else 1.0  # 新增
            features['body_length_change'] = body_lengths[-1] - body_lengths[0]
            features['body_length_shrink_rate'] = (body_lengths[-1] - body_lengths[0]) / len(body_lengths)
        else:
            features['body_length_ratio'] = 1.0
            features['body_length_change'] = 0
            features['body_length_shrink_rate'] = 0
        
        # ===== 12. 横向运动 (新增) ★ =====
        # 身体X坐标
        body_x = []
        for idx, row in valid_df.iterrows():
            weights = []
            x_values = []
            x_cols = [col for col in valid_df.columns if col.endswith('_x')]
            cx_cols = [col.replace('_x', '_conf') for col in x_cols]
            for x_col, conf_col in zip(x_cols, cx_cols):
                if conf_col in row and row[conf_col] > 0.3:
                    weights.append(row[conf_col])
                    x_values.append(row[x_col])
            if weights:
                body_x.append(np.average(x_values, weights=weights))
            else:
                body_x.append(np.nan)
        
        body_x = np.array(body_x)
        body_x = body_x[~np.isnan(body_x)]
        
        if len(body_x) >= 2:
            features['x_displacement'] = abs(body_x[-1] - body_x[0])
            vx = np.diff(body_x)
            features['x_velocity_std'] = np.std(vx)
        else:
            features['x_displacement'] = 0
            features['x_velocity_std'] = 0
        
        return features
    
    def extract_features(self, keypoints_sequence):
        """Extract features from keypoints sequence"""
        if len(keypoints_sequence) < 2:
            return None
        
        # Convert to DataFrame
        if keypoints_sequence.shape[1] == 15:
            columns = []
            for kp in ['nose', 'right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']:
                columns.extend([f'{kp}_x', f'{kp}_y', f'{kp}_conf'])
            df = pd.DataFrame(keypoints_sequence, columns=columns)
            df['frame_index'] = range(len(df))
        else:
            columns = []
            for kp in ['nose', 'right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']:
                columns.extend([f'{kp}_x', f'{kp}_y'])
            df = pd.DataFrame(keypoints_sequence, columns=columns)
            df['frame_index'] = range(len(df))
            for kp in ['nose', 'right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']:
                df[f'{kp}_conf'] = np.where(
                    (df[f'{kp}_x'] != 0) | (df[f'{kp}_y'] != 0), 0.9, 0.0
                )
        
        features_dict = self.extract_features_from_df(df)
        
        if self.feature_names:
            feature_values = [features_dict.get(k, 0) for k in self.feature_names]
        else:
            feature_values = [features_dict[k] for k in sorted(features_dict.keys())]
        
        return np.array(feature_values)
    
    def predict(self, keypoints_sequence):
        """Predict fall type"""
        features = self.extract_features(keypoints_sequence)
        
        if features is None:
            return {
                'prediction': -1,
                'label': 'Unknown',
                'confidence': 0.0,
                'probabilities': [0, 0, 0],
                'valid': False,
                'is_fall': False
            }
        
        # Check for no-person scene
        if self.feature_names:
            features_dict = dict(zip(self.feature_names, features))
        else:
            features_dict = {}
        
        valid_frame_ratio = features_dict.get('valid_frame_ratio', 1.0)
        first_half_conf = features_dict.get('first_half_conf', 1.0)
        second_half_conf = features_dict.get('second_half_conf', 1.0)
        
        is_no_person_scene = (
            valid_frame_ratio < 0.2 or
            (first_half_conf < 0.1 and second_half_conf < 0.1)
        )
        
        if is_no_person_scene:
            return {
                'prediction': 0,
                'label': 'No Person',
                'confidence': 0.0,
                'probabilities': [1, 0, 0],
                'valid': False,
                'is_fall': False,
                'reason': f'no_detection (valid_ratio={valid_frame_ratio:.2f})'
            }
        
        # Predict
        prediction = self.model.predict(features.reshape(1, -1))[0]
        
        # Get probabilities
        try:
            proba = self.model.predict_proba(features.reshape(1, -1))[0]
            confidence = proba[prediction]
            probabilities = proba.tolist()
        except:
            confidence = 1.0
            probabilities = [0, 0, 0]
            probabilities[prediction] = 1.0
        
        label = self.class_names[prediction] if prediction < len(self.class_names) else 'Unknown'
        is_fall = prediction in [1, 2]
        
        # Debug output
        print(f"[RF V6] pred={prediction} ({label}), probs=[N:{proba[0]:.2f}, F:{proba[1]:.2f}, B:{proba[2]:.2f}], conf={confidence:.2f}")
        
        return {
            'prediction': int(prediction),
            'label': label,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'valid': True,
            'is_fall': is_fall
        }
    
    def predict_from_csv(self, csv_path):
        """Predict from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            columns = ['nose_x', 'nose_y', 'nose_conf',
                       'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_conf',
                       'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_conf',
                       'right_hip_x', 'right_hip_y', 'right_hip_conf',
                       'left_hip_x', 'left_hip_y', 'left_hip_conf']
            keypoints = df[columns].values
            return self.predict(keypoints)
        except Exception as e:
            return {
                'prediction': -1,
                'label': f'Error: {e}',
                'confidence': 0.0,
                'valid': False,
                'is_fall': False
            }


# Alias
FallClassifier = FallClassifierV6
