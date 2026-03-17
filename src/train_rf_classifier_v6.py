# -*- coding: utf-8 -*-
"""
Train Random Forest Fall Detection Classifier V6

改进 (相对V5):
  - 去除绝对位置特征 (mean_height, min_height, mean/min/max_body_length 等)
  - 新增加速度特征 (max_acceleration, mean_abs_acceleration, acceleration_std)
  - 新增相对比例特征 (height_ratio, body_length_ratio)
  - 新增横向运动特征 (x_displacement, x_velocity_std)
  - 新增体态稳定性特征 (shoulder_hip_ratio_change, pose_angle_std, velocity_spike_count)

三分类: Normal(0), Fall(1), Backward(2)
窗口: 50帧 @ 10 FPS = 5秒
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ======================= 配置 =======================

ORIGINAL_FPS = 30
TARGET_FPS = 10
DOWNSAMPLE_RATIO = ORIGINAL_FPS // TARGET_FPS  # 3

WINDOW_SECONDS = 5
WINDOW_SIZE = WINDOW_SECONDS * TARGET_FPS  # 50

MIN_VALID_FRAMES = 10

CLASS_NAMES = ['Normal', 'Fall', 'Backward']

BASE_DIR = Path(__file__).parent
TRAINING_DATA_DIR = BASE_DIR / "training_data"


def load_keypoints_csv(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Failed to load {csv_path}: {e}")
        return None


def downsample_to_target_fps(df: pd.DataFrame, ratio: int = DOWNSAMPLE_RATIO) -> pd.DataFrame:
    if ratio <= 1:
        return df
    indices = list(range(0, len(df), ratio))
    downsampled = df.iloc[indices].reset_index(drop=True)
    if 'frame_index' in downsampled.columns:
        downsampled['frame_index'] = range(len(downsampled))
    return downsampled


def calculate_body_length(row) -> float:
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
    
    return np.sqrt((nose_x - hip_x)**2 + (nose_y - hip_y)**2)


def extract_sequence_features_v6(df: pd.DataFrame) -> Dict:
    """
    V6 特征提取 — 去除绝对位置，增强动态特征
    
    总特征数: 27个 (V5: 28个)
    去掉: mean_height, min_height, mean/min/max_body_length, total_frames, valid_frames (7个)
    新增: max_acceleration, mean_abs_acceleration, acceleration_std, 
          height_ratio, body_length_ratio,
          x_displacement, x_velocity_std,
          shoulder_hip_ratio_change, pose_angle_std, velocity_spike_count (10个)
    保留: V5原有 11个特征
    """
    features = {}
    total_frames = len(df)
    
    # ===== 1. 置信度特征 (4个) =====
    nose_conf = df['nose_conf'].values
    half = total_frames // 2
    first_half_conf = nose_conf[:half].mean() if half > 0 else 0
    second_half_conf = nose_conf[half:].mean() if half > 0 else 0
    
    features['conf_drop'] = first_half_conf - second_half_conf
    features['first_half_conf'] = first_half_conf
    features['second_half_conf'] = second_half_conf
    
    second_half_low_conf = (nose_conf[half:] < 0.3).sum()
    features['disappear_ratio'] = second_half_low_conf / max(total_frames - half, 1)
    
    # ===== 2. 有效帧比例 (1个) =====
    valid_mask = df['nose_conf'] > 0.3
    valid_df = df[valid_mask].copy()
    valid_frames = len(valid_df)
    valid_ratio = valid_frames / total_frames if total_frames > 0 else 0
    features['valid_frame_ratio'] = valid_ratio
    
    # 默认值
    default_keys = [
        'mean_vy', 'max_vy', 'min_vy', 'std_vy',
        'max_acceleration', 'mean_abs_acceleration', 'acceleration_std',
        'height_change', 'abs_height_change', 'height_ratio',
        'mean_shoulder_hip_ratio', 'min_shoulder_hip_ratio', 'shoulder_hip_ratio_change',
        'mean_pose_angle', 'max_pose_angle_change', 'pose_angle_std',
        'frame_gap_std', 'max_frame_gap',
        'y_velocity_increase', 'sudden_drop', 'velocity_spike_count',
        'body_length_ratio', 'body_length_change', 'body_length_shrink_rate',
        'x_displacement', 'x_velocity_std',
    ]
    
    if valid_frames < MIN_VALID_FRAMES:
        for key in default_keys:
            features[key] = 0
        return features
    
    # ===== 3. 身体Y/X坐标 =====
    y_cols = [col for col in valid_df.columns if col.endswith('_y')]
    conf_cols = [col.replace('_y', '_conf') for col in y_cols]
    x_cols = [col for col in valid_df.columns if col.endswith('_x')]
    xconf_cols = [col.replace('_x', '_conf') for col in x_cols]
    
    body_y = []
    body_x = []
    for idx, row in valid_df.iterrows():
        # Y坐标
        yw, yv = [], []
        for y_col, conf_col in zip(y_cols, conf_cols):
            if conf_col in row and row[conf_col] > 0.3:
                yw.append(row[conf_col])
                yv.append(row[y_col])
        body_y.append(np.average(yv, weights=yw) if yw else np.nan)
        
        # X坐标
        xw, xv = [], []
        for x_col, conf_col in zip(x_cols, xconf_cols):
            if conf_col in row and row[conf_col] > 0.3:
                xw.append(row[conf_col])
                xv.append(row[x_col])
        body_x.append(np.average(xv, weights=xw) if xw else np.nan)
    
    body_y = np.array(body_y)
    body_x = np.array(body_x)
    body_y = body_y[~np.isnan(body_y)]
    body_x = body_x[~np.isnan(body_x)]
    
    if len(body_y) < MIN_VALID_FRAMES:
        for key in default_keys:
            features[key] = 0
        return features
    
    # ===== 4. Y速度 (一阶导, 4个) =====
    vy = np.diff(body_y)
    features['mean_vy'] = np.mean(vy)
    features['max_vy'] = np.max(vy)
    features['min_vy'] = np.min(vy)
    features['std_vy'] = np.std(vy)
    
    # ===== 5. Y加速度 (二阶导, 3个) ★ =====
    if len(vy) >= 2:
        ay = np.diff(vy)
        features['max_acceleration'] = np.max(ay)
        features['mean_abs_acceleration'] = np.mean(np.abs(ay))
        features['acceleration_std'] = np.std(ay)
    else:
        features['max_acceleration'] = 0
        features['mean_abs_acceleration'] = 0
        features['acceleration_std'] = 0
    
    # ===== 6. 高度变化 (相对量, 3个) ★ =====
    features['height_change'] = body_y[-1] - body_y[0]
    features['abs_height_change'] = abs(body_y[-1] - body_y[0])
    max_h = np.max(body_y)
    min_h = np.min(body_y)
    features['height_ratio'] = min_h / max_h if max_h > 0 else 1.0
    
    # ===== 7. 肩髋比 (3个) =====
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
        features['shoulder_hip_ratio_change'] = shoulder_hip_ratios[-1] - shoulder_hip_ratios[0] if len(shoulder_hip_ratios) > 1 else 0
    else:
        features['mean_shoulder_hip_ratio'] = 0
        features['min_shoulder_hip_ratio'] = 0
        features['shoulder_hip_ratio_change'] = 0
    
    # ===== 8. 姿态角度 (3个) =====
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
        features['pose_angle_std'] = np.std(pose_angles)
    else:
        features['mean_pose_angle'] = 0
        features['max_pose_angle_change'] = 0
        features['pose_angle_std'] = 0
    
    # ===== 9. 帧间隔 (2个) =====
    if 'frame_index' in valid_df.columns:
        frame_indices = valid_df['frame_index'].values
        frame_gaps = np.diff(frame_indices)
        features['frame_gap_std'] = np.std(frame_gaps) if len(frame_gaps) > 0 else 0
        features['max_frame_gap'] = np.max(frame_gaps) if len(frame_gaps) > 0 else 0
    else:
        features['frame_gap_std'] = 0
        features['max_frame_gap'] = 0
    
    # ===== 10. 跌倒动态 (3个) =====
    if len(vy) >= 4:
        first_half_vy = vy[:len(vy)//2]
        second_half_vy = vy[len(vy)//2:]
        features['y_velocity_increase'] = np.mean(second_half_vy) - np.mean(first_half_vy)
    else:
        features['y_velocity_increase'] = 0
    
    window = 5
    sudden_drops = []
    for i in range(len(body_y) - window):
        drop = body_y[i + window] - body_y[i]
        sudden_drops.append(drop)
    features['sudden_drop'] = np.max(sudden_drops) if sudden_drops else 0
    
    vy_threshold = np.mean(np.abs(vy)) + 2 * np.std(vy) if np.std(vy) > 0 else 999
    features['velocity_spike_count'] = np.sum(np.abs(vy) > vy_threshold) / len(vy)
    
    # ===== 11. 体长 (相对变化, 3个) ★ =====
    body_lengths = []
    for idx, row in valid_df.iterrows():
        length = calculate_body_length(row)
        if length > 0:
            body_lengths.append(length)
    
    if len(body_lengths) >= 3:
        max_bl = np.max(body_lengths)
        min_bl = np.min(body_lengths)
        features['body_length_ratio'] = min_bl / max_bl if max_bl > 0 else 1.0
        features['body_length_change'] = body_lengths[-1] - body_lengths[0]
        features['body_length_shrink_rate'] = (body_lengths[-1] - body_lengths[0]) / len(body_lengths)
    else:
        features['body_length_ratio'] = 1.0
        features['body_length_change'] = 0
        features['body_length_shrink_rate'] = 0
    
    # ===== 12. 横向运动 (2个) ★ =====
    if len(body_x) >= 2:
        features['x_displacement'] = abs(body_x[-1] - body_x[0])
        vx = np.diff(body_x)
        features['x_velocity_std'] = np.std(vx)
    else:
        features['x_displacement'] = 0
        features['x_velocity_std'] = 0
    
    return features


def load_training_data():
    """加载所有训练数据（带降采样）"""
    all_features = []
    all_labels = []
    
    print(f"\n降采样: {ORIGINAL_FPS} FPS -> {TARGET_FPS} FPS (每{DOWNSAMPLE_RATIO}帧取1帧)")
    print(f"窗口: {WINDOW_SIZE} 帧 ({WINDOW_SECONDS}秒 @ {TARGET_FPS} FPS)")
    
    categories = [
        ('normal', 0, ['normal', 'normal_1217']),
        ('fall', 1, ['fall', 'fall_1130']),
        ('backward', 2, ['fall_1217', 'backward_0203']),
    ]
    
    for category_name, label, subdirs in categories:
        print(f"\n加载 {category_name.upper()} (label={label})...")
        count = 0
        for subdir in subdirs:
            category_dir = TRAINING_DATA_DIR / subdir
            if not category_dir.exists():
                print(f"  [跳过] {category_dir} 不存在")
                continue
            
            for folder in sorted(category_dir.iterdir()):
                if folder.is_dir():
                    csv_path = folder / "keypoints.csv"
                    if csv_path.exists():
                        df = load_keypoints_csv(str(csv_path))
                        if df is None or len(df) < MIN_VALID_FRAMES * DOWNSAMPLE_RATIO:
                            continue
                        
                        df_down = downsample_to_target_fps(df, DOWNSAMPLE_RATIO)
                        features = extract_sequence_features_v6(df_down)
                        all_features.append(features)
                        all_labels.append(label)
                        count += 1
                        
                        # 显示关键特征
                        h_chg = features.get('height_change', 0)
                        max_acc = features.get('max_acceleration', 0)
                        bl_ratio = features.get('body_length_ratio', 1)
                        disappear = features.get('disappear_ratio', 0) * 100
                        print(f"  [{CLASS_NAMES[label]:8}] {folder.name}: "
                              f"{len(df)}->{len(df_down)}帧, "
                              f"max_acc={max_acc:.2f}, h_chg={h_chg:.2f}, "
                              f"bl_ratio={bl_ratio:.2f}, disappear={disappear:.1f}%")
        
        print(f"  {category_name}: {count} 个样本")
    
    return all_features, all_labels


def train_classifier():
    """训练V6 RF分类器"""
    
    print("=" * 70)
    print(f"训练 RF 分类器 V6 (去除绝对位置, 增强动态特征)")
    print("=" * 70)
    
    features_list, labels = load_training_data()
    
    if len(features_list) == 0:
        print("\n[错误] 没有找到训练数据！")
        print(f"请检查目录: {TRAINING_DATA_DIR}")
        return
    
    df = pd.DataFrame(features_list)
    X = df.values
    y = np.array(labels)
    feature_names = list(df.columns)
    
    print(f"\n{'='*70}")
    print(f"数据统计")
    print(f"{'='*70}")
    print(f"总样本数: {len(y)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == i).sum()}")
    print(f"特征数: {len(feature_names)}")
    print(f"特征列表: {feature_names}")
    
    # 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集: {len(y_train)}, 测试集: {len(y_test)}")
    
    # 训练
    print("\n训练 Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\n{'='*70}")
    print("测试集结果")
    print(f"{'='*70}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=[f"Actual_{n}" for n in CLASS_NAMES],
                         columns=[f"Pred_{n}" for n in CLASS_NAMES])
    print(cm_df)
    
    # 特征重要性
    print(f"\n{'='*70}")
    print("Feature Importance (全部):")
    print(f"{'='*70}")
    importance = list(zip(feature_names, clf.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, imp) in enumerate(importance, 1):
        bar = '█' * int(imp * 100)
        print(f"  {rank:2d}. {name:30}: {imp:.4f}  {bar}")
    
    # V5 vs V6 对比
    print(f"\n{'='*70}")
    print("V6 vs V5 特征改动对比:")
    print(f"{'='*70}")
    removed = ['mean_height', 'min_height', 'mean_body_length', 'min_body_length', 
               'max_body_length', 'total_frames', 'valid_frames']
    added = ['max_acceleration', 'mean_abs_acceleration', 'acceleration_std',
             'height_ratio', 'body_length_ratio', 'x_displacement', 'x_velocity_std',
             'shoulder_hip_ratio_change', 'pose_angle_std', 'velocity_spike_count']
    print(f"  去除 ({len(removed)}个): {removed}")
    print(f"  新增 ({len(added)}个): {added}")
    
    # 保存
    model_path = BASE_DIR / "fall_classifier_v6.joblib"
    model_data = {
        'classifier': clf,
        'feature_names': feature_names,
        'class_names': CLASS_NAMES,
        'n_classes': 3,
        'target_fps': TARGET_FPS,
        'window_size': WINDOW_SIZE,
        'window_seconds': WINDOW_SECONDS,
        'version': 'v6',
        'trained_at': datetime.now().isoformat(),
        'changes': {
            'removed_features': removed,
            'added_features': added,
            'reason': '去除绝对位置特征降低误报，增加加速度等动态特征'
        }
    }
    
    joblib.dump(model_data, model_path)
    print(f"\n模型已保存: {model_path}")
    print(f"  - 版本: V6")
    print(f"  - 目标帧率: {TARGET_FPS} FPS")
    print(f"  - 窗口大小: {WINDOW_SIZE} 帧 ({WINDOW_SECONDS}秒)")
    print(f"  - 特征数: {len(feature_names)}")
    
    return clf, feature_names


if __name__ == "__main__":
    train_classifier()
