import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import cv2
from datetime import datetime
from tqdm import tqdm
from crossmodel import CrossModel
from preprocess import load_visual_model, extract_video_features, extract_audio_features
from utils import cc

def load_model(model_path):
    """
    加载模型
    Args:
        model_path (str): 模型文件路径
    Returns:
        model: 加载的模型
    """
    # 初始化模型（使用默认配置）
    model = CrossModel({
        'model': {
            'output_dim': 64,
            'conv_channels': 32,
            'dropout': 0.1
        }
    })
    
    # 加载模型权重
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = cc(model)
    model.eval()
    
    return model

def process_video(video_path, model, visual_model, transform, window_size=32):
    """
    处理单个视频并返回时序一致性得分
    Args:
        video_path (str): 视频文件路径
        model: 音视频一致性检测模型
        visual_model: 视觉特征提取模型
        transform: 预处理转换
        window_size (int): 滑动窗口大小
    Returns:
        tuple: (总体得分, 时序得分列表)
    """
    # 提取特征
    visual_feature = extract_video_features(video_path, visual_model, transform)
    audio_feature = extract_audio_features(video_path)
    
    # 确保特征长度匹配
    min_len = min(len(visual_feature), len(audio_feature))
    visual_feature = visual_feature[:min_len]
    audio_feature = audio_feature[:min_len]
    
    # 初始化结果列表
    scores = []
    
    # 使用滑动窗口计算时序得分
    for i in range(0, min_len - window_size + 1, window_size // 2):
        # 提取窗口数据
        v_window = torch.from_numpy(visual_feature[i:i+window_size]).unsqueeze(0).unsqueeze(0)
        a_window = torch.from_numpy(audio_feature[i:i+window_size]).unsqueeze(0).unsqueeze(0)
        
        # 移动到GPU
        v_window = cc(v_window)
        a_window = cc(a_window)
        
        # 计算特征嵌入和距离
        with torch.no_grad():
            audio_emb, video_emb = model(a_window, v_window)
            distance = torch.pow((video_emb - audio_emb), 2).sum(dim=1).item()
            
        # 将距离转换为得分（0-1之间）
        score = 1.0 / (1.0 + distance)
        scores.append(score)
    
    # 计算总体得分（使用平均值）
    overall_score = np.mean(scores)
    
    return overall_score, scores

def get_video_info(video_path):
    """
    获取视频基本信息
    Args:
        video_path (str): 视频文件路径
    Returns:
        dict: 包含视频信息的字典
    Raises:
        ValueError: 如果视频文件无法打开或信息无法获取
    """
    if not os.path.exists(video_path):
        raise ValueError(f"视频文件不存在: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 处理异常情况
    if fps <= 0:
        # 如果无法获取fps，尝试手动计算
        print("警告: 无法直接获取视频帧率，尝试手动计算...")
        start = 0
        frames = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 读取前100帧来计算fps
        for _ in range(100):
            ret = cap.grab()  # 只获取帧，不解码（更快）
            if not ret:
                break
            frames += 1
            
        if frames > 0:
            end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 转换为秒
            if end > 0:
                fps = frames / end
            else:
                fps = 25.0  # 使用默认帧率
        else:
            fps = 25.0  # 使用默认帧率
            
        print(f"使用估算/默认帧率: {fps:.2f} fps")
        
        # 重置视频位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 如果无法获取总帧数，尝试手动计算
    if frame_count <= 0:
        print("警告: 无法直接获取总帧数，尝试手动计算...")
        frame_count = 0
        while True:
            ret = cap.grab()  # 只获取帧，不解码（更快）
            if not ret:
                break
            frame_count += 1
        print(f"计算得到总帧数: {frame_count}")
        
        # 重置视频位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 计算视频时长
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
    
    # 验证获取的信息是否合理
    if width <= 0 or height <= 0:
        raise ValueError(f"无法获取有效的视频分辨率: {width}x{height}")
    if duration <= 0:
        raise ValueError(f"无法计算有效的视频时长")
    
    info = {
        'width': width,
        'height': height,
        'fps': float(f"{fps:.2f}"),  # 保留两位小数
        'frame_count': frame_count,
        'duration': float(f"{duration:.2f}")  # 保留两位小数
    }
    
    cap.release()
    return info

def save_key_frames(video_path, output_dir, suspicious_segments=None, max_frames=5):
    """
    保存关键帧（可疑片段的开始和结束帧）
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_dir = os.path.join(output_dir, 'key_frames')
    os.makedirs(frames_dir, exist_ok=True)
    saved_frames = []

    if suspicious_segments:
        # 保存可疑片段的关键帧
        for i, (start, end) in enumerate(suspicious_segments):
            # 保存片段开始帧
            start_frame = int(start * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(frames_dir, f'suspicious_{i+1}_start_{start:.2f}s.jpg')
                cv2.imwrite(frame_path, frame)
                saved_frames.append({
                    'time': start,
                    'type': 'suspicious_start',
                    'segment': i+1,
                    'path': os.path.basename(frame_path)
                })

            # 保存片段结束帧
            end_frame = int(end * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(frames_dir, f'suspicious_{i+1}_end_{end:.2f}s.jpg')
                cv2.imwrite(frame_path, frame)
                saved_frames.append({
                    'time': end,
                    'type': 'suspicious_end',
                    'segment': i+1,
                    'path': os.path.basename(frame_path)
                })
    else:
        # 如果没有可疑片段，均匀保存几帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(min(max_frames, total_frames)):
            frame_idx = i * total_frames // min(max_frames, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                time = frame_idx / fps
                frame_path = os.path.join(frames_dir, f'frame_{time:.2f}s.jpg')
                cv2.imwrite(frame_path, frame)
                saved_frames.append({
                    'time': time,
                    'type': 'regular',
                    'path': os.path.basename(frame_path)
                })

    cap.release()
    return saved_frames

def visualize_results(video_path, scores, output_dir, video_info):
    """
    增强的可视化结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 一致性得分曲线
    plt.figure(figsize=(12, 6))
    x = np.arange(len(scores))
    time_axis = x * (video_info['duration'] / len(scores))
    
    plt.plot(time_axis, scores, 'b-', label='Consistency Score')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.fill_between(time_axis, scores, 0.5, where=(np.array(scores) < 0.5),
                    color='red', alpha=0.3, label='Suspicious Segments')
    
    plt.title('Audio-Visual Consistency Analysis')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Consistency Score')
    plt.legend()
    plt.grid(True)
    
    # 添加得分分布直方图
    plt.axes([0.7, 0.7, 0.2, 0.2])  # 在主图右上角添加小图
    plt.hist(scores, bins=20, color='blue', alpha=0.7)
    plt.title('Score Distribution')
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_analysis_report(video_path, overall_score, time_scores, video_info, 
                        suspicious_segments, saved_frames, output_dir):
    """
    保存详细的分析报告
    """
    report = {
        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_info': {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'resolution': f"{video_info['width']}x{video_info['height']}",
            'duration': f"{video_info['duration']:.2f}s",
            'frame_count': video_info['frame_count'],
            'fps': video_info['fps']
        },
        'analysis_results': {
            'overall_score': float(f"{overall_score:.4f}"),
            'verdict': '真实视频' if overall_score > 0.5 else '伪造视频',
            'confidence': float(f"{abs(overall_score - 0.5) * 2:.4f}"),  # 转换为0-1的置信度
            'score_statistics': {
                'min_score': float(f"{min(time_scores):.4f}"),
                'max_score': float(f"{max(time_scores):.4f}"),
                'mean_score': float(f"{np.mean(time_scores):.4f}"),
                'std_score': float(f"{np.std(time_scores):.4f}")
            }
        }
    }
    
    if suspicious_segments:
        report['suspicious_segments'] = [
            {
                'start_time': f"{start:.2f}s",
                'end_time': f"{end:.2f}s",
                'duration': f"{end-start:.2f}s",
                'average_score': float(f"{np.mean([s for i, s in enumerate(time_scores) if start <= i*(video_info['duration']/len(time_scores)) <= end]):.4f}")
            }
            for start, end in suspicious_segments
        ]
    
    if saved_frames:
        report['saved_frames'] = saved_frames
    
    # 保存JSON报告
    report_path = os.path.join(output_dir, 'analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成可读性更好的文本报告
    txt_report = f"""音视频一致性分析报告
{'='*50}
分析时间: {report['analysis_time']}

视频信息:
- 文件名: {report['video_info']['filename']}
- 分辨率: {report['video_info']['resolution']}
- 时长: {report['video_info']['duration']}
- 总帧数: {report['video_info']['frame_count']}
- 帧率: {report['video_info']['fps']}fps

分析结果:
- 总体一致性得分: {report['analysis_results']['overall_score']:.4f}
- 判定结果: {report['analysis_results']['verdict']}
- 判定置信度: {report['analysis_results']['confidence']:.4f}

得分统计:
- 最低得分: {report['analysis_results']['score_statistics']['min_score']:.4f}
- 最高得分: {report['analysis_results']['score_statistics']['max_score']:.4f}
- 平均得分: {report['analysis_results']['score_statistics']['mean_score']:.4f}
- 标准差: {report['analysis_results']['score_statistics']['std_score']:.4f}
"""

    if suspicious_segments:
        txt_report += "\n可疑时间段:\n"
        for i, seg in enumerate(report['suspicious_segments'], 1):
            txt_report += f"{i}. {seg['start_time']} - {seg['end_time']} (持续{seg['duration']}, 平均得分: {seg['average_score']:.4f})\n"
    
    if saved_frames:
        txt_report += "\n保存的关键帧:\n"
        for frame in saved_frames:
            if frame['type'].startswith('suspicious'):
                txt_report += f"- 可疑片段 {frame['segment']} {frame['type'].split('_')[-1]}: {frame['time']:.2f}s ({frame['path']})\n"
            else:
                txt_report += f"- 常规帧: {frame['time']:.2f}s ({frame['path']})\n"

    # 保存文本报告
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt_report)

def main():
    parser = argparse.ArgumentParser(description='测试单个视频的音视频一致性')
    parser.add_argument('--video_path', type=str, required=True, help='待测试的视频文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output_dir', type=str, default='results', help='结果输出目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    args = parser.parse_args()
    
    # 加载视觉特征提取模型
    visual_model, transform = load_visual_model('resnet50')
    
    # 加载音视频一致性检测模型
    model = load_model(args.model_path)
    
    print(f"\n正在处理视频: {args.video_path}")
    
    # 获取视频信息
    video_info = get_video_info(args.video_path)
    
    # 处理视频
    overall_score, time_scores = process_video(args.video_path, model, visual_model, transform)
    
    # 识别可疑时间段
    suspicious_segments = []
    if overall_score <= 0.5:
        window_size = 32
        for i, score in enumerate(time_scores):
            if score <= 0.5:
                start_time = i * (window_size // 2) / video_info['fps']
                end_time = (i * (window_size // 2) + window_size) / video_info['fps']
                suspicious_segments.append((start_time, end_time))
        
        # 合并相邻的可疑片段
        if suspicious_segments:
            merged_segments = [suspicious_segments[0]]
            for segment in suspicious_segments[1:]:
                if segment[0] - merged_segments[-1][1] < 1.0:  # 如果间隔小于1秒
                    merged_segments[-1] = (merged_segments[-1][0], segment[1])
                else:
                    merged_segments.append(segment)
            suspicious_segments = merged_segments
    
    # 保存关键帧
    saved_frames = save_key_frames(args.video_path, args.output_dir, suspicious_segments)
    
    # 可视化结果
    if args.visualize:
        visualize_results(args.video_path, time_scores, args.output_dir, video_info)
    
    # 保存分析报告
    save_analysis_report(args.video_path, overall_score, time_scores, video_info, 
                        suspicious_segments, saved_frames, args.output_dir)
    
    # 打印基本结果
    print(f"\n检测结果:")
    print(f"总体一致性得分: {overall_score:.4f}")
    print(f"判定结果: {'真实视频' if overall_score > 0.5 else '伪造视频'}")
    print(f"判定置信度: {abs(overall_score - 0.5) * 2:.4f}")
    
    if suspicious_segments:
        print("\n可疑时间段:")
        for start, end in suspicious_segments:
            print(f"- {start:.2f}s 到 {end:.2f}s")
    
    print(f"\n详细分析报告已保存至: {args.output_dir}")

if __name__ == '__main__':
    main() 