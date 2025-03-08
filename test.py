import os
import torch
import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
import json
import cv2
from datetime import datetime
from tqdm import tqdm
from crossmodel import CrossModel
from utils import cc
from preprocess import extract_video_features, extract_audio_features, load_visual_model
import pickle
from sklearn.metrics import roc_auc_score

def load_model(model_path):
    """
    加载模型
    Args:
        model_path (str): 模型文件路径
    Returns:
        model: 加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path)
    
    # 从检查点中获取配置
    config = checkpoint.get('config', {
        'model': {
            'output_dim': 64,
            'conv_channels': 32,
            'dropout': 0.1
        }
    })
    
    # 初始化模型
    model = CrossModel(config)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = cc(model)
    model.eval()
    
    return model

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
        print("警告: 无法直接获取视频帧率，尝试手动计算...")
        start = 0
        frames = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for _ in range(100):
            ret = cap.grab()
            if not ret:
                break
            frames += 1
            
        if frames > 0:
            end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end > 0:
                fps = frames / end
            else:
                fps = 25.0
        else:
            fps = 25.0
            
        print(f"使用估算/默认帧率: {fps:.2f} fps")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if frame_count <= 0:
        print("警告: 无法直接获取总帧数，尝试手动计算...")
        frame_count = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            frame_count += 1
        print(f"计算得到总帧数: {frame_count}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
    
    if width <= 0 or height <= 0:
        raise ValueError(f"无法获取有效的视频分辨率: {width}x{height}")
    if duration <= 0:
        raise ValueError(f"无法计算有效的视频时长")
    
    info = {
        'width': width,
        'height': height,
        'fps': float(f"{fps:.2f}"),
        'frame_count': frame_count,
        'duration': float(f"{duration:.2f}")
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
        for i, (start, end) in enumerate(suspicious_segments):
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

def process_video(video_path, model, visual_model, transform, window_size=32):
    """
    处理单个视频并返回时序一致性得分
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
        v_window = torch.from_numpy(visual_feature[i:i+window_size]).unsqueeze(0).unsqueeze(0)
        a_window = torch.from_numpy(audio_feature[i:i+window_size]).unsqueeze(0).unsqueeze(0)
        
        v_window = cc(v_window)
        a_window = cc(a_window)
        
        with torch.no_grad():
            audio_emb, video_emb = model(a_window, v_window)
            distance = torch.pow((video_emb - audio_emb), 2).sum(dim=1).item()
            
        score = 1.0 / (1.0 + distance)
        scores.append(score)
    
    overall_score = np.mean(scores)
    return overall_score, scores

def process_features(features_path, model):
    """
    处理单个特征文件
    """
    try:
        with open(features_path, 'rb') as f:
            data = pickle.load(f)
        
        # 确保特征维度正确
        visual_feature = data['visual_feature']
        audio_feature = data['audio_feature']
        
        # 打印原始特征的统计信息
        print(f"\n处理文件: {features_path}")
        print(f"视觉特征统计: min={visual_feature.min():.4f}, max={visual_feature.max():.4f}, mean={visual_feature.mean():.4f}, std={visual_feature.std():.4f}")
        print(f"音频特征统计: min={audio_feature.min():.4f}, max={audio_feature.max():.4f}, mean={audio_feature.mean():.4f}, std={audio_feature.std():.4f}")
        
        # 如果特征是2D的，添加batch和channel维度
        if visual_feature.ndim == 2:
            visual_feature = torch.from_numpy(visual_feature).unsqueeze(0).unsqueeze(0)
        else:
            visual_feature = torch.from_numpy(visual_feature)
            
        if audio_feature.ndim == 2:
            audio_feature = torch.from_numpy(audio_feature).unsqueeze(0).unsqueeze(0)
        else:
            audio_feature = torch.from_numpy(audio_feature)
        
        # 移动到GPU
        visual_feature = cc(visual_feature)
        audio_feature = cc(audio_feature)
        
        with torch.no_grad():
            audio_emb, video_emb = model(audio_feature, visual_feature)
            
            # 打印嵌入向量的统计信息
            print(f"视觉嵌入统计: min={video_emb.min().item():.4f}, max={video_emb.max().item():.4f}, mean={video_emb.mean().item():.4f}, std={video_emb.std().item():.4f}")
            print(f"音频嵌入统计: min={audio_emb.min().item():.4f}, max={audio_emb.max().item():.4f}, mean={audio_emb.mean().item():.4f}, std={audio_emb.std().item():.4f}")
            
            # 确保嵌入向量是2D的 [batch_size, feature_dim]
            if audio_emb.dim() > 2:
                audio_emb = audio_emb.mean(dim=1)  # 对时间维度取平均
            if video_emb.dim() > 2:
                video_emb = video_emb.mean(dim=1)  # 对时间维度取平均
            
            # 标准化嵌入向量
            video_emb = torch.nn.functional.normalize(video_emb, p=2, dim=-1)
            audio_emb = torch.nn.functional.normalize(audio_emb, p=2, dim=-1)
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(video_emb, audio_emb)
            # 将相似度转换到[0,1]范围
            score = (similarity + 1) / 2
            
            print(f"最终得分: {score.item():.4f}")
                
    except Exception as e:
        print(f"\n处理特征时出错 {features_path}: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        score = 0.0
    
    return score

def evaluate_dataset(test_dir, model):
    """
    评估测试集
    """
    results = {
        'scores': [],
        'labels': [],
        'filenames': []
    }
    
    real_dir = os.path.join(test_dir, 'real')
    if os.path.exists(real_dir):
        for file in tqdm(os.listdir(real_dir), desc='处理真实视频'):
            if file.endswith('.pkl'):
                score = process_features(os.path.join(real_dir, file), model)
                # 确保分数是CPU上的标量值
                if isinstance(score, torch.Tensor):
                    score = score.cpu().item()
                results['scores'].append(score)
                results['labels'].append(1)
                results['filenames'].append(os.path.join('real', file))
    
    fake_dir = os.path.join(test_dir, 'fake')
    if os.path.exists(fake_dir):
        for file in tqdm(os.listdir(fake_dir), desc='处理伪造视频'):
            if file.endswith('.pkl'):
                score = process_features(os.path.join(fake_dir, file), model)
                # 确保分数是CPU上的标量值
                if isinstance(score, torch.Tensor):
                    score = score.cpu().item()
                results['scores'].append(score)
                results['labels'].append(0)
                results['filenames'].append(os.path.join('fake', file))
    
    # 确保所有分数都是标量值
    results['scores'] = np.array(results['scores'])
    results['labels'] = np.array(results['labels'])
    
    if len(set(results['labels'])) > 1:
        auc_score = roc_auc_score(results['labels'], results['scores'])
    else:
        auc_score = None
    
    return auc_score, results

def visualize_results(scores, output_dir, video_info=None, time_axis=None):
    """
    可视化结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    x = time_axis if time_axis is not None else np.arange(len(scores))
    
    plt.plot(x, scores, 'b-', label='Consistency Score')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.fill_between(x, scores, 0.5, where=(np.array(scores) < 0.5),
                    color='red', alpha=0.3, label='Suspicious Segments')
    
    plt.title('Audio-Visual Consistency Analysis')
    plt.xlabel('Time (seconds)' if video_info else 'Time Window')
    plt.ylabel('Consistency Score')
    plt.legend()
    plt.grid(True)
    
    plt.axes([0.7, 0.7, 0.2, 0.2])
    plt.hist(scores, bins=20, color='blue', alpha=0.7)
    plt.title('Score Distribution')
    
    plt.savefig(os.path.join(output_dir, 'consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_single_video_report(video_path, overall_score, time_scores, video_info, 
                           suspicious_segments, saved_frames, output_dir):
    """
    保存单个视频的分析报告
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
            'confidence': float(f"{abs(overall_score - 0.5) * 2:.4f}"),
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON报告
    with open(os.path.join(output_dir, 'analysis_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成文本报告
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

    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt_report)

def save_batch_evaluation_report(results, output_dir):
    """
    保存批量评估报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_results = []
    for filename, score, label in zip(results['filenames'], results['scores'], results['labels']):
        detailed_results.append({
            'filename': filename,
            'score': float(f"{score:.4f}"),
            'label': int(label),
            'prediction': '真实' if score > 0.5 else '伪造'
        })
    
    real_scores = [s for s, l in zip(results['scores'], results['labels']) if l == 1]
    fake_scores = [s for s, l in zip(results['scores'], results['labels']) if l == 0]
    
    statistics = {
        'total_samples': len(results['scores']),
        'real_samples': len(real_scores),
        'fake_samples': len(fake_scores),
        'real_avg_score': float(f"{np.mean(real_scores):.4f}") if real_scores else None,
        'fake_avg_score': float(f"{np.mean(fake_scores):.4f}") if fake_scores else None,
        'auc_score': float(f"{roc_auc_score(results['labels'], results['scores']):.4f}") if len(set(results['labels'])) > 1 else None
    }
    
    output = {
        'statistics': statistics,
        'detailed_results': detailed_results
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    txt_report = f"""音视频一致性检测评估报告
{'='*50}
评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

统计信息:
- 总样本数: {statistics['total_samples']}
- 真实样本数: {statistics['real_samples']}
- 伪造样本数: {statistics['fake_samples']}
- 真实样本平均得分: {statistics['real_avg_score']:.4f}
- 伪造样本平均得分: {statistics['fake_avg_score']:.4f}
- AUC得分: {statistics['auc_score']:.4f}

详细结果:
"""
    
    for result in detailed_results:
        txt_report += f"- {result['filename']}: 得分={result['score']:.4f}, 标签={'真实' if result['label'] == 1 else '伪造'}, 预测={result['prediction']}\n"
    
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt_report)

def main():
    parser = argparse.ArgumentParser(description='音视频一致性检测工具')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], required=True,
                      help='运行模式: single(单个视频) 或 batch(批量评估)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型文件路径')
    parser.add_argument('--input', type=str, required=True,
                      help='输入路径 (单个视频文件或特征目录)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='结果输出目录')
    parser.add_argument('--visualize', action='store_true',
                      help='是否生成可视化结果')
    args = parser.parse_args()
    
    # 加载模型
    print("\n加载模型...")
    model = load_model(args.model_path)
    
    if args.mode == 'single':
        # 单个视频测试模式
        print(f"\n处理视频: {args.input}")
        
        # 加载视觉特征提取模型
        visual_model, transform = load_visual_model('resnet50')
        
        # 获取视频信息
        video_info = get_video_info(args.input)
        
        # 处理视频
        overall_score, time_scores = process_video(args.input, model, visual_model, transform)
        
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
                    if segment[0] - merged_segments[-1][1] < 1.0:
                        merged_segments[-1] = (merged_segments[-1][0], segment[1])
                    else:
                        merged_segments.append(segment)
                suspicious_segments = merged_segments
        
        # 保存关键帧
        saved_frames = save_key_frames(args.input, args.output_dir, suspicious_segments)
        
        # 可视化结果
        if args.visualize:
            time_axis = np.arange(len(time_scores)) * (video_info['duration'] / len(time_scores))
            visualize_results(time_scores, args.output_dir, video_info, time_axis)
        
        # 保存分析报告
        save_single_video_report(args.input, overall_score, time_scores, video_info,
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
        
    else:
        # 批量评估模式
        print("\n开始批量评估...")
        auc_score, results = evaluate_dataset(args.input, model)
        
        # 保存评估结果
        save_batch_evaluation_report(results, args.output_dir)
        
        # 可视化结果
        if args.visualize:
            visualize_results(results['scores'], args.output_dir)
        
        # 打印主要结果
        print(f"\n评估完成!")
        if auc_score is not None:
            print(f"AUC得分: {auc_score:.4f}")
    
    print(f"\n详细结果已保存至: {args.output_dir}")

if __name__ == '__main__':
    main()





