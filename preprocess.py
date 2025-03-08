import os
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import librosa
import tempfile
from pydub import AudioSegment
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_visual_model(model_name='resnet50'):
    """
    加载视觉特征提取模型
    Args:
        model_name (str): 模型名称，支持 'resnet50', 'resnet18', 'vgg16' 等
    Returns:
        model: 预训练模型
        transform: 预处理转换
    """
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.features), model.avgpool)
    else:
        raise ValueError(f"不支持的视觉模型: {model_name}")
    
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform

def extract_video_features(video_path, model, transform):
    """
    提取视频特征
    Args:
        video_path (str): 视频文件路径
        model: 预训练视觉模型
        transform: 预处理转换
    Returns:
        np.ndarray: 视频特征
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换BGR到RGB并转为PIL图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            # 应用预处理
            frame = transform(frame).unsqueeze(0)
            if torch.cuda.is_available():
                frame = frame.cuda()
            
            # 提取特征
            with torch.no_grad():
                feat = model(frame)
            frames.append(feat.cpu().numpy())
            
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"视频没有帧: {video_path}")
        
        return np.concatenate(frames)
        
    except Exception as e:
        print(f"提取视频特征时出错 {video_path}: {str(e)}")
        raise

def extract_audio_features(video_path, sr=16000, n_mfcc=40):
    """
    提取音频特征
    Args:
        video_path (str): 视频文件路径
        sr (int): 采样率
        n_mfcc (int): MFCC特征数量
    Returns:
        np.ndarray: 音频特征
    """
    try:
        # 使用pydub加载视频的音频
        audio = AudioSegment.from_file(video_path)
        
        # 创建临时文件来保存音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # 导出为WAV格式
            audio.export(temp_audio.name, format='wav')
            
            # 使用librosa加载音频
            y, sr = librosa.load(temp_audio.name, sr=sr)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
        # 删除临时文件
        os.unlink(temp_audio.name)
        
        return mfcc.T  # 转置以匹配 [T, F] 格式
        
    except Exception as e:
        print(f"提取音频特征时出错 {video_path}: {str(e)}")
        raise

def load_metadata(metadata_path):
    """
    加载元数据文件
    Args:
        metadata_path (str): 元数据文件路径
    Returns:
        dict: 视频标签字典，键为视频ID，值为标签（1为真实，0为伪造）
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        if not isinstance(metadata_list, list):
            raise ValueError("元数据文件格式错误：应为列表格式")
        
        labels = {}
        for item in metadata_list:
            # 检查必要的字段
            if 'file' not in item:
                continue
                
            # 从文件路径中提取视频ID
            video_file = os.path.basename(item['file'])
            video_id = os.path.splitext(video_file)[0]
            
            # 根据n_fakes确定真假：n_fakes=0为真实，n_fakes>0为伪造
            labels[video_id] = 1 if item.get('n_fakes', 0) == 0 else 0
        
        if labels:
            print(f"成功加载 {len(labels)} 个视频的标签信息")
        else:
            raise ValueError("未能从元数据中提取到任何标签信息")
            
        return labels
        
    except json.JSONDecodeError:
        raise ValueError("元数据文件格式错误：不是有效的JSON格式")
    except Exception as e:
        raise ValueError(f"加载元数据文件时出错: {str(e)}")

def process_video(video_path, output_path, label, visual_model, transform):
    """
    处理单个视频
    Args:
        video_path (str): 输入视频路径
        output_path (str): 输出特征文件路径
        label (int): 标签（1为真实，0为伪造）
        visual_model: 视觉特征提取模型
        transform: 预处理转换
    Returns:
        bool: 处理是否成功
    """
    try:
        # 提取特征
        visual_feature = extract_video_features(video_path, visual_model, transform)
        audio_feature = extract_audio_features(video_path)
        
        # 保存特征
        data = {
            'visual_feature': visual_feature,
            'audio_feature': audio_feature,
            'label': label,
            'filename': os.path.basename(video_path)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
            
        return True
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {str(e)}")
        return False

def process_dataset(input_dir, output_dir, split, visual_model, transform, metadata):
    """处理数据集的一个划分"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频文件列表
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    print(f"\nProcessing {split} set: {len(video_files)} videos")
    success_count = 0
    failed_files = []
    
    for video_file in tqdm(video_files):
        video_id = os.path.splitext(video_file)[0]  # 去除.mp4后缀
        input_path = os.path.join(input_dir, video_file)
        
        # 根据metadata确定输出目录
        label = metadata.get(video_id, 0)  # 如果在metadata中找不到，默认为伪造
        output_subdir = 'real' if label == 1 else 'fake'
        output_path = os.path.join(output_dir, output_subdir, video_file.replace('.mp4', '.pkl'))
        
        try:
            if process_video(input_path, output_path, label, visual_model, transform):
                success_count += 1
            else:
                failed_files.append(video_file)
        except Exception as e:
            print(f"处理 {video_file} 时出错: {str(e)}")
            failed_files.append(video_file)
            continue
            
    print(f"\nSuccessfully processed {success_count}/{len(video_files)} videos")
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"- {f}")
            
    return success_count, failed_files

def main():
    parser = argparse.ArgumentParser(description='预处理LAV-DF数据集')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--visual_model', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet18', 'vgg16'], 
                       help='视觉特征提取器')
    parser.add_argument('--audio_feature', type=str, default='mfcc',
                       choices=['mfcc', 'mel', 'stft'],
                       help='音频特征类型')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, help='工作线程数')
    args = parser.parse_args()
    
    # 加载视觉模型
    print(f"\n加载视觉模型 {args.visual_model}...")
    visual_model, transform = load_visual_model(args.visual_model)
    
    # 加载metadata
    metadata_path = os.path.join(args.input_dir, 'metadata.json')
    print(f"\n加载元数据文件: {metadata_path}")
    metadata = load_metadata(metadata_path)
    
    # 处理每个数据集划分
    splits = ['train', 'test', 'dev']
    stats = {}
    all_failed_files = {}
    
    for split in splits:
        print(f"\nProcessing {split} set...")
        input_split_dir = os.path.join(args.input_dir, split)
        output_split_dir = os.path.join(args.output_dir, split)
        
        if not os.path.exists(input_split_dir):
            print(f"Warning: {input_split_dir} does not exist, skipping...")
            continue
            
        success_count, failed_files = process_dataset(
            input_split_dir, 
            output_split_dir, 
            split,
            visual_model,
            transform,
            metadata
        )
        
        stats[split] = {
            'total': len([f for f in os.listdir(input_split_dir) if f.endswith('.mp4')]),
            'success': success_count,
            'failed': len(failed_files)
        }
        all_failed_files[split] = failed_files
    
    # 保存处理统计信息
    stats_info = {
        'config': {
            'visual_model': args.visual_model,
            'audio_feature': args.audio_feature,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        },
        'stats': stats,
        'failed_files': all_failed_files
    }
    
    with open(os.path.join(args.output_dir, 'preprocessing_stats.json'), 'w') as f:
        json.dump(stats_info, f, indent=2)
        
    print("\nPreprocessing completed!")
    print("\nStatistics:")
    for split, stat in stats.items():
        print(f"{split}:")
        print(f"  Total: {stat['total']}")
        print(f"  Success: {stat['success']}")
        print(f"  Failed: {stat['failed']}")

if __name__ == '__main__':
    main() 