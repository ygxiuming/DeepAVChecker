import os
import torch
import numpy as np
import argparse
import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import librosa
import cv2
from PIL import Image
import yaml

def extract_visual_features(video_path, model, transform, device):
    """
    提取视频的视觉特征
    """
    cap = cv2.VideoCapture(video_path)
    features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转换BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        
        # 应用预处理
        frame = transform(frame).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            feat = model(frame)
        features.append(feat.cpu().numpy())
    
    cap.release()
    return np.concatenate(features)

def extract_audio_features(video_path, sr=16000, n_mfcc=40):
    """
    提取视频的音频特征
    """
    # 加载音频
    y, _ = librosa.load(video_path, sr=sr)
    
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def process_video(video_path, visual_model, transform, device, output_dir, filename):
    """
    处理单个视频文件
    """
    try:
        # 提取特征
        visual_feature = extract_visual_features(video_path, visual_model, transform, device)
        audio_feature = extract_audio_features(video_path)
        
        # 保存特征
        output_path = os.path.join(output_dir, f"{filename}.pkl")
        features = {
            'visual_feature': visual_feature,
            'audio_feature': audio_feature,
            'filename': filename
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        
        return True
    except Exception as e:
        print(f"处理视频 {filename} 时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='提取视频的音视频特征')
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--visual_model', type=str, default='resnet50', help='视觉特征提取器')
    parser.add_argument('--audio_feature', type=str, default='mfcc', help='音频特征类型')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='工作线程数')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载视觉模型
    if args.visual_model == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的FC层
    else:
        raise ValueError(f"不支持的视觉模型: {args.visual_model}")
    
    model = model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 处理数据集
    for split in ['train', 'test', 'dev']:
        split_dir = os.path.join(args.data_root, split)
        if not os.path.exists(split_dir):
            continue
            
        print(f"\n处理 {split} 集...")
        
        # 处理真实视频
        real_dir = os.path.join(split_dir, 'real')
        if os.path.exists(real_dir):
            output_real_dir = os.path.join(args.output_dir, split, 'real')
            os.makedirs(output_real_dir, exist_ok=True)
            
            for video_name in tqdm.tqdm(os.listdir(real_dir)):
                if video_name.endswith('.mp4'):
                    video_path = os.path.join(real_dir, video_name)
                    process_video(video_path, model, transform, device, 
                                output_real_dir, os.path.splitext(video_name)[0])
        
        # 处理伪造视频
        fake_dir = os.path.join(split_dir, 'fake')
        if os.path.exists(fake_dir):
            output_fake_dir = os.path.join(args.output_dir, split, 'fake')
            os.makedirs(output_fake_dir, exist_ok=True)
            
            for video_name in tqdm.tqdm(os.listdir(fake_dir)):
                if video_name.endswith('.mp4'):
                    video_path = os.path.join(fake_dir, video_name)
                    process_video(video_path, model, transform, device, 
                                output_fake_dir, os.path.splitext(video_name)[0])

if __name__ == '__main__':
    main() 