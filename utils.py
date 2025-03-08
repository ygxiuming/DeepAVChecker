import torch 
import numpy as np
from tensorboardX import SummaryWriter
import pickle
import editdistance
import torch.nn as nn
import torch.nn.init as init
import json
import os
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import tqdm
import yaml
#import pyworld

def cc(net):
    """
    将模型移动到可用的设备（GPU/CPU）上
    Args:
        net: PyTorch模型或张量
    Returns:
        移动到指定设备的模型或张量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)

def load_data_zero_audio(path,key):
    with open(path,'rb') as f:
        data=pickle.load(f)
        data_list=data[key]
    data_zeroaudio_list=[]
    for item in data_list:
        zero_audio_feature=np.zeros(np.array(item[2]).shape).tolist()
        data_zeroaudio_list.append([item[0],item[1],zero_audio_feature])
    return data_zeroaudio_list

def load_data(path,key):
    with open(path,'rb') as f:
        data=pickle.load(f)
        data_list=data[key]
    return data_list


class Logger(object):
    """
    日志记录器类
    用于记录训练过程中的各种指标，支持TensorBoard可视化
    """
    def __init__(self, log_dir, tag='default'):
        """
        初始化日志记录器
        Args:
            log_dir (str): 日志保存目录
            tag (str): 实验标签
        """
        self.writer = SummaryWriter(os.path.join(log_dir, tag))
        self.step = 0
        self.mode = ''
        self.tag = tag
        self.scalar_metrics = {}

    def set_step(self, step, mode='train'):
        """
        设置当前步数和模式
        Args:
            step (int): 当前步数
            mode (str): 当前模式（train/val/test）
        """
        self.mode = mode
        self.step = step
        # 每次设置步数时，记录之前累积的指标
        self._write_scalars()

    def update(self, metrics):
        """
        更新指标
        Args:
            metrics (dict): 包含各项指标的字典
        """
        # 确保所有值都是标量
        for key, value in metrics.items():
            value = self.get_scalar_value(value)
            if key not in self.scalar_metrics:
                self.scalar_metrics[key] = {}
            if self.mode not in self.scalar_metrics[key]:
                self.scalar_metrics[key][self.mode] = []
            self.scalar_metrics[key][self.mode].append(value)
            
            # 立即记录到TensorBoard
            self.scalar_summary(f"{key}/{self.mode}", value, self.step)

    def _write_scalars(self):
        """
        将累积的指标写入TensorBoard
        """
        for key, modes in self.scalar_metrics.items():
            for mode, values in modes.items():
                if values:  # 如果有值
                    # 计算平均值
                    avg_value = sum(values) / len(values)
                    # 记录到TensorBoard
                    self.scalar_summary(f"{key}/{mode}", avg_value, self.step)
            # 清空累积的值
            for mode in modes:
                modes[mode] = []

    def get_scalar_value(self, value):
        """
        获取标量值
        Args:
            value: 输入值（可能是标量、张量或numpy数组）
        Returns:
            float: 转换后的标量值
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item()
        return value

    def scalar_summary(self, tag, value, step):
        """
        记录单个标量值
        Args:
            tag (str): 指标名称
            value: 指标值
            step (int): 当前步数
        """
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, main_tag, tag_scalar_dict, step):
        """
        记录多个标量值
        Args:
            main_tag (str): 主标签
            tag_scalar_dict (dict): 标签-数值字典
            step (int): 当前步数
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def text_summary(self, tag, value, step):
        """
        记录文本信息
        Args:
            tag (str): 文本标签
            value (str): 文本内容
            step (int): 当前步数
        """
        self.writer.add_text(tag, str(value), step)

    def audio_summary(self, tag, value, step, sr):
        """记录音频数据"""
        self.writer.add_audio(tag, value, step, sample_rate=sr)

    def close(self):
        """
        关闭日志记录器
        """
        self._write_scalars()  # 确保所有数据都被写入
        self.writer.close()

class LavDFDataset(Dataset):
    """
    LAV-DF数据集加载器
    用于加载和预处理音视频特征数据
    """
    def __init__(self, data_path, mode='train'):
        """
        初始化数据集
        Args:
            data_path (str): 数据集路径
            mode (str): 数据集模式（train/val/test）
        """
        self.data_path = data_path
        self.mode = mode
        self.data = self._load_data()

    def _load_data(self):
        """
        加载数据集
        Returns:
            list: 包含所有样本信息的列表
        """
        data = []
        # 加载真实样本
        real_path = os.path.join(self.data_path, 'real')
        if os.path.exists(real_path):
            for file in os.listdir(real_path):
                if file.endswith('.pkl'):
                    data.append({
                        'path': os.path.join(real_path, file),
                        'label': 1,
                        'filename': file
                    })
        
        # 加载伪造样本
        fake_path = os.path.join(self.data_path, 'fake')
        if os.path.exists(fake_path):
            for file in os.listdir(fake_path):
                if file.endswith('.pkl'):
                    data.append({
                        'path': os.path.join(fake_path, file),
                        'label': 0,
                        'filename': file
                    })
        return data

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx (int): 样本索引
        Returns:
            dict: 包含特征和标签的字典
        """
        item = self.data[idx]
        with open(item['path'], 'rb') as f:
            data = pickle.load(f)
        
        return {
            'visual_feature': data['visual_feature'],
            'audio_feature': data['audio_feature'],
            'label': item['label'],
            'filename': item['filename']
        }

def custom_collate_fn(batch):
    """
    自定义的批处理函数，处理不同长度的序列
    Args:
        batch: 批次数据列表
    Returns:
        批处理后的数据字典
    """
    # 获取批次中的最大长度
    max_visual_len = max([b['visual_feature'].shape[0] for b in batch])
    max_audio_len = max([b['audio_feature'].shape[0] for b in batch])
    
    # 初始化张量
    batch_size = len(batch)
    
    # 获取特征维度
    visual_shape = batch[0]['visual_feature'].shape
    if len(visual_shape) == 4:  # [T, C, H, W]
        visual_features = torch.zeros(batch_size, max_visual_len, visual_shape[1], visual_shape[2], visual_shape[3])
    else:  # [T, F]
        visual_features = torch.zeros(batch_size, max_visual_len, visual_shape[1])
    
    audio_dim = batch[0]['audio_feature'].shape[1]
    audio_features = torch.zeros(batch_size, max_audio_len, audio_dim)
    labels = torch.zeros(batch_size)
    filenames = []
    
    # 填充数据
    for i, item in enumerate(batch):
        # 获取当前序列的长度
        curr_visual_len = item['visual_feature'].shape[0]
        curr_audio_len = item['audio_feature'].shape[0]
        
        # 复制数据
        visual_feat = torch.from_numpy(item['visual_feature'])
        if len(visual_feat.shape) == 4:  # [T, C, H, W]
            visual_features[i, :curr_visual_len] = visual_feat
        else:  # [T, F]
            visual_features[i, :curr_visual_len] = visual_feat
            
        audio_features[i, :curr_audio_len] = torch.from_numpy(item['audio_feature'])
        labels[i] = item['label']
        filenames.append(item['filename'])
    
    return {
        'visual_feature': visual_features,
        'audio_feature': audio_features,
        'label': labels,
        'filename': filenames
    }

def get_dataloader(config, mode):
    """
    获取数据加载器
    Args:
        config (dict): 配置字典
        mode (str): 数据集模式（train/val/test）
    Returns:
        DataLoader: PyTorch数据加载器
    """
    if mode == 'train':
        path = config['train_set_path']
    elif mode == 'test':
        path = config['test_set_path']
    else:  # dev
        path = config['dev_set_path']
    
    dataset = LavDFDataset(path, mode)
    
    return DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn  # 使用自定义的批处理函数
    )

def generate_phoneme_sequence(audio_feature):
    """
    从音频特征生成音素序列
    Args:
        audio_feature (np.ndarray): 音频特征矩阵
    Returns:
        list: 音素序列
    """
    # 这里是一个示例实现，实际应用中需要根据具体的音素识别模型来生成序列
    return ["p1", "p2", "p3"]  # 示例返回值

def save_phoneme_sequences(dataset: LavDFDataset, output_dir: str):
    """
    为数据集中的所有样本生成音素序列文件
    Args:
        dataset: LavDF数据集实例
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating phoneme sequences for {len(dataset)} samples...")
    for idx in tqdm.tqdm(range(len(dataset))):
        sample = dataset[idx]
        audio_feature = sample['audio_feature'].numpy()
        filename = sample['filename']
        
        # 生成音素序列
        phonemes = generate_phoneme_sequence(audio_feature)
        
        # 保存音素序列到文本文件
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_phonemes.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(phonemes))

def process_all_phoneme_sequences(config: Dict):
    """
    处理所有数据集的音素序列
    Args:
        config: 配置字典
    """
    splits = ['train', 'test', 'dev']
    
    for split in splits:
        print(f"\nProcessing {split} set...")
        try:
            # 使用正确的数据路径
            data_path = os.path.normpath(config[f'{split}_set_path'])
            if not os.path.exists(data_path):
                print(f"警告: 数据目录不存在: {data_path}")
                continue
                
            dataset = LavDFDataset(
                root_dir=data_path,
                split=split
            )
            
            # 确保输出目录使用正确的路径分隔符
            output_dir = os.path.normpath(os.path.join(config['phoneme']['output_dir'], split))
            os.makedirs(output_dir, exist_ok=True)
            
            save_phoneme_sequences(dataset, output_dir)
            print(f"音素序列已保存到 {output_dir}")
        except Exception as e:
            print(f"处理 {split} 集时出错: {str(e)}")
            continue

class PhonemeProcessor:
    """
    音素处理器类
    用于处理音频特征并生成音素序列
    """
    def __init__(self, config: Dict):
        """
        初始化音素处理器
        Args:
            config (Dict): 配置字典
        """
        self.config = config
        # 自动检测并更新音素数量
        self.num_phonemes = self._detect_phoneme_count()
        self.phoneme_map = self._load_phoneme_map()
    
    def _detect_phoneme_count(self) -> int:
        """
        自动检测数据集中的音素类别数量
        Returns:
            int: 实际的音素类别数量
        """
        unique_phonemes = set()
        
        # 获取音素信息文件路径
        phoneme_info_path = self.config.get('phoneme_info_path', 'preprocess/phoneme_info.txt')
        
        try:
            with open(phoneme_info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 假设每行格式为: "fadg0_sa1 SH 21 24 860.544 980.27 119.72"
                    # 其中第二个字段是音素标签
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        phoneme = parts[1]
                        unique_phonemes.add(phoneme)
            
            num_phonemes = len(unique_phonemes)
            print(f"检测到 {num_phonemes} 个不同的音素类别")
            
            # 更新配置文件中的音素数量
            self.config['phoneme']['num_phonemes'] = num_phonemes
            
            # 保存更新后的配置
            config_path = os.path.join(os.path.dirname(self.config.get('store_model_dir', 'output/models')), 'config.yaml')
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True)
            
            print(f"配置文件已更新：{config_path}")
            return num_phonemes
            
        except FileNotFoundError:
            print(f"警告：未找到音素信息文件 {phoneme_info_path}")
            print("使用默认音素数量：40")
            return 40
        except Exception as e:
            print(f"警告：音素数量检测失败 - {str(e)}")
            print("使用默认音素数量：40")
            return 40
    
    def _load_phoneme_map(self) -> Dict[int, str]:
        """
        加载音素映射表
        Returns:
            Dict[int, str]: 音素ID到音素符号的映射
        """
        try:
            # 获取音素信息文件路径
            phoneme_info_path = self.config.get('phoneme_info_path', 'preprocess/phoneme_info.txt')
            
            # 读取所有唯一的音素
            unique_phonemes = set()
            with open(phoneme_info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        phoneme = parts[1]
                        unique_phonemes.add(phoneme)
            
            # 创建音素到ID的映射
            phoneme_list = sorted(list(unique_phonemes))
            return {i: phoneme for i, phoneme in enumerate(phoneme_list)}
            
        except Exception as e:
            print(f"警告：加载音素映射失败 - {str(e)}")
            print("使用默认音素映射")
            return {i: f"p{i}" for i in range(self.num_phonemes)}
    
    def get_phoneme_list(self) -> List[str]:
        """
        获取所有音素的列表
        Returns:
            List[str]: 音素列表
        """
        return list(self.phoneme_map.values())
    
    def process_audio_feature(self, audio_feature: np.ndarray) -> List[str]:
        """
        处理单个音频特征，返回音素序列
        Args:
            audio_feature (np.ndarray): 音频特征矩阵
        Returns:
            List[str]: 音素序列
        """
        return generate_phoneme_sequence(audio_feature)
    
    def save_sequence(self, sequence: List[str], filepath: str):
        """
        保存音素序列到文件
        Args:
            sequence (List[str]): 音素序列
            filepath (str): 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sequence))



