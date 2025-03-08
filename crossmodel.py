import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np
from math import ceil
from functools import reduce
from torch.nn.utils import spectral_norm

def cc(net):
    """
    将模型移动到可用的设备（GPU/CPU）上
    Args:
        net: PyTorch模型
    Returns:
        移动到指定设备的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)


def pad_layer(inp, layer, pad_type='constant'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode=pad_type,
            value=0.0)  # 使用0进行填充
    out = layer(inp)
    return out

def conv_bank_cal(x, module_list, act, pad_type='constant'):
    """
    应用卷积bank进行特征提取
    Args:
        x (torch.Tensor): 输入张量 [B, C, L]
        module_list (nn.ModuleList): 卷积层列表
        act (nn.Module): 激活函数
        pad_type (str): 填充类型，默认为'constant'
    Returns:
        torch.Tensor: 拼接后的特征 [B, C', L]
    """
    outs = []
    for layer in module_list:
        # 计算需要的填充大小
        kernel_size = layer.kernel_size[0]
        pad_size = kernel_size - 1
        # 对输入进行填充，确保输出大小与输入相同
        pad = (pad_size // 2, (pad_size + 1) // 2)
        padded_x = F.pad(x, pad=pad, mode=pad_type, value=0.0)
        out = act(layer(padded_x))
        outs.append(out)
    # 确保所有张量具有相同的大小后再拼接
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    """
    获取激活函数
    Args:
        act (str): 激活函数名称
    Returns:
        nn.Module: 激活函数模块
    """
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            #nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            #nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
            #nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )
    def forward(self, x):
        return x + self.main(x)

#卷积填充https://blog.csdn.net/qq_26369907/article/details/88366147
#d = (d - kennel_size + 2 * padding) / stride + 1
class VideoEncoder(nn.Module):
    """
    视频特征编码器
    用于将视频特征编码为固定维度的向量表示
    """
    def __init__(self, output_dim=64):
        """
        初始化视频编码器
        Args:
            output_dim (int): 输出特征维度，默认64
        """
        super(VideoEncoder, self).__init__()
        self.output_dim = output_dim
        
        # 定义卷积bank，包含不同kernel size的卷积层
        self.conv_bank = nn.ModuleList([
            nn.Conv1d(1, 32, k) for k in range(1, 8)
        ])
        
        # 定义主干网络
        self.conv2 = nn.Conv1d(32*7 + 1, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        
        # 批归一化层
        self.batch_norm32 = nn.BatchNorm1d(32)
        self.batch_norm64 = nn.BatchNorm1d(64)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(64, output_dim)
        
        # 激活函数
        self.act = nn.ReLU()

    def forward(self, input):
        """
        前向传播
        Args:
            input (torch.Tensor): 输入视频特征 [B, T, F] 或 [B, 1, T, F]
        Returns:
            torch.Tensor: 编码后的特征向量 [B, output_dim]
        """
        # 处理输入维度
        if input.dim() == 4:
            B, _, T, F = input.shape
            input = input.squeeze(1)
        
        # 转换为卷积所需的格式
        input = input.transpose(1, 2)
        input = input.reshape(input.size(0), 1, -1)
        
        # 应用卷积bank
        out = conv_bank_cal(input, self.conv_bank, act=self.act)
        
        # 应用主干网络
        out = self.conv2(out)
        out = self.batch_norm32(out)
        out = self.act(out)
        out = self.pool(out)
        
        out = self.conv3(out)
        out = self.batch_norm64(out)
        out = self.act(out)
        out = self.pool(out)
        
        # 自适应池化得到固定维度
        out = self.adaptive_pool(out)
        out = out.squeeze(-1)
        
        # 全连接层得到最终特征
        out = self.fc(out)
        
        return out

class AudioEncoder(nn.Module):
    """
    音频特征编码器
    用于将音频特征编码为固定维度的向量表示
    """
    def __init__(self, output_dim=64):
        """
        初始化音频编码器
        Args:
            output_dim (int): 输出特征维度，默认64
        """
        super(AudioEncoder, self).__init__()
        self.output_dim = output_dim
        
        # 定义卷积bank
        self.conv_bank = nn.ModuleList([
            nn.Conv1d(1, 32, k) for k in range(1, 8)
        ])
        
        # 定义主干网络
        self.conv2 = nn.Conv1d(32*7 + 1, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        
        # 批归一化层
        self.batch_norm32 = nn.BatchNorm1d(32)
        self.batch_norm64 = nn.BatchNorm1d(64)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(64, output_dim)
        
        # 激活函数
        self.act = nn.ReLU()

    @torch.amp.autocast('cuda')
    def forward(self, input):
        """
        前向传播
        Args:
            input (torch.Tensor): 输入音频特征 [B, T, F] 或 [B, 1, T, F]
        Returns:
            torch.Tensor: 编码后的特征向量 [B, output_dim]
        """
        # 处理输入维度
        if input.dim() == 4:
            B, _, T, F = input.shape
            input = input.squeeze(1)
        
        # 转换为卷积所需的格式
        input = input.transpose(1, 2)
        input = input.reshape(input.size(0), 1, -1)
        
        # 应用卷积bank
        out = conv_bank_cal(input, self.conv_bank, act=self.act)
        
        # 应用主干网络
        out = self.conv2(out)
        out = self.batch_norm32(out)
        out = self.act(out)
        out = self.pool(out)
        
        out = self.conv3(out)
        out = self.batch_norm64(out)
        out = self.act(out)
        out = self.pool(out)
        
        # 自适应池化得到固定维度
        out = self.adaptive_pool(out)
        out = out.squeeze(-1)
        
        # 全连接层得到最终特征
        out = self.fc(out)
        
        return out

class CrossModel(nn.Module):
    """
    跨模态一致性检测模型
    用于检测视频和音频是否匹配
    """
    def __init__(self, config):
        """
        初始化跨模态模型
        Args:
            config (dict): 配置字典，包含模型参数
        """
        super(CrossModel, self).__init__()
        output_dim = 64  # 设置统一的输出维度
        
        # 初始化音频和视频编码器
        self.audio_encoder = AudioEncoder(output_dim=output_dim)
        self.video_encoder = VideoEncoder(output_dim=output_dim)

    @torch.amp.autocast('cuda')
    def forward(self, audio_data, video_data):
        """
        前向传播
        Args:
            audio_data (torch.Tensor): 音频特征
            video_data (torch.Tensor): 视频特征
        Returns:
            tuple: (音频特征向量, 视频特征向量)
        """
        audio_emb = self.audio_encoder(audio_data)
        video_emb = self.video_encoder(video_data)
        return audio_emb, video_emb

# Find total parameters and trainable parameters
# model=SpeakerEncoder()
# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')