# 音视频一致性检测系统

本项目实现了一个基于深度学习的音视频一致性检测系统，可用于检测视频中的音频和视频是否匹配，从而识别可能的换脸或换音频等伪造内容。

## 主要特性

- 音视频跨模态学习
- 基于边际损失的对比学习
- 自动混合精度训练
- 梯度累积以提高内存效率
- 学习率预热机制
- 数值稳定性优化
- 全面的训练监控

## 环境要求

```bash
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
tqdm>=4.50.0
PyYAML>=5.4.1
scikit-learn>=0.24.0
tensorboard>=2.4.0
```

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/AVCDetection.git
cd AVCDetection
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装 ffmpeg（用于音频处理）：
```bash
# Windows (使用 chocolatey)
choco install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS (使用 homebrew)
brew install ffmpeg
```

## 数据准备

### 1. 原始数据集结构
您可以从以下链接下载 LAV-DF 数据集：
- OneDrive: [下载链接](https://1drv.ms/f/s!AjQHQ9AZ_T7JgUwZwXvArbJTWqYx?e=dxhXs7)
- Google Drive: [下载链接](https://drive.google.com/drive/folders/1_eUpuZXJf7oOGxMV1yPHxFvzXHO34Hou?usp=sharing)
- HuggingFace: [下载链接](https://huggingface.co/datasets/ControlNet/LAV-DF)

下载后，请按以下结构组织数据集：
```
LAV-DF/                    # 根目录名必须为 LAV-DF
├── metadata.json          # 元数据文件，包含视频标签信息
├── train/                 # 训练集
│   ├── 000001.mp4        # 视频文件
│   ├── 000002.mp4
│   └── ...
├── test/                  # 测试集
│   ├── 000101.mp4
│   ├── 000102.mp4
│   └── ...
└── dev/                   # 验证集
    ├── 000201.mp4
    ├── 000202.mp4
    └── ...
```

### 2. 元数据文件格式
metadata.json 的格式如下：
```json
[
    {
        "file": "train/000001.mp4",
        "n_fakes": 0,                    # 0表示真实视频，>0表示伪造视频
        "fake_periods": [],              # 伪造片段的时间区间
        "timestamps": [...],             # 音频时间戳信息
        "duration": 5.44,               # 视频时长
        "transcript": "...",            # 音频文本
        "original": null,               # 原始视频（如果是伪造视频）
        "modify_video": false,          # 是否修改了视频
        "modify_audio": false,          # 是否修改了音频
        "split": "train",              # 数据集划分
        "video_frames": 134,           # 视频帧数
        "audio_channels": 1,           # 音频通道数
        "audio_frames": 84992          # 音频帧数
    }
]
```

### 3. 特征提取

#### 基本用法
使用 `preprocess.py` 脚本进行特征提取：

```bash
python preprocess.py \
    --input_dir LAV-DF \
    --output_dir LAV-DF_processed \
    --visual_model resnet50 \
    --audio_feature mfcc \
    --batch_size 32 \
    --num_workers 4
```

#### 参数说明
- `--input_dir`：原始视频数据集路径（必须包含 metadata.json）
- `--output_dir`：处理后特征保存路径
- `--visual_model`：视觉特征提取器选择
  - resnet50（默认，推荐）
  - resnet18（更快，更省内存）
  - vgg16（可选）
- `--audio_feature`：音频特征类型
  - mfcc（默认，推荐）
  - mel（梅尔频谱图）
  - stft（短时傅里叶变换）
- `--batch_size`：批处理大小（默认32，如果显存不足请减小）
- `--num_workers`：数据加载线程数（建议设置为 CPU 核心数）

#### 输出结构
处理完成后，会在输出目录生成以下内容：
```
LAV-DF_processed/
├── preprocessing_stats.json  # 处理统计信息
├── train/
│   ├── real/               # 真实视频特征
│   │   └── *.pkl
│   └── fake/              # 伪造视频特征
│       └── *.pkl
├── test/
│   ├── real/
│   └── fake/
└── dev/
    ├── real/
    └── fake/
```

每个 .pkl 文件包含：
```python
{
    'visual_feature': np.ndarray,  # 视觉特征，形状 [T, 2048]
    'audio_feature': np.ndarray,   # 音频特征，形状 [T, 40]
    'label': int,                  # 1: 真实, 0: 伪造
    'filename': str                # 原始视频文件名
}
```

## 配置说明

修改 `config.yaml` 以设置训练参数：

```yaml
# 数据路径
train_set_path: "path/to/train/features"
dev_set_path: "path/to/val/features"
test_set_path: "path/to/test/features"
store_model_dir: "output/models"
evaluate_path: "output/evaluation"

# 训练设置
train:
  iters: 100  # 总训练轮数
  batch_size: 32
  save_steps: 1
  loaded_model: false
  loaded_model_num: 0

# 模型设置
model:
  # ... (具体的模型参数)

# 优化器设置
optimizer:
  lr: 0.001  # 初始学习率（将被缩放为0.1倍）
  beta1: 0.9
  beta2: 0.999
  amsgrad: true
  weight_decay: 0.0001
  grad_norm: 5.0

# 损失函数设置
loss:
  margin: 1.0

# 日志设置
logger:
  logger_dir: "output/logs"
  tag: "training"
```

## 模型训练

### 基本训练
启动基本训练：
```bash
python train.py --config config.yaml
```

### 从检查点恢复训练
```bash
python train.py \
    --config config.yaml \
    --resume \
    --checkpoint path/to/checkpoint.pth
```

### 使用预训练模型
```bash
python train.py \
    --config config.yaml \
    --pretrained path/to/pretrained.pth
```

### 训练参数说明
- `--config`：配置文件路径
- `--resume`：是否从检查点恢复训练
- `--checkpoint`：检查点文件路径
- `--pretrained`：预训练模型路径
- `--seed`：随机种子（默认42）
- `--device`：训练设备（默认 'cuda'）

## 模型评估

### 使用说明
该工具支持两种模式：单个视频测试和批量评估。

#### 命令行参数
- `--mode`: 运行模式（必需）
  - `single`: 单个视频测试模式
  - `batch`: 批量评估模式
- `--model_path`: 模型文件路径（必需）
- `--input`: 输入路径（必需）
  - 单个视频模式：视频文件路径
  - 批量评估模式：预处理特征目录路径
- `--output_dir`: 结果输出目录（默认：results）
- `--visualize`: 是否生成可视化结果（可选）

#### 单个视频测试
```bash
# 基本测试
python test.py --mode single \
    --model_path output/models/best.pth \
    --input path/to/video.mp4 \
    --output_dir results

# 带可视化的测试
python test.py --mode single \
    --model_path output/models/best.pth \
    --input path/to/video.mp4 \
    --output_dir results \
    --visualize
```

输出内容：
- 分析报告（JSON和文本格式）
- 一致性得分可视化（使用--visualize时）
- 可疑片段的关键帧
- 视频基本信息
- 一致性得分和判定结果
- 可疑时间段标注（如果有）

#### 批量评估
```bash
# 基本评估
python test.py --mode batch \
    --model_path output/models/best.pth \
    --input LAV-DF_processed/test \
    --output_dir results

# 带可视化的评估
python test.py --mode batch \
    --model_path output/models/best.pth \
    --input LAV-DF_processed/test \
    --output_dir results \
    --visualize
```

输出内容：
- 评估报告（JSON和文本格式）
- 得分分布可视化（使用--visualize时）
- 总体统计信息（AUC等指标）
- 每个样本的详细结果

### 评估指标
评估结果包含：
- Clip-level AUC：片段级别的ROC曲线下面积
- 真实/伪造样本的平均得分
- 样本数量统计
- 每个样本的预测结果

## 单个视频测试

### 使用预训练模型测试视频
```bash
python test_single.py \
    --video_path path/to/video.mp4 \
    --model_path path/to/model.pth \
    --output_dir results \
    --visualize
```

### 测试参数说明
- `--video_path`：待测试视频路径
- `--model_path`：模型文件路径
- `--output_dir`：结果输出目录
- `--visualize`：是否生成可视化结果
- `--threshold`：判定阈值（默认0.5）

### 输出说明
- 视频真实性得分（0-1之间）
- 可疑时间段标注（如果检测为伪造）
- 可视化结果（如果启用）
  - 音频波形图
  - 视频关键帧
  - 一致性得分曲线

## 模型架构

模型使用跨模态架构来学习音频和视频特征之间的一致性：
- 音频编码器：处理音频特征
- 视频编码器：处理视频特征
- 对比学习损失函数
- 特征向量L2归一化

## 训练监控

训练过程提供详细的监控指标：
- 损失值
- 梯度范数
- 学习率
- NaN/Inf值检测
- AUC指标（片段级和视频级）
- 等错误率（EER）
- 10%虚警率下的漏检率（FRR@10）

进度条显示示例：
```
Epoch [1/100]: 45%|█████████████▌         | 17/38 [01:35<01:57, 5.61s/it, Loss=0.4268, Grad=34.75, LR=1.00e-4, NaN=0]
```

## 常见问题解决

1. 显存不足（OOM）问题：
   - 增加梯度累积步数
   - 减小批次大小
   - 启用混合精度训练

2. 训练不稳定：
   - 检查学习率设置
   - 调整边际损失值
   - 监控梯度范数
   - 检查是否出现NaN/Inf值

3. 性能不佳：
   - 验证数据预处理
   - 调整模型架构
   - 尝试不同的超参数
   - 确保数据集类别平衡


