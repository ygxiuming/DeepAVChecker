from os.path import join, basename
import torch.nn as nn
import yaml
from crossmodel import CrossModel
from utils import *
from collections import defaultdict
from count_metrics import calculate_metrics
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

class Solver(object):
    """
    求解器类，负责模型的训练、评估和推理
    包含了模型训练的完整流程，包括数据加载、优化器配置、损失计算等
    """
    def __init__(self, config):
        """
        初始化求解器
        Args:
            config (dict): 配置字典，包含所有训练相关参数
        """
        self.config = config
        print("Initializing solver with config:")
        print(config)

        # 文件路径和配置
        self.train_set_path = self.config['train_set_path']
        self.test_set_path = self.config['test_set_path']
        self.dev_set_path = self.config['dev_set_path']
        self.store_model_dir = self.config['store_model_dir']
        self.evaluate_path = self.config['evaluate_path']
        
        # 训练配置
        self.save_steps = self.config['train']['save_steps']
        self.loaded_model_num = self.config['train']['loaded_model_num']
        self.loaded_model = self.config['train']['loaded_model']
        self.iters = self.config['train']['iters']
        self.batch_size = self.config['train']['batch_size']
        
        # 日志配置
        self.logger = Logger(self.config['logger']['logger_dir'])
        self.tag = self.config['logger']['tag']
        
        # 损失函数和优化器配置
        self.margin = self.config['loss']['margin']
        # 降低初始学习率
        self.lr = self.config['optimizer']['lr'] * 0.1  # 降低学习率
        self.beta1 = self.config['optimizer']['beta1']
        self.beta2 = self.config['optimizer']['beta2']
        self.amsgrad = self.config['optimizer']['amsgrad']
        self.weight_decay = self.config['optimizer']['weight_decay']
        self.grad_norm = self.config['optimizer']['grad_norm']

        # 获取数据加载器
        self.train_loader = get_dataloader(config, 'train')
        self.test_loader = get_dataloader(config, 'test')
        self.dev_loader = get_dataloader(config, 'dev')

        # 构建模型
        self.build_model()
        self.save_config()
        
        # 如果需要，加载预训练模型
        if self.loaded_model:
            self.load_model(self.loaded_model_num)
        else:
            self.loaded_model_num = 0
            
    def build_model(self):
        """
        构建模型和优化器
        包括初始化CrossModel和配置Adam优化器
        """
        # 创建模型和优化器
        self.model = cc(CrossModel(self.config))
        print(self.model)
        
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, 
            betas=(self.beta1, self.beta2),
            amsgrad=self.amsgrad, 
            weight_decay=self.weight_decay
        )
        print(self.opt)
        return

    def load_model(self, iteration):
        """
        加载预训练模型
        Args:
            iteration (int): 要加载的模型迭代次数
        """
        path = join(self.store_model_dir, 'model_' + f'{iteration}')
        print(f'Load model from {path}')
        self.model.load_state_dict(torch.load(f'{path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{path}.opt'))
        return

    def save_config(self):
        """
        保存配置文件
        将当前配置保存到模型目录下
        """
        with open(join(self.store_model_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        return

    def save_model(self, iteration, is_best=False):
        """
        保存模型
        Args:
            iteration (int): 当前迭代次数
            is_best (bool): 是否为最佳模型
        """
        # 准备要保存的数据
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'iteration': iteration,
            'config': self.config
        }
        
        # 总是保存最新的模型为 last.pth
        last_path = join(self.store_model_dir, 'last.pth')
        torch.save(checkpoint, last_path)
        print(f'Last model saved to {last_path}')
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = join(self.store_model_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best model saved to {best_path}')

    def cal_distance(self, video_embed, audio_embed):
        """
        计算视频和音频特征之间的欧氏距离
        Args:
            video_embed (torch.Tensor): 视频特征嵌入 [B, D]
            audio_embed (torch.Tensor): 音频特征嵌入 [B, D]
        Returns:
            torch.Tensor: 每个样本的距离 [B]
        """
        distances = torch.pow((video_embed - audio_embed), 2).sum(dim=1)
        return distances

    def contrastive_margin_loss(self, video_embed, audio_embed, labels, margin):
        """
        计算对比损失，增加数值稳定性
        """
        # 检查输入
        if torch.isnan(video_embed).any() or torch.isnan(audio_embed).any():
            print("Warning: NaN found in embeddings")
            return None
            
        if torch.isinf(video_embed).any() or torch.isinf(audio_embed).any():
            print("Warning: Inf found in embeddings")
            return None

        # 归一化嵌入向量，提高稳定性
        video_embed = torch.nn.functional.normalize(video_embed, p=2, dim=1)
        audio_embed = torch.nn.functional.normalize(audio_embed, p=2, dim=1)
        
        # 计算欧氏距离，添加数值稳定性
        diff = video_embed - audio_embed
        distances = torch.clamp(torch.pow(diff, 2).sum(dim=1), min=1e-7, max=1e7)
        
        # 将标签转换为浮点数并压缩维度
        labels = labels.float().squeeze()
        
        # 使用更稳定的损失计算方式
        positive_loss = labels * distances
        negative_loss = (1 - labels) * torch.clamp(margin - distances, min=0)
        
        # 分别检查正负样本的损失
        if torch.isnan(positive_loss).any() or torch.isnan(negative_loss).any():
            print("Warning: NaN found in loss components")
            return None
            
        if torch.isinf(positive_loss).any() or torch.isinf(negative_loss).any():
            print("Warning: Inf found in loss components")
            return None
        
        # 计算总损失并添加L2正则化
        loss = (positive_loss + negative_loss).mean()
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Final loss is NaN or Inf")
            return None
            
        return loss

    def ae_step(self,item,margin):
        video_data = cc(torch.Tensor(item[1]))
        video_data.transpose(1, 2)
        video_data.unsqueeze_(1)
        audio_data = cc(torch.Tensor(item[2]))
        audio_data.unsqueeze_(1)  # (B, 1, 512)
        if 'real' in item[0]:
            label = 1
        elif 'fake' in item[0]:
            label = 0
        else:
            assert 1==0

        audio_emb,video_emb = self.model(audio_data,video_data)
        self.opt.zero_grad()
        loss = self.contrastive_margin_loss(video_emb, audio_emb, label, margin)
        #print("label:"+item[0]+"  loss:"+str(loss))
        loss.backward(torch.ones_like(loss))
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                max_norm=self.grad_norm)
        self.opt.step()
        #use for auc
        distance = self.cal_distance(video_emb, audio_emb)

        meta = {'loss': loss.item(),
                'grad_norm': grad_norm,
                'distance': distance.item()}
        return meta

    def test_evaluation(self,item):
        video_data = cc(torch.Tensor(item[1]))
        video_data.transpose(1, 2)
        video_data.unsqueeze_(1)
        audio_data = cc(torch.Tensor(item[2]))
        audio_data.unsqueeze_(1)  # (B, 1, 512)
        audio_emb,video_emb = self.model(audio_data,video_data)
        #use for auc
        distance = self.cal_distance(video_emb, audio_emb)
        return distance.item()

    def train(self):
        """
        训练模型，增加稳定性和监控
        """
        start_iterations = self.loaded_model_num
        best_auc = 0.0
        print(f"\nStarting training from iteration {start_iterations}")
        
        # 初始化日志记录器
        logger = Logger(self.config['logger']['logger_dir'], self.config['logger']['tag'])
        
        # 检查数据加载器
        if not self.train_loader or len(self.train_loader) == 0:
            raise ValueError("训练数据加载器为空！请检查数据集路径和预处理步骤。")
            
        # 计算每个epoch的总批次数
        total_batches = len(self.train_loader)
        print(f"\nDataset info:")
        print(f"- Training batches per epoch: {total_batches}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Samples per epoch: {total_batches * self.batch_size}")
        
        # 使用迭代次数作为总epoch数
        total_epochs = self.iters
        current_epoch = start_iterations // total_batches
        
        # 设置梯度累积和学习率预热
        gradient_accumulation_steps = 4
        warmup_epochs = 5
        effective_batch_size = self.batch_size * gradient_accumulation_steps
        
        print(f"\nTraining settings:")
        print(f"- Total epochs: {total_epochs}")
        print(f"- Warmup epochs: {warmup_epochs}")
        print(f"- Initial learning rate: {self.lr}")
        print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"- Effective batch size: {effective_batch_size}")
        
        # 初始化训练所需的变量
        total_loss = 0
        total_grad_norm = 0
        nan_count = 0  # 记录NaN出现次数
        
        # 启用自动混合精度训练
        scaler = torch.amp.GradScaler()
        
        for epoch in range(current_epoch, total_epochs):
            # 学习率预热
            if epoch < warmup_epochs:
                lr_scale = min(1., float(epoch + 1) / warmup_epochs)
                for param_group in self.opt.param_groups:
                    param_group['lr'] = self.lr * lr_scale
            
            # 在每个epoch开始时初始化统计数据
            total_loss = 0
            total_grad_norm = 0
            batch_count = 0
            nan_count = 0
            real_clip_distance_map = defaultdict(lambda: [])
            fake_clip_distance_map = defaultdict(lambda: [])
            
            # 训练进度条
            train_bar = tqdm(self.train_loader, 
                            desc=f'Epoch [{epoch + 1}/{total_epochs}]',
                            ncols=150,  # 增加进度条宽度
                            bar_format='{l_bar}{bar:50}{r_bar}')  # 自定义进度条格式
            
            self.model.train()
            
            for batch_idx, batch in enumerate(train_bar):
                # 定期清理缓存
                if batch_idx % gradient_accumulation_steps == 0:
                    torch.cuda.empty_cache()
                
                try:
                    # 准备数据
                    video_data = cc(batch['visual_feature'].unsqueeze(1))
                    audio_data = cc(batch['audio_feature'].unsqueeze(1))
                    labels = cc(batch['label'])
                    
                    # 使用自动混合精度训练
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        # 前向传播
                        audio_emb, video_emb = self.model(audio_data, video_data)
                        
                        # 计算损失
                        loss = self.contrastive_margin_loss(video_emb, audio_emb, labels, self.margin)
                        if loss is None:
                            nan_count += 1
                            if nan_count > 10:  # 如果连续出现太多NaN，提前停止训练
                                print("\nToo many NaN values encountered. Stopping training.")
                                return
                            continue
                        
                        # 根据梯度累积步数缩放损失
                        loss = loss / gradient_accumulation_steps
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    
                    # 梯度累积
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        scaler.unscale_(self.opt)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_norm
                        )
                        
                        # 检查梯度
                        if not torch.isfinite(grad_norm):
                            print(f"\nWarning: Non-finite gradient norm at batch {batch_idx}")
                            self.opt.zero_grad()
                            continue
                        
                        # 优化器步进
                        scaler.step(self.opt)
                        scaler.update()
                        self.opt.zero_grad()
                        
                        total_grad_norm += grad_norm.item()
                        batch_count += 1
                    
                    total_loss += loss.item() * gradient_accumulation_steps
                    
                    # 更新进度条，优化显示格式
                    train_bar.set_postfix({
                        'Loss': f'{loss.item() * gradient_accumulation_steps:7.4f}',
                        'Grad': f'{grad_norm.item() if "grad_norm" in locals() else 0:6.2f}',
                        'LR': f'{self.opt.param_groups[0]["lr"]:.2e}',
                        'NaN': f'{nan_count:3d}'
                    })
                    
                    # 收集距离数据用于评估
                    with torch.no_grad():
                        distances = self.cal_distance(video_emb, audio_emb)
                        for i, (dist, label, fname) in enumerate(zip(distances.cpu(), labels.cpu(), batch['filename'])):
                            if label.item() == 1:
                                real_clip_distance_map[fname].append(dist.item())
                            else:
                                fake_clip_distance_map[fname].append(dist.item())
                
                except RuntimeError as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    continue
            
            # 在epoch结束时记录平均指标
            if batch_count > 0:  # 确保至少处理了一些批次
                avg_loss = total_loss / batch_count
                avg_grad_norm = total_grad_norm / batch_count
                
                logger.set_step(epoch, 'epoch')
                logger.update({
                    'train/epoch_loss': avg_loss,
                    'train/epoch_grad_norm': avg_grad_norm,
                    'train/learning_rate': self.opt.param_groups[0]['lr'],
                    'train/nan_count': nan_count
                })
                
                # 验证和保存模型
                self.model.eval()
                val_metrics = self.evaluate(self.dev_loader)
                
                # 更新验证指标
                logger.update({
                    'val/clip_auc': val_metrics['clip_auc'],
                    'val/video_auc': val_metrics['video_auc'],
                    'val/clip_eer': val_metrics['clip_eer'],
                    'val/video_eer': val_metrics['video_eer'],
                    'val/clip_frr_10': val_metrics['clip_frr_10'],
                    'val/video_frr_10': val_metrics['video_frr_10']
                })
                
                # 保存模型
                is_best = val_metrics['clip_auc'] > best_auc
                if is_best:
                    best_auc = val_metrics['clip_auc']
                    print(f'\nNew best model! Clip AUC: {best_auc:.4f}')
                self.save_model(epoch, is_best=is_best)
        
        # 关闭日志记录器
        logger.close()
        print('\nTraining completed!')

    def evaluate(self, dataloader, prefix='test'):
        """
        评估模型
        Args:
            dataloader (DataLoader): 用于评估的数据加载器
            prefix (str): 评估类型前缀，用于日志记录
        Returns:
            float: AUC值
        """
        self.model.eval()
        real_clip_distance_map = defaultdict(lambda: [])
        fake_clip_distance_map = defaultdict(lambda: [])
        
        # 评估进度条
        eval_bar = tqdm(dataloader, 
                       desc=f'Evaluating ({prefix})', 
                       ncols=150,  # 增加进度条宽度
                       bar_format='{l_bar}{bar:50}{r_bar}')  # 自定义进度条格式
        
        with torch.no_grad():
            for batch in eval_bar:
                # 准备数据
                video_data = cc(batch['visual_feature'].unsqueeze(1))
                audio_data = cc(batch['audio_feature'].unsqueeze(1))
                labels = batch['label']
                
                # 计算特征和距离
                audio_emb, video_emb = self.model(audio_data, video_data)
                distances = self.cal_distance(video_emb, audio_emb)
                
                # 收集评估结果
                for i, (dist, label, fname) in enumerate(zip(distances.cpu().detach(), labels.cpu(), batch['filename'])):
                    if label.item() == 1:
                        real_clip_distance_map[fname].append(dist.item())
                    else:
                        fake_clip_distance_map[fname].append(dist.item())
        
        # 计算评估指标
        auc = calculate_metrics(
            real_clip_distance_map,
            fake_clip_distance_map,
            prefix,
            self.loaded_model_num,
            self.evaluate_path
        )
        return auc

    def infer(self):
        """
        模型推理
        在测试集上进行推理并计算评估指标
        """
        # 测试集评估
        test_real_clip_distance_map = defaultdict(lambda: [])
        test_fake_clip_distance_map = defaultdict(lambda: [])
        
        self.model.eval()
        with torch.no_grad():
            # 推理进度条
            for batch in tqdm(self.test_loader, 
                            desc='Inferring', 
                            ncols=150,  # 增加进度条宽度
                            bar_format='{l_bar}{bar:50}{r_bar}'):  # 自定义进度条格式
                # 准备数据
                video_data = cc(batch['visual_feature'].unsqueeze(1))
                audio_data = cc(batch['audio_feature'].unsqueeze(1))
                labels = batch['label']
                
                # 计算特征和距离
                audio_emb, video_emb = self.model(audio_data, video_data)
                distances = self.cal_distance(video_emb, audio_emb)
                
                # 收集推理结果
                for i, (dist, label, fname) in enumerate(zip(distances.cpu().detach(), labels.cpu(), batch['filename'])):
                    if label.item() == 1:
                        test_real_clip_distance_map[fname].append(dist.item())
                    else:
                        test_fake_clip_distance_map[fname].append(dist.item())
        
        # 计算并打印评估指标
        test_clip_auc = calculate_metrics(
            test_real_clip_distance_map,
            test_fake_clip_distance_map,
            'test',
            0,
            self.evaluate_path
        )
        print(f'Test AUC: {test_clip_auc:.4f}')


