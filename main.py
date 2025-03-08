'''
Author: lzm lzmpt@qq.com
Date: 2025-03-07 21:04:11
LastEditors: lzm lzmpt@qq.com
LastEditTime: 2025-03-08 00:24:38
FilePath: \AVCDetection\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from argparse import ArgumentParser
import torch
from solver import Solver
import yaml
import os
import logging

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def setup_directories(config):
    """创建必要的目录"""
    dirs = [
        config['store_model_dir'],
        config['logger']['logger_dir'],
        os.path.dirname(config['evaluate_path'])
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    # 参数解析
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='运行模式：训练或测试')
    args = parser.parse_args()

    # 设置日志
    setup_logging()
    logging.info(f"Starting in {args.mode} mode")

    # 加载配置
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建必要的目录
    setup_directories(config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 创建求解器
    solver = Solver(config)
    
    # 训练或测试
    if args.mode == 'train':
        logging.info("Starting training...")
        try:
            solver.train()
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
    else:
        logging.info("Starting inference...")
        solver.infer()

if __name__ == '__main__':
    main()