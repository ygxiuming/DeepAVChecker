import argparse
import yaml
from utils import process_all_phoneme_sequences, PhonemeProcessor

def main():
    parser = argparse.ArgumentParser(description='生成音素序列文件')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建音素处理器
    processor = PhonemeProcessor(config)
    
    # 处理所有数据集的音素序列
    process_all_phoneme_sequences(config)

if __name__ == '__main__':
    main() 