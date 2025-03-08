from argparse import ArgumentParser
import os
import math
import librosa
import yaml
import numpy as np
import pickle
from preprocess.tacotron.hyperparams import Hyperparams as hy
from preprocess.tacotron import utils

# file_path=r'D:\document\pycharmproject\mouth_voice\preprocess\phoneme_video_model_file.txt'
# audio_file_prefix=r'D:\document\paper\personpaper\audio-visual_consistance\data\timit_audio2'
# mean_std_path=r'D:\document\pycharmproject\mouth_voice\output'
test_set=['mwbt0','msjs1','mrgg0','mpgl0',
          'fram1','fjwb0','fjem0','felc0']

def get_audio_mel(data_dir,phoneme_info_path,out_dir):
    audio_signal=[]
    loaded_audio_list=[]
    audio_feature_map = {}
    train_audio_feature_list = []
    with open(phoneme_info_path,'r') as phoneme_info_file:
        #fadg0_sa1 SH 21 24 860.544 980.27 119.72
        for i, line in enumerate(phoneme_info_file):
            print(line)
            phoneme_info = line.strip().split()
            figure_id=phoneme_info[0].split("_")[0]
            word_id=phoneme_info[0].split("_")[1]
            phoneme_label=phoneme_info[1]
            # fadg0_sa1_SH_i
            phoneme_unit_label=phoneme_info[0]+"_"+phoneme_label+'_'+str(i)
            # The corresponding video frame
            start_frame = int(phoneme_info[2])
            end_frame = int(phoneme_info[3])
            # interval start and end time
            start_time = float(phoneme_info[4])
            end_time = float(phoneme_info[5])
            # audio tag
            audio_label=figure_id+'_'+word_id+'.wav'
            if audio_label not in loaded_audio_list:
                audio_file=os.path.join(data_dir,figure_id,word_id+'.wav')
                audio_signal, _ = librosa.load(audio_file, hy.sr)
                loaded_audio_list.append(audio_label)

            # clip segment with matching strategy
            phoneme_unit_mel_features=[]
            for index in range(start_frame, end_frame + 1):
                if index == start_frame:
                    start_point = math.floor(start_time * hy.sr * 1.0/1000 )
                    end_point = start_point + int(hy.frame_length * hy.sr * 1.0 )
                elif index == end_frame:
                    end_point = math.floor(end_time * hy.sr * 1.0 / 1000.0)
                    start_point = end_point - int(hy.frame_length * hy.sr * hy.sr * 1.0 )
                else:
                    start_point = math.floor(index*hy.frame_length * hy.sr * 1.0 )
                    end_point=start_point+int(hy.frame_length * hy.sr * 1.0/1000 )
                clip_signal = audio_signal[start_point:end_point]
                mel, _ = utils.get_spectrogramsfromsignal(clip_signal)
                # n_mels 512
                assert len(mel[0]) == hy.n_mels
                phoneme_unit_mel_features.append(mel[0])
                # calcute for mean and std
                if figure_id not in test_set:
                    train_audio_feature_list.append(mel[0])
            audio_feature_map[phoneme_unit_label]=phoneme_unit_mel_features

    train_audio_features = np.concatenate(train_audio_feature_list)
    print('len(train_audio_feature_list)', len(train_audio_feature_list))
    mean = np.mean(train_audio_features, axis=0)
    std = np.std(train_audio_features, axis=0)
    attr = {'mean': mean, 'std': std}
    with open(os.path.join(out_dir, 'audio_attr.pkl'), 'wb') as f:
        pickle.dump(attr, f)

    normalized_audio_feature_map = {}
    for key, val in audio_feature_map.items():
        processed_value_list=[(i - mean) / std for i in val]
        normalized_audio_feature_map[key]=processed_value_list
    # audio_feature.pkl
    audio_feature = {'audio_feature': normalized_audio_feature_map}
    with open(os.path.join(out_dir, 'audio_feature.pkl'), 'wb') as f:
        pickle.dump(audio_feature, f)

    # 添加音素统计
    unique_phonemes = set()
    with open(phoneme_info_path, 'r') as phoneme_info_file:
        for line in phoneme_info_file:
            phoneme_info = line.strip().split()
            if len(phoneme_info) >= 2:
                phoneme_label = phoneme_info[1]
                unique_phonemes.add(phoneme_label)
    
    num_phonemes = len(unique_phonemes)
    print(f"\n检测到的音素类别数量: {num_phonemes}")
    print("音素列表:", sorted(list(unique_phonemes)))
    
    # 更新配置文件
    config_path = os.path.join(os.path.dirname(out_dir), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'phoneme' not in config:
            config['phoneme'] = {}
        config['phoneme']['num_phonemes'] = num_phonemes
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, allow_unicode=True)
        print(f"配置文件已更新：{config_path}")

if  __name__  == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    args = parser.parse_args()

    args.config = r'D:\document\pycharmproject\AVCDetection\preprocess\preprocess_config.yaml'

    # load config file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['audio']['data_dir']
    phoneme_info_path = config['audio']['phoneme_info_path']
    out_dir = config['audio']['out_dir']


    get_audio_mel(data_dir,phoneme_info_path,out_dir)

