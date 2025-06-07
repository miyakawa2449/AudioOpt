import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from typing import List, Tuple

class VoiceDataset(Dataset):
    """音声データセット"""
    
    def __init__(self, audio_files: List[str], text_files: List[str], 
                 preprocessor, text_processor):
        self.audio_files = audio_files
        self.text_files = text_files
        self.preprocessor = preprocessor
        self.text_processor = text_processor
        
        # メルスペクトログラム変換器を設定
        self.mel_transform = T.MelSpectrogram(
            sample_rate=preprocessor.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0
        )
        
        # デシベル変換器
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # データをロード
        self.load_data()
    
    def load_data(self):
        """データをロード"""
        self.audio_features = []
        self.text_sequences = []
        
        print("Loading dataset...")
        successful_loads = 0
        
        for i, (audio_file, text_file) in enumerate(zip(self.audio_files, self.text_files)):
            print(f"Loading {i+1}/{len(self.audio_files)}: {audio_file}")
            
            # 音声データの処理
            audio = self.preprocessor.process_audio(audio_file)
            if len(audio) > 0:
                try:
                    # メルスペクトログラムに変換
                    audio_tensor = torch.from_numpy(audio).float()
                    
                    # メルスペクトログラム計算
                    mel_spec = self.mel_transform(audio_tensor)
                    
                    # デシベルスケールに変換
                    mel_spec_db = self.amplitude_to_db(mel_spec)
                    
                    # 転置して (time, mel_bins) の形にする
                    mel_spec_db = mel_spec_db.transpose(0, 1)
                    
                    # 有効な長さをチェック
                    if mel_spec_db.shape[0] > 10:  # 最低10フレーム必要
                        self.audio_features.append(mel_spec_db.numpy())
                        
                        # テキストデータの処理
                        try:
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text = f.read()
                            
                            text_seq = self.text_processor.text_to_sequence(text)
                            if len(text_seq) > 0:
                                self.text_sequences.append(text_seq)
                                successful_loads += 1
                                print(f"  ✓ Success: mel_shape={mel_spec_db.shape}, text_len={len(text_seq)}")
                            else:
                                print(f"  ❌ Empty text sequence")
                                self.audio_features.pop()  # 対応する音声データも削除
                            
                        except Exception as e:
                            print(f"  ❌ Text loading error: {e}")
                            self.audio_features.pop()  # 対応する音声データも削除
                    else:
                        print(f"  ❌ Audio too short: {mel_spec_db.shape[0]} frames")
                        
                except Exception as e:
                    print(f"  ❌ Mel spectrogram creation failed: {e}")
            else:
                print(f"  ❌ Audio loading failed")
        
        print(f"\nDataset loading complete: {successful_loads}/{len(self.audio_files)} files successfully loaded")
        
        if successful_loads == 0:
            print("⚠️  No valid data found! Please check your audio files and text files.")
    
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_feature = torch.FloatTensor(self.audio_features[idx])
        text_sequence = torch.LongTensor(self.text_sequences[idx])
        
        return audio_feature, text_sequence
    
    def collate_fn(self, batch):
        """バッチ処理用のcollate関数"""
        audio_features, text_sequences = zip(*batch)
        
        # パディング処理
        audio_lengths = [len(audio) for audio in audio_features]
        text_lengths = [len(text) for text in text_sequences]
        
        max_audio_len = max(audio_lengths)
        max_text_len = max(text_lengths)
        
        # 音声特徴量のパディング
        padded_audio = torch.zeros(len(batch), max_audio_len, audio_features[0].size(1))
        for i, audio in enumerate(audio_features):
            padded_audio[i, :len(audio)] = audio
        
        # テキスト系列のパディング
        padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        for i, text in enumerate(text_sequences):
            padded_text[i, :len(text)] = text
        
        return (padded_audio, torch.LongTensor(audio_lengths),
                padded_text, torch.LongTensor(text_lengths))