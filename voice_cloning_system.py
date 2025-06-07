import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
from typing import List, Tuple, Dict
import re
import shutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AudioPreprocessor:
    """音声前処理クラス（ノイズ除去など）"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """音声ファイルを読み込み"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return np.array([])
    
    def denoise_spectral_subtraction(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """スペクトル減算によるノイズ除去"""
        if len(audio) == 0:
            return audio
            
        # 短時間フーリエ変換
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # ノイズ推定（最初の0.5秒をノイズとして使用）
        noise_frames = max(1, int(0.5 * self.sample_rate / 512))
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # スペクトル減算
        alpha = 2.0  # オーバーサブトラクション係数
        beta = 0.01  # フロア係数
        
        enhanced_magnitude = magnitude - alpha * noise_profile
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # 位相を復元して逆変換
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
        
        return enhanced_audio
    
    def denoise_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """ウィーナーフィルタによるノイズ除去"""
        if len(audio) == 0:
            return audio
            
        # 簡易的なウィーナーフィルタ実装
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # ノイズ分散推定
        noise_frames = max(1, int(0.5 * self.sample_rate / 512))
        noise_var = np.var(magnitude[:, :noise_frames], axis=1, keepdims=True)
        noise_var = np.maximum(noise_var, 1e-10)  # ゼロ除算防止
        
        # ウィーナーフィルタ適用
        signal_var = np.var(magnitude, axis=1, keepdims=True)
        wiener_gain = signal_var / (signal_var + noise_var)
        
        enhanced_magnitude = magnitude * wiener_gain
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
        
        return enhanced_audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """音声の正規化"""
        if len(audio) == 0:
            return audio
            
        # RMS正規化
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.1
        
        # クリッピング防止
        audio = np.clip(audio, -1.0, 1.0)
        return audio
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """無音部分の除去"""
        if len(audio) == 0:
            return audio
            
        # 音声の強度を計算
        frame_length = 2048
        hop_length = 512
        
        # 短時間エネルギーを計算
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # しきい値以上のフレームを特定
        frames_above_threshold = energy > threshold
        
        # フレームを時間に変換
        times = librosa.frames_to_time(np.arange(len(energy)), sr=self.sample_rate, hop_length=hop_length)
        
        # 音声部分のみを抽出
        audio_segments = []
        for i, is_speech in enumerate(frames_above_threshold):
            if is_speech:
                start_sample = int(times[i] * self.sample_rate)
                end_sample = int((times[i] + hop_length/self.sample_rate) * self.sample_rate)
                end_sample = min(end_sample, len(audio))
                if start_sample < len(audio):
                    audio_segments.append(audio[start_sample:end_sample])
        
        if audio_segments:
            return np.concatenate(audio_segments)
        else:
            return audio
    
    def process_audio(self, file_path: str, output_path: str = None) -> np.ndarray:
        """音声の総合的な前処理"""
        print(f"Processing: {file_path}")
        
        # 音声読み込み
        audio = self.load_audio(file_path)
        if len(audio) == 0:
            print(f"Warning: Could not load audio from {file_path}")
            return audio
        
        # ノイズ除去
        audio = self.denoise_spectral_subtraction(audio)
        
        # 無音除去
        audio = self.remove_silence(audio)
        
        # 正規化
        audio = self.normalize_audio(audio)
        
        # 保存
        if output_path:
            try:
                sf.write(output_path, audio, self.sample_rate)
                print(f"Saved processed audio to: {output_path}")
            except Exception as e:
                print(f"Error saving audio to {output_path}: {e}")
        
        return audio
    
class TextProcessor:
    """テキスト前処理クラス"""
    
    def __init__(self):
        self.vocab = set()
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def clean_text(self, text: str) -> str:
        """テキストの清浄化"""
        # 改行文字を除去
        text = text.replace('\n', ' ').replace('\r', '')
        
        # 余分な空白を除去
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocab(self, texts: List[str]):
        """語彙を構築"""
        for text in texts:
            self.vocab.update(text.lower())
        
        # 特殊トークンを追加
        self.vocab.add('<PAD>')
        self.vocab.add('<SOS>')
        self.vocab.add('<EOS>')
        
        # インデックスマッピングを作成
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(self.vocab))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def text_to_sequence(self, text: str) -> List[int]:
        """テキストを数値序列に変換"""
        text = self.clean_text(text).lower()
        sequence = [self.char_to_idx.get(char, 0) for char in text]
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """数値序列をテキストに変換"""
        text = ''.join([self.idx_to_char.get(idx, '') for idx in sequence])
        return text

class VoiceDataset(Dataset):
    """音声データセット"""
    
    def __init__(self, audio_files: List[str], text_files: List[str], 
                 preprocessor: AudioPreprocessor, text_processor: TextProcessor):
        self.audio_files = audio_files
        self.text_files = text_files
        self.preprocessor = preprocessor
        self.text_processor = text_processor
        
        # データをロード
        self.load_data()
    
    def load_data(self):
        """データをロード"""
        self.audio_features = []
        self.text_sequences = []
        
        for audio_file, text_file in zip(self.audio_files, self.text_files):
            # 音声処理
            audio = self.preprocessor.process_audio(audio_file)
            
            # メル・スペクトログラム特徴量を抽出
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.preprocessor.sample_rate,
                n_mels=80,
                n_fft=2048,
                hop_length=512
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # テキスト処理
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text_seq = self.text_processor.text_to_sequence(text)
            
            self.audio_features.append(mel_spec)
            self.text_sequences.append(text_seq)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        return {
            'audio': torch.FloatTensor(self.audio_features[idx]),
            'text': torch.LongTensor(self.text_sequences[idx])
        }

class VoiceCloneModel(nn.Module):
    """音声クローニングモデル（簡易版）"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 256, mel_dim: int = 80):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim
        
        # テキストエンコーダ
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 音声デコーダ
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.mel_projection = nn.Linear(hidden_dim, mel_dim)
        
        # アテンションメカニズム
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
    
    def forward(self, text_input, audio_target=None):
        # テキストエンコーディング
        text_embedded = self.text_embedding(text_input)
        encoder_output, _ = self.text_encoder(text_embedded)
        
        if audio_target is not None:
            # 訓練時
            batch_size, seq_len, _ = audio_target.shape
            decoder_input = torch.zeros(batch_size, seq_len, self.hidden_dim * 2)
            
            # アテンション適用
            attended_output, _ = self.attention(decoder_input, encoder_output, encoder_output)
            
            # メル・スペクトログラム予測
            decoder_output, _ = self.decoder(attended_output)
            mel_output = self.mel_projection(decoder_output)
            
            return mel_output
        else:
            # 推論時
            batch_size = text_input.size(0)
            max_length = 1000  # 最大生成長
            
            outputs = []
            hidden = None
            
            for _ in range(max_length):
                if len(outputs) == 0:
                    decoder_input = torch.zeros(batch_size, 1, self.hidden_dim * 2)
                else:
                    decoder_input = outputs[-1].unsqueeze(1)
                
                # アテンション適用
                attended_output, _ = self.attention(decoder_input, encoder_output, encoder_output)
                
                # デコーダ出力
                decoder_output, hidden = self.decoder(attended_output, hidden)
                mel_output = self.mel_projection(decoder_output)
                
                outputs.append(mel_output.squeeze(1))
            
            return torch.stack(outputs, dim=1)

class VoiceCloner:
    """音声クローニングメインクラス"""
    
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = dataset_path
        self.audio_path = os.path.join(dataset_path, "audio_files")
        self.meta_path = os.path.join(dataset_path, "meta_files")
        
        # ディレクトリを作成
        os.makedirs(self.audio_path, exist_ok=True)
        os.makedirs(self.meta_path, exist_ok=True)
        
        # 各コンポーネントを初期化
        self.preprocessor = AudioPreprocessor()
        self.text_processor = TextProcessor()
        self.model = None
        
        # 処理済みデータ保存用
        self.processed_path = os.path.join(dataset_path, "processed")
        os.makedirs(self.processed_path, exist_ok=True)
        
        # GPU使用可能かチェック
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def collect_data_files(self) -> Tuple[List[str], List[str]]:
        """データファイルを収集"""
        audio_files = []
        text_files = []
        
        # audio_files フォルダ内のwavファイルを検索
        for i in range(1, 100):  # 最大99個まで対応
            audio_file = os.path.join(self.audio_path, f"audio_{i}.wav")
            meta_file = os.path.join(self.meta_path, f"meta_{i}.txt")
            
            if os.path.exists(audio_file) and os.path.exists(meta_file):
                audio_files.append(audio_file)
                text_files.append(meta_file)
            else:
                # 連続しない番号もチェック
                continue
        
        # 実際に存在するファイルを順番に確認
        if not audio_files:
            print("No audio-text pairs found. Checking all possible combinations...")
            for i in range(1, 100):
                audio_file = os.path.join(self.audio_path, f"audio_{i}.wav")
                meta_file = os.path.join(self.meta_path, f"meta_{i}.txt")
                
                if os.path.exists(audio_file) and os.path.exists(meta_file):
                    audio_files.append(audio_file)
                    text_files.append(meta_file)
        
        print(f"Found {len(audio_files)} audio-text pairs")
        if len(audio_files) == 0:
            print(f"Please ensure your files are in:")
            print(f"  Audio files: {self.audio_path}")
            print(f"  Text files: {self.meta_path}")
        
        return audio_files, text_files
    
    def preprocess_dataset(self):
        """データセット全体を前処理"""
        audio_files, text_files = self.collect_data_files()
        
        processed_audio_files = []
        texts = []
        
        for i, (audio_file, text_file) in enumerate(zip(audio_files, text_files)):
            # 音声前処理
            processed_audio_path = os.path.join(self.processed_path, f"processed_audio_{i+1}.wav")
            self.preprocessor.process_audio(audio_file, processed_audio_path)
            processed_audio_files.append(processed_audio_path)
            
            # テキスト読み込み
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            texts.append(self.text_processor.clean_text(text))
        
        # 語彙構築
        self.text_processor.build_vocab(texts)
        
        return processed_audio_files, text_files
    
    def create_dataset(self) -> VoiceDataset:
        """データセットを作成"""
        processed_audio_files, text_files = self.preprocess_dataset()
        
        dataset = VoiceDataset(
            processed_audio_files, 
            text_files, 
            self.preprocessor, 
            self.text_processor
        )
        
        return dataset
    
    def train_model(self, epochs: int = 100, batch_size: int = 4, learning_rate: float = 0.001):
        """モデルを訓練"""
        print("Creating dataset...")
        dataset = self.create_dataset()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        # モデル初期化
        vocab_size = len(self.text_processor.vocab)
        self.model = VoiceCloneModel(vocab_size)
        
        # 最適化
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                text_input = batch['text']
                audio_target = batch['audio'].transpose(1, 2)  # [batch, time, mel_dim]
                
                # フォワードパス
                predicted_mel = self.model(text_input, audio_target)
                
                # 損失計算
                loss = criterion(predicted_mel, audio_target)
                
                # バックプロパゲーション
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def collate_fn(self, batch):
        """バッチ処理用のコレート関数"""
        # 可変長データのパディング処理
        texts = [item['text'] for item in batch]
        audios = [item['audio'] for item in batch]
        
        # テキストのパディング
        max_text_len = max(len(text) for text in texts)
        padded_texts = []
        for text in texts:
            padded = torch.cat([text, torch.zeros(max_text_len - len(text), dtype=torch.long)])
            padded_texts.append(padded)
        
        # 音声のパディング
        max_audio_len = max(audio.shape[1] for audio in audios)
        padded_audios = []
        for audio in audios:
            padding = torch.zeros(audio.shape[0], max_audio_len - audio.shape[1])
            padded = torch.cat([audio, padding], dim=1)
            padded_audios.append(padded)
        
        return {
            'text': torch.stack(padded_texts),
            'audio': torch.stack(padded_audios)
        }
    
    def save_model(self, path: str = "voice_clone_model.pth"):
        """モデルを保存"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'text_processor': self.text_processor,
                'vocab_size': len(self.text_processor.vocab)
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str = "voice_clone_model.pth"):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.text_processor = checkpoint['text_processor']
        vocab_size = checkpoint['vocab_size']
        
        self.model = VoiceCloneModel(vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {path}")
    
    def synthesize_speech(self, text: str, output_path: str = "output.wav"):
        """テキストから音声を合成"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # テキストを数値序列に変換
        text_sequence = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence).unsqueeze(0)
        
        # 音声合成
        with torch.no_grad():
            mel_output = self.model(text_tensor)
        
        # メル・スペクトログラムから音声波形を復元（Griffin-Lim算法）
        mel_np = mel_output.squeeze().numpy()
        mel_np = librosa.db_to_power(mel_np)
        
        # Griffin-Lim算法で音声復元
        audio = librosa.feature.inverse.mel_to_audio(
            mel_np,
            sr=self.preprocessor.sample_rate,
            n_fft=2048,
            hop_length=512
        )
        
        # 音声を保存
        sf.write(output_path, audio, self.preprocessor.sample_rate)
        print(f"Synthesized speech saved to {output_path}")
        
        return audio
    
    def add_new_data(self, new_audio_path: str, new_text_path: str):
        """新しいデータを追加して追加学習"""
        # 既存のデータ数を取得
        existing_files, _ = self.collect_data_files()
        next_index = len(existing_files) + 1
        
        # 新しいファイルを適切な場所にコピー
        new_audio_dest = os.path.join(self.audio_path, f"audio_{next_index}.wav")
        new_text_dest = os.path.join(self.meta_path, f"meta_{next_index}.txt")
        
        try:
            # ファイルをコピー
            shutil.copy2(new_audio_path, new_audio_dest)
            shutil.copy2(new_text_path, new_text_dest)
            
            print(f"Copied audio file to: {new_audio_dest}")
            print(f"Copied text file to: {new_text_dest}")
            
            # 追加学習を実行
            print("Performing incremental learning...")
            self.train_model(epochs=20)  # 少ないエポック数で追加学習
            
        except Exception as e:
            print(f"Error adding new data: {e}")

# 使用例
def main():
    # 音声クローナーを初期化
    cloner = VoiceCloner()
    
    print("=== 音声クローニングシステム ===")
    print("1. データセットの前処理とモデル訓練")
    print("2. 既存モデルの読み込み")
    print("3. 音声合成")
    print("4. 新しいデータの追加")
    print("5. データファイル確認")
    
    choice = input("選択してください (1-5): ")
    
    if choice == "1":
        # モデル訓練
        print("モデル訓練を開始します...")
        cloner.train_model(epochs=50)
        cloner.save_model()
        
    elif choice == "2":
        # モデル読み込み
        model_path = input("モデルファイルのパス（空白でデフォルト）: ")
        if not model_path:
            model_path = "voice_clone_model.pth"
        
        try:
            cloner.load_model(model_path)
            print("モデル読み込み完了")
        except FileNotFoundError:
            print("モデルファイルが見つかりません")
            return
    
    elif choice == "3":
        # 音声合成
        try:
            cloner.load_model()
        except:
            print("モデルが見つかりません。先に訓練してください。")
            return
        
        text = input("合成したいテキストを入力: ")
        output_path = input("出力ファイル名（空白でdefault）: ")
        if not output_path:
            output_path = "synthesized_speech.wav"
        
        cloner.synthesize_speech(text, output_path)
        
    elif choice == "4":
        # 新しいデータ追加
        audio_path = input("新しい音声ファイルのパス: ")
        text_path = input("新しいテキストファイルのパス: ")
        
        cloner.add_new_data(audio_path, text_path)
    
    elif choice == "5":
        # データファイル確認
        audio_files, text_files = cloner.collect_data_files()
        print("\n=== 検出されたデータファイル ===")
        for i, (audio, text) in enumerate(zip(audio_files, text_files)):
            print(f"{i+1}. Audio: {audio}")
            print(f"   Text: {text}")
        
        if len(audio_files) == 0:
            print("データファイルが見つかりません。")
            print("正しいディレクトリ構造を確認してください：")
            print("dataset/")
            print("├── audio_files/")
            print("│   ├── audio_1.wav")
            print("│   ├── audio_2.wav")
            print("│   └── ...")
            print("└── meta_files/")
            print("    ├── meta_1.txt")
            print("    ├── meta_2.txt")
            print("    └── ...")

if __name__ == "__main__":
    main()