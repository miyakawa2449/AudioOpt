import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
import shutil
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from ..audio import AudioPreprocessor
from ..text import TextProcessor
from ..model import VoiceDataset, VoiceCloneModel

class VoiceCloner:
    """音声クローニングメインクラス"""
    
    def __init__(self, dataset_path: str = "dataset"):
        """音声クローニングメインクラス"""
    
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
        
        # モデル保存用
        self.models_path = "models"
        os.makedirs(self.models_path, exist_ok=True)
        
        # 出力用
        self.output_path = "output"
        os.makedirs(self.output_path, exist_ok=True)
        
        # GPU使用可能かチェック
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Griffin-Lim変換器を設定（デバイスに配置）
        self.griffin_lim = T.GriffinLim(
            n_fft=2048,
            hop_length=512,
            power=1.0,
            n_iter=60
        ).to(self.device)  # デバイスに移動
    
        # メル逆変換器を設定（デバイスに配置）
        self.inverse_mel = T.InverseMelScale(
            n_stft=1025,  # n_fft // 2 + 1
            n_mels=80,
            sample_rate=self.preprocessor.sample_rate,
            f_min=0.0,
            f_max=8000.0
        ).to(self.device)  # デバイスに移動
    
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
        
        print(f"Found {len(audio_files)} audio-text pairs")
        if len(audio_files) == 0:
            print(f"Please ensure your files are in:")
            print(f"  Audio files: {self.audio_path}")
            print(f"  Text files: {self.meta_path}")
        
        return audio_files, text_files
    
    def train_model(self, epochs: int = 50, batch_size: int = 4, learning_rate: float = 1e-3):
        """モデルを訓練"""
        print("Starting model training...")
        
        # データ収集
        audio_files, text_files = self.collect_data_files()
        if len(audio_files) == 0:
            print("No training data found!")
            return
        
        # テキスト処理
        texts = []
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                texts.append(self.text_processor.clean_text(text))
            except Exception as e:
                print(f"Error reading {text_file}: {e}")
        
        # 語彙構築
        self.text_processor.build_vocab(texts)
        print(f"Vocabulary size: {len(self.text_processor.vocab)}")
        
        # 語彙を保存
        vocab_path = os.path.join(self.processed_path, "vocabulary.json")
        self.text_processor.save_vocab(vocab_path)
        
        # データセット作成
        dataset = VoiceDataset(audio_files, text_files, self.preprocessor, self.text_processor)
        
        if len(dataset) == 0:
            print("No valid data found in dataset!")
            return
        
        # 前処理済みデータを保存
        self._save_processed_data(dataset)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=dataset.collate_fn)
        
        # モデル初期化
        vocab_size = len(self.text_processor.vocab)
        self.model = VoiceCloneModel(vocab_size=vocab_size).to(self.device)
        
        # 損失関数と最適化器
        criterion_mel = nn.MSELoss()
        criterion_stop = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 訓練ループ
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (audio_features, audio_lengths, text_sequences, text_lengths) in enumerate(dataloader):
                audio_features = audio_features.to(self.device)
                text_sequences = text_sequences.to(self.device)
                audio_lengths = audio_lengths.to(self.device)
                text_lengths = text_lengths.to(self.device)
                
                optimizer.zero_grad()
                
                # モデル推論
                mel_outputs, stop_outputs = self.model(text_sequences, text_lengths, audio_features)
                
                # 損失計算
                # パディング部分を除外するためのマスク
                max_len = min(mel_outputs.size(1), audio_features.size(1))
                mel_outputs = mel_outputs[:, :max_len, :]
                target_mels = audio_features[:, :max_len, :]
                
                loss_mel = criterion_mel(mel_outputs, target_mels)
                
                # 停止トークン用の正解データ作成
                stop_targets = torch.zeros_like(stop_outputs)
                for i, length in enumerate(audio_lengths):
                    if length < stop_outputs.size(1):
                        stop_targets[i, length-1:, :] = 1.0
                
                loss_stop = criterion_stop(stop_outputs, stop_targets)
                
                total_loss_batch = loss_mel + loss_stop
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}], No valid batches processed")
        
        print("Training completed!")
    
    def _save_processed_data(self, dataset):
        """前処理済みデータを保存"""
        import pickle
        
        processed_data = {
            'audio_features': dataset.audio_features,
            'text_sequences': dataset.text_sequences,
            'sample_rate': self.preprocessor.sample_rate
        }
        
        processed_file = os.path.join(self.processed_path, "processed_data.pkl")
        
        try:
            with open(processed_file, 'wb') as f:
                pickle.dump(processed_data, f)
            print(f"Processed data saved to: {processed_file}")
        except Exception as e:
            print(f"Error saving processed data: {e}")

    def save_model(self, model_path: Optional[str] = None):
        """モデルを保存"""
        if model_path is None:
            model_path = os.path.join(self.models_path, "voice_clone_model.pth")
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'text_processor': self.text_processor,
                'preprocessor': self.preprocessor
            }, model_path)
            print(f"Model saved to: {model_path}")
        else:
            print("No model to save!")
    
    def load_model(self, model_path: Optional[str] = None):
        """モデルを読み込み"""
        if model_path is None:
            model_path = os.path.join(self.models_path, "voice_clone_model.pth")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # テキストプロセッサとプリプロセッサを復元
        self.text_processor = checkpoint['text_processor']
        self.preprocessor = checkpoint['preprocessor']
        
        # モデルを再構築
        vocab_size = len(self.text_processor.vocab)
        self.model = VoiceCloneModel(vocab_size=vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
    
    def synthesize_speech(self, text: str, output_path: Optional[str] = None):
        """テキストから音声を合成"""
        if self.model is None:
            print("Model not loaded!")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_path, "synthesized_speech.wav")
        
        print(f"Synthesizing: '{text}'")
        
        # テキストを数値序列に変換
        text_sequence = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([len(text_sequence)]).to(self.device)
        
        print(f"Text sequence length: {len(text_sequence)}")
        
        # 音声合成
        with torch.no_grad():
            mel_outputs, _ = self.model(text_tensor, text_lengths)
        
        print(f"Mel outputs shape: {mel_outputs.shape}")
        
        # CPUに移動
        mel_spec = mel_outputs.squeeze(0).transpose(0, 1).cpu()  # (mel_bins, time)
        print(f"Mel spec shape after transpose: {mel_spec.shape}")
        
        # デシベルからリニアスケールに変換
        mel_spec_linear = torch.pow(10, mel_spec / 20.0)  # 20で割る（dBの正しい変換）
        
        # メルスペクトログラムを線形スペクトログラムに変換
        inverse_mel_cpu = T.InverseMelScale(
            n_stft=1025,  # n_fft // 2 + 1 = 2048 // 2 + 1
            n_mels=80,
            sample_rate=self.preprocessor.sample_rate,
            f_min=0.0,
            f_max=8000.0
        )
        
        try:
            # メル→線形変換
            linear_spec = inverse_mel_cpu(mel_spec_linear)
            print(f"Linear spec shape: {linear_spec.shape}")
            
            # Griffin-Limで音声復元
            griffin_lim_cpu = T.GriffinLim(
                n_fft=2048,
                hop_length=512,
                power=1.0,
                n_iter=60
            )
            
            audio = griffin_lim_cpu(linear_spec)
            print(f"Audio shape: {audio.shape}")
            
            # 正規化
            audio = audio / torch.max(torch.abs(audio)) * 0.9
            
            # 音声を保存
            torchaudio.save(output_path, audio.unsqueeze(0), self.preprocessor.sample_rate)
            print(f"✓ Synthesized speech saved to: {output_path}")
            
        except Exception as e:
            print(f"Error in mel-to-linear conversion: {e}")
            
            # フォールバック：簡易的な音声生成
            print("Attempting fallback synthesis...")
            try:
                # 最も簡単な方法：メルスペクトログラムを直接時間軸に変換
                # これは音質が良くないですが、動作確認のため
                audio_simple = torch.mean(mel_spec_linear, dim=0)  # 周波数軸を平均
                
                # 長さを調整
                target_length = int(self.preprocessor.sample_rate * 2)  # 2秒
                if len(audio_simple) < target_length:
                    # 繰り返し
                    repeat_times = target_length // len(audio_simple) + 1
                    audio_simple = audio_simple.repeat(repeat_times)[:target_length]
                else:
                    audio_simple = audio_simple[:target_length]
                
                # 正規化
                audio_simple = audio_simple - torch.mean(audio_simple)
                audio_simple = audio_simple / torch.max(torch.abs(audio_simple)) * 0.5
                
                # サイン波でトーンを生成（テスト用）
                t = torch.linspace(0, 2, target_length)
                frequency = 440  # A4音
                audio_tone = 0.3 * torch.sin(2 * 3.14159 * frequency * t)
                
                # 音声を保存
                torchaudio.save(output_path, audio_tone.unsqueeze(0), self.preprocessor.sample_rate)
                print(f"✓ Fallback audio saved to: {output_path}")
                print("Note: This is a simple tone for testing. The model may need more training.")
                
            except Exception as e2:
                print(f"Fallback synthesis also failed: {e2}")
    
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
    
    def synthesize_speech_with_vocoder(self, text: str, output_path: Optional[str] = None):
        """Vocoderを使用した高品質音声合成"""
        if self.model is None:
            print("Model not loaded!")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_path, "synthesized_speech_hq.wav")
        
        # テキストを数値序列に変換
        text_sequence = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([len(text_sequence)]).to(self.device)
        
        # 音声合成
        with torch.no_grad():
            mel_outputs, _ = self.model(text_tensor, text_lengths)
        
        # CPUに移動
        mel_spec = mel_outputs.squeeze(0).transpose(0, 1).cpu()
        
        # 簡易的なvocoderの実装
        try:
            # フィルタバンクを使用した音声復元
            audio = self._simple_vocoder(mel_spec)
            
            # 正規化
            audio = audio / torch.max(torch.abs(audio)) * 0.9
            
            # 音声を保存
            torchaudio.save(output_path, audio.unsqueeze(0), self.preprocessor.sample_rate)
            print(f"✓ High-quality synthesized speech saved to: {output_path}")
            
        except Exception as e:
            print(f"Vocoder synthesis failed: {e}")

    def _simple_vocoder(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """簡易的なボコーダーの実装"""
        mel_bins, time_frames = mel_spec.shape
        
        # フィルタバンクを作成
        frequencies = torch.linspace(80, 8000, mel_bins)  # 80Hz - 8kHz
        
        # 各時間フレームで正弦波を合成
        sample_rate = self.preprocessor.sample_rate
        hop_length = 512
        frame_length = hop_length
        
        audio_length = time_frames * hop_length
        audio = torch.zeros(audio_length)
        
        for t in range(time_frames):
            start_sample = t * hop_length
            end_sample = start_sample + frame_length
            
            frame_audio = torch.zeros(frame_length)
            
            for f in range(mel_bins):
                # 各周波数成分の振幅
                amplitude = mel_spec[f, t]
                frequency = frequencies[f]
                
                # 正弦波を生成
                time_indices = torch.arange(frame_length, dtype=torch.float32) / sample_rate
                sine_wave = amplitude * torch.sin(2 * 3.14159 * frequency * time_indices)
                frame_audio += sine_wave
            
            # 音声に加算
            if end_sample <= len(audio):
                audio[start_sample:end_sample] += frame_audio
    
        return audio
    
    def synthesize_speech_improved(self, text: str, output_path: Optional[str] = None):
        """改善された音声合成"""
        if self.model is None:
            print("Model not loaded!")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_path, "synthesized_speech.wav")
        
        print(f"Synthesizing: '{text}'")
        
        # テキストを数値序列に変換
        text_sequence = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([len(text_sequence)]).to(self.device)
        
        # 音声合成
        with torch.no_grad():
            mel_outputs, _ = self.model(text_tensor, text_lengths)
        
        # CPUに移動
        mel_spec = mel_outputs.squeeze(0).cpu()  # (time, mel_bins)
        
        print(f"Generated mel spectrogram shape: {mel_spec.shape}")
        
        # メルスペクトログラムの後処理
        mel_spec = torch.clamp(mel_spec, -10, 10)  # 値をクリップ
        
        try:
            # 改善されたボコーダー
            audio = self._improved_vocoder(mel_spec)
            
            # 音声の後処理
            audio = self._post_process_audio(audio)
            
            # 音声を保存
            torchaudio.save(output_path, audio.unsqueeze(0), self.preprocessor.sample_rate)
            print(f"✓ Synthesized speech saved to: {output_path}")
            
        except Exception as e:
            print(f"Synthesis failed: {e}")
            # 最終的なフォールバック：トーン生成
            self._generate_tone_sequence(text, output_path)

    def _improved_vocoder(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """改善されたボコーダー"""
        time_frames, mel_bins = mel_spec.shape
        
        # パラメータ
        sample_rate = self.preprocessor.sample_rate
        hop_length = 512
        
        # フィルタバンクの設定（メル尺度）
        mel_freqs = torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), mel_bins)
        linear_freqs = 700 * (torch.pow(10, mel_freqs / 2595) - 1)
        
        # 音声長を計算
        audio_length = time_frames * hop_length
        audio = torch.zeros(audio_length)
        
        # 時間窓
        window = torch.hann_window(hop_length * 2)
        
        for t in range(time_frames):
            start_sample = t * hop_length
            end_sample = min(start_sample + hop_length * 2, audio_length)
            frame_length = end_sample - start_sample
            
            if frame_length <= 0:
                continue
            
            frame_audio = torch.zeros(frame_length)
            
            # 基本周波数の推定（簡易）
            f0 = 150 + 50 * torch.sin(torch.tensor(t * 0.1))  # 動的なピッチ
            
            # 倍音を生成
            for harmonic in range(1, 6):  # 5倍音まで
                freq = f0 * harmonic
                if freq > 8000:  # ナイキスト周波数以下
                    break
                
                # この周波数に対応するメルビンを見つける
                mel_bin = torch.argmin(torch.abs(linear_freqs - freq))
                amplitude = mel_spec[t, mel_bin] * torch.exp(-torch.tensor((harmonic - 1) * 0.5))
                
                # 正弦波を生成
                time_indices = torch.arange(frame_length, dtype=torch.float32) / sample_rate
                phase = 2 * 3.14159 * freq * time_indices
                sine_wave = amplitude * torch.sin(phase)
                
                frame_audio += sine_wave
            
            # ノイズ成分を追加（無声音用）
            noise_level = torch.mean(mel_spec[t, 60:])  # 高周波成分
            if noise_level > 0.1:
                noise = noise_level * 0.1 * torch.randn(frame_length)
                frame_audio += noise
            
            # 窓関数を適用
            if frame_length == len(window):
                frame_audio *= window
            else:
                frame_audio *= window[:frame_length]
            
            # 音声に追加
            audio[start_sample:end_sample] += frame_audio
        
        return audio

    def _post_process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """音声の後処理"""
        # DC成分除去
        audio = audio - torch.mean(audio)
        
        # 正規化
        max_amp = torch.max(torch.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp * 0.8
        
        # 簡易的なローパスフィルタ
        kernel_size = 5
        kernel = torch.ones(kernel_size) / kernel_size
        audio_padded = F.pad(audio, (kernel_size//2, kernel_size//2), mode='reflect')
        audio_filtered = F.conv1d(audio_padded.unsqueeze(0).unsqueeze(0), 
                                 kernel.unsqueeze(0).unsqueeze(0)).squeeze()
        
        return audio_filtered

    def _generate_tone_sequence(self, text: str, output_path: str):
        """テキストに基づいたトーン系列を生成（最終フォールバック）"""
        # 文字数に基づく音の長さ
        duration_per_char = 0.2
        total_duration = len(text) * duration_per_char
        
        sample_rate = self.preprocessor.sample_rate
        total_samples = int(total_duration * sample_rate)
        
        audio = torch.zeros(total_samples)
        
        # 各文字に異なる周波数を割り当て
        for i, char in enumerate(text):
            start_sample = int(i * duration_per_char * sample_rate)
            end_sample = int((i + 1) * duration_per_char * sample_rate)
            
            if end_sample > total_samples:
                end_sample = total_samples
            
            # 文字コードに基づく周波数
            char_code = ord(char) if ord(char) < 1000 else ord(char) % 1000
            frequency = 200 + (char_code % 500)  # 200-700Hz
            
            # トーン生成
            t = torch.linspace(0, duration_per_char, end_sample - start_sample)
            tone = 0.3 * torch.sin(2 * 3.14159 * frequency * t)
            
            # エンベロープ
            envelope = torch.sin(3.14159 * t / duration_per_char)
            tone *= envelope
            
            audio[start_sample:end_sample] = tone
        
        # 音声を保存
        torchaudio.save(output_path, audio.unsqueeze(0), sample_rate)
        print(f"✓ Tone sequence saved to: {output_path}")
        print("Note: This is a synthetic tone sequence. More training data is needed for speech synthesis.")

    def augment_dataset(self):
        """データ拡張"""
        audio_files, text_files = self.collect_data_files()
        
        augmented_audio_path = os.path.join(self.dataset_path, "augmented_audio")
        augmented_text_path = os.path.join(self.dataset_path, "augmented_text")
        
        os.makedirs(augmented_audio_path, exist_ok=True)
        os.makedirs(augmented_text_path, exist_ok=True)
        
        for i, (audio_file, text_file) in enumerate(zip(audio_files, text_files)):
            # 元のテキストを読み込み
            with open(text_file, 'r', encoding='utf-8') as f:
                original_text = f.read().strip()
            
            # 音声を読み込み
            audio = self.preprocessor.load_audio(audio_file)
            
            if len(audio) > 0:
                # 1. 速度変更
                for speed in [0.9, 1.1]:
                    augmented_audio = self._change_speed(audio, speed)
                    aug_audio_file = os.path.join(augmented_audio_path, f"aug_speed_{speed}_{i}.wav")
                    aug_text_file = os.path.join(augmented_text_path, f"aug_speed_{speed}_{i}.txt")
                    
                    torchaudio.save(aug_audio_file, torch.from_numpy(augmented_audio).unsqueeze(0), 
                                  self.preprocessor.sample_rate)
                    
                    with open(aug_text_file, 'w', encoding='utf-8') as f:
                        f.write(original_text)
                
                # 2. ピッチ変更
                for pitch_shift in [-2, 2]:  # 半音
                    augmented_audio = self._change_pitch(audio, pitch_shift)
                    aug_audio_file = os.path.join(augmented_audio_path, f"aug_pitch_{pitch_shift}_{i}.wav")
                    aug_text_file = os.path.join(augmented_text_path, f"aug_pitch_{pitch_shift}_{i}.txt")
                    
                    torchaudio.save(aug_audio_file, torch.from_numpy(augmented_audio).unsqueeze(0), 
                                  self.preprocessor.sample_rate)
                    
                    with open(aug_text_file, 'w', encoding='utf-8') as f:
                        f.write(original_text)

    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """音声の速度を変更"""
        indices = np.arange(0, len(audio), speed)
        return np.interp(indices, np.arange(len(audio)), audio)

    def _change_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """音声のピッチを変更（簡易版）"""
        shift_factor = 2 ** (semitones / 12)
        indices = np.arange(len(audio)) * shift_factor
        indices = np.clip(indices, 0, len(audio) - 1)
        return np.interp(indices, np.arange(len(audio)), audio)