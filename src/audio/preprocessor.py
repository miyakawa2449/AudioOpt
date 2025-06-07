import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional
import warnings

class AudioPreprocessor:
    """音声前処理クラス（ノイズ除去など）"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
        # リサンプラーを事前定義（よく使うサンプルレートに対応）
        self.resamplers = {}
    
    def _get_resampler(self, orig_sr: int, target_sr: int):
        """リサンプラーを取得（キャッシュ機能付き）"""
        key = f"{orig_sr}_{target_sr}"
        if key not in self.resamplers:
            self.resamplers[key] = T.Resample(orig_sr, target_sr)
        return self.resamplers[key]
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """音声ファイルを読み込み"""
        try:
            # torchaudioを使用して読み込み
            waveform, sample_rate = torchaudio.load(file_path)
            
            # モノラルに変換（必要に応じて）
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # numpy配列に変換
            audio = waveform.squeeze().numpy()
            
            # リサンプリング（必要に応じて）
            if sample_rate != self.sample_rate:
                resampler = self._get_resampler(sample_rate, self.sample_rate)
                waveform_resampled = resampler(waveform)
                audio = waveform_resampled.squeeze().numpy()
            
            return audio
            
        except Exception as e:
            print(f"Error loading audio file {file_path} with torchaudio: {e}")
            # soundfileでの代替読み込み
            try:
                audio, sr = sf.read(file_path)
                
                # ステレオからモノラルに変換
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                if sr != self.sample_rate:
                    audio = self._simple_resample(audio, sr, self.sample_rate)
                return audio
            except Exception as e2:
                print(f"Fallback loading with soundfile also failed: {e2}")
                return np.array([])
    
    def _simple_resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """簡易的なリサンプリング"""
        if orig_sr == target_sr:
            return audio
        
        # 線形補間によるリサンプリング
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        if new_length <= 0:
            return np.array([])
        
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        new_audio = np.interp(old_indices, np.arange(len(audio)), audio)
        
        return new_audio
    
    def _stft(self, audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
        """STFTを計算"""
        # numpyからtensorに変換
        audio_tensor = torch.from_numpy(audio).float()
        
        # ウィンドウ関数
        window = torch.hann_window(n_fft)
        
        # STFT計算
        stft = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        
        return stft
    
    def _istft(self, stft_matrix: torch.Tensor, hop_length: int = 512) -> np.ndarray:
        """iSTFTを計算"""
        # ウィンドウ関数
        n_fft = (stft_matrix.shape[0] - 1) * 2
        window = torch.hann_window(n_fft)
        
        # iSTFT計算
        audio = torch.istft(
            stft_matrix,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window
        )
        
        return audio.numpy()
    
    def denoise_spectral_subtraction(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """スペクトル減算によるノイズ除去"""
        if len(audio) == 0:
            return audio
            
        # 短時間フーリエ変換
        stft = self._stft(audio, n_fft=2048, hop_length=512)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # ノイズ推定（最初の0.5秒をノイズとして使用）
        noise_frames = max(1, int(0.5 * self.sample_rate / 512))
        noise_profile = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)
        
        # スペクトル減算
        alpha = 2.0  # オーバーサブトラクション係数
        beta = 0.01  # フロア係数
        
        enhanced_magnitude = magnitude - alpha * noise_profile
        enhanced_magnitude = torch.maximum(enhanced_magnitude, beta * magnitude)
        
        # 位相を復元して逆変換
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = self._istft(enhanced_stft, hop_length=512)
        
        return enhanced_audio
    
    def denoise_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """ウィーナーフィルタによるノイズ除去"""
        if len(audio) == 0:
            return audio
            
        # 簡易的なウィーナーフィルタ実装
        stft = self._stft(audio, n_fft=2048, hop_length=512)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # ノイズ分散推定
        noise_frames = max(1, int(0.5 * self.sample_rate / 512))
        noise_var = torch.var(magnitude[:, :noise_frames], dim=1, keepdim=True)
        noise_var = torch.maximum(noise_var, torch.tensor(1e-10))  # ゼロ除算防止
        
        # ウィーナーフィルタ適用
        signal_var = torch.var(magnitude, dim=1, keepdim=True)
        wiener_gain = signal_var / (signal_var + noise_var)
        
        enhanced_magnitude = magnitude * wiener_gain
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = self._istft(enhanced_stft, hop_length=512)
        
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
        
        # torchaudioを使ったRMS計算
        audio_tensor = torch.from_numpy(audio).float()
        
        # フレーム化
        frames = audio_tensor.unfold(0, frame_length, hop_length)
        
        # RMSを計算
        energy = torch.sqrt(torch.mean(frames**2, dim=1))
        
        # しきい値以上のフレームを特定
        frames_above_threshold = energy > threshold
        
        # 時間軸の計算
        times = torch.arange(len(energy)) * hop_length / self.sample_rate
        
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
    
    def process_audio(self, file_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """音声の総合的な前処理"""
        print(f"Processing: {file_path}")
        
        # 音声読み込み
        audio = self.load_audio(file_path)
        if len(audio) == 0:
            print(f"Warning: Could not load audio from {file_path}")
            return audio
        
        print(f"Loaded audio: shape={audio.shape}, duration={len(audio)/self.sample_rate:.2f}s")
        
        # ノイズ除去
        try:
            audio = self.denoise_spectral_subtraction(audio)
            print("Applied spectral subtraction denoising")
        except Exception as e:
            print(f"Denoising failed: {e}, skipping...")
        
        # 無音除去
        try:
            original_length = len(audio)
            audio = self.remove_silence(audio)
            print(f"Silence removal: {original_length} -> {len(audio)} samples")
        except Exception as e:
            print(f"Silence removal failed: {e}, skipping...")
        
        # 正規化
        audio = self.normalize_audio(audio)
        print("Applied normalization")
        
        # 保存
        if output_path:
            try:
                sf.write(output_path, audio, self.sample_rate)
                print(f"Saved processed audio to: {output_path}")
            except Exception as e:
                print(f"Error saving audio to {output_path}: {e}")
        
        return audio