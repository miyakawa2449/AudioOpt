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
from datetime import datetime

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
        
        print(f"Found {len(audio_files)} audio-text pairs")
        
        # 前処理済み音声を保存するディレクトリを作成
        processed_audio_dir = os.path.join(self.processed_path, "cleaned_audio")
        comparison_dir = os.path.join(self.processed_path, "comparison")
        os.makedirs(processed_audio_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)
        
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
        print(f"Built vocabulary with {len(self.text_processor.vocab)} characters")
        print(f"Vocabulary size: {len(self.text_processor.vocab)}")
    
        # 前処理済み音声を保存しながらデータセット作成
        print("\n=== 音声前処理と保存 ===")
        valid_audio_files = []
        valid_text_files = []
        
        for i, (audio_file, text_file) in enumerate(zip(audio_files, text_files)):
            print(f"\nProcessing {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
            
            # オリジナル音声を読み込み
            original_audio = self.preprocessor.load_audio(audio_file)
            if len(original_audio) == 0:
                print(f"  ❌ Failed to load audio")
                continue
            
            print(f"  Original: {len(original_audio)} samples ({len(original_audio)/self.preprocessor.sample_rate:.2f}s)")
            
            # 前処理実行
            processed_audio = self._process_and_save_audio(audio_file, i+1, processed_audio_dir, comparison_dir)
            
            if len(processed_audio) > 0:
                valid_audio_files.append(audio_file)
                valid_text_files.append(text_file)
                print(f"  ✓ Processed: {len(processed_audio)} samples ({len(processed_audio)/self.preprocessor.sample_rate:.2f}s)")
            else:
                print(f"  ❌ Processing failed")
        
        print(f"\n✓ Successfully processed {len(valid_audio_files)}/{len(audio_files)} files")
        
        if len(valid_audio_files) == 0:
            print("❌ No valid audio files found after preprocessing!")
            return
        
        # データセット作成（前処理済みファイルを使用）
        dataset = VoiceDataset(valid_audio_files, valid_text_files, self.preprocessor, self.text_processor)
        
        if len(dataset) == 0:
            print("No valid data found in dataset!")
            return
        
        # 前処理統計を保存
        self._save_preprocessing_stats(valid_audio_files, processed_audio_dir)
        
        # 以下、既存の訓練コード...
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=dataset.collate_fn
        )
        
        # モデル初期化
        if self.model is None:
            vocab_size = len(self.text_processor.vocab)
            self.model = VoiceCloneModel(vocab_size).to(self.device)
    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        
        print(f"\nStarting training: {epochs} epochs, batch size {batch_size}")
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (audio_features, audio_lengths, text_sequences, text_lengths) in enumerate(dataloader):
                audio_features = audio_features.to(self.device)
                text_sequences = text_sequences.to(self.device)
                audio_lengths = audio_lengths.to(self.device)
                text_lengths = text_lengths.to(self.device)
                
                optimizer.zero_grad()
                
                # モデル推論
                mel_outputs, stop_tokens = self.model(text_sequences, text_lengths, audio_features)
                
                # 損失計算
                loss = criterion(mel_outputs, audio_features)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        print("Training completed!")

    def _process_and_save_audio(self, audio_file: str, file_index: int, processed_dir: str, comparison_dir: str) -> np.ndarray:
        """音声を前処理して保存"""
        import soundfile as sf
        import matplotlib.pyplot as plt
    
        # オリジナル音声を読み込み
        original_audio = self.preprocessor.load_audio(audio_file)
        if len(original_audio) == 0:
            return np.array([])
    
        # 段階的な前処理
        print("  Step 1: ノイズ除去中...")
        try:
            denoised_audio = self.preprocessor.denoise_spectral_subtraction(original_audio)
            print(f"    ✓ Denoised: {len(denoised_audio)} samples")
        except Exception as e:
            print(f"    ❌ Denoising failed: {e}")
            denoised_audio = original_audio.copy()
    
        print("  Step 2: 無音除去中...")
        try:
            trimmed_audio = self.preprocessor.remove_silence(denoised_audio)
            print(f"    ✓ Trimmed: {len(trimmed_audio)} samples")
        except Exception as e:
            print(f"    ❌ Silence removal failed: {e}")
            trimmed_audio = denoised_audio.copy()
    
        print("  Step 3: 正規化中...")
        try:
            normalized_audio = self.preprocessor.normalize_audio(trimmed_audio)
            print(f"    ✓ Normalized: {len(normalized_audio)} samples")
        except Exception as e:
            print(f"    ❌ Normalization failed: {e}")
            normalized_audio = trimmed_audio.copy()
    
        # === 比較用音声ディレクトリを作成 ===
        audio_comparison_dir = os.path.join(comparison_dir, "audio_pairs")
        os.makedirs(audio_comparison_dir, exist_ok=True)
    
        # オリジナル音声をコピー保存（比較用）
        original_filename = f"original_audio_{file_index:03d}.wav"
        original_comparison_path = os.path.join(audio_comparison_dir, original_filename)
        try:
            sf.write(original_comparison_path, original_audio, self.preprocessor.sample_rate)
            print(f"    ✓ Original saved for comparison: {original_comparison_path}")
        except Exception as e:
            print(f"    ❌ Original save failed: {e}")
    
        # 前処理済み音声を保存（メイン）
        processed_filename = f"processed_audio_{file_index:03d}.wav"
        processed_path = os.path.join(processed_dir, processed_filename)
    
        try:
            sf.write(processed_path, normalized_audio, self.preprocessor.sample_rate)
            print(f"    ✓ Processed saved: {processed_path}")
        except Exception as e:
            print(f"    ❌ Processed save failed: {e}")
            return np.array([])
    
        # 前処理済み音声を比較用ディレクトリにもコピー
        processed_comparison_filename = f"processed_audio_{file_index:03d}.wav"
        processed_comparison_path = os.path.join(audio_comparison_dir, processed_comparison_filename)
        try:
            sf.write(processed_comparison_path, normalized_audio, self.preprocessor.sample_rate)
            print(f"    ✓ Processed saved for comparison: {processed_comparison_path}")
        except Exception as e:
            print(f"    ❌ Processed comparison save failed: {e}")
    
        # 段階別音声も保存（詳細な比較用）
        stages_dir = os.path.join(comparison_dir, "processing_stages")
        os.makedirs(stages_dir, exist_ok=True)
    
        try:
            # 段階1: ノイズ除去のみ
            sf.write(
                os.path.join(stages_dir, f"stage1_denoised_{file_index:03d}.wav"),
                denoised_audio, self.preprocessor.sample_rate
            )
            
            # 段階2: 無音除去まで
            sf.write(
                os.path.join(stages_dir, f"stage2_trimmed_{file_index:03d}.wav"),
                trimmed_audio, self.preprocessor.sample_rate
            )
            
            # 段階3: 正規化まで（最終）
            sf.write(
                os.path.join(stages_dir, f"stage3_normalized_{file_index:03d}.wav"),
                normalized_audio, self.preprocessor.sample_rate
            )
            
            print(f"    ✓ Processing stages saved to: {stages_dir}")
            
        except Exception as e:
            print(f"    ❌ Stages save failed: {e}")
    
        # 比較用の可視化を作成
        self._create_comparison_plot(original_audio, normalized_audio, file_index, comparison_dir)
    
        return normalized_audio

    def _create_comparison_plot(self, original: np.ndarray, processed: np.ndarray, file_index: int, comparison_dir: str):
        """オリジナルと前処理済み音声の比較プロットを作成"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            fig.suptitle(f'Audio Processing Comparison - File {file_index:03d}')
            
            # 時間軸
            time_orig = np.arange(len(original)) / self.preprocessor.sample_rate
            time_proc = np.arange(len(processed)) / self.preprocessor.sample_rate
            
            # 1. 波形比較
            axes[0].plot(time_orig, original, alpha=0.7, label='Original', color='blue')
            axes[0].set_title('Original Waveform')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_proc, processed, alpha=0.7, label='Processed', color='red')
            axes[1].set_title('Processed Waveform')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)
            
            # 2. スペクトログラム（オリジナル）
            if len(original) > 1024:
                D_orig = np.abs(np.fft.fft(original[:len(original)//1024*1024].reshape(-1, 1024), axis=1))
                axes[2].imshow(D_orig.T, aspect='auto', origin='lower', cmap='viridis')
                axes[2].set_title('Original Spectrogram')
                axes[2].set_ylabel('Frequency')
        
            # 3. スペクトログラム（前処理済み）
            if len(processed) > 1024:
                D_proc = np.abs(np.fft.fft(processed[:len(processed)//1024*1024].reshape(-1, 1024), axis=1))
                axes[3].imshow(D_proc.T, aspect='auto', origin='lower', cmap='viridis')
                axes[3].set_title('Processed Spectrogram')
                axes[3].set_ylabel('Frequency')
                axes[3].set_xlabel('Time')
        
            plt.tight_layout()
        
            # 保存
            plot_filename = f"comparison_{file_index:03d}.png"
            plot_path = os.path.join(comparison_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        
            print(f"    ✓ Comparison plot saved: {plot_path}")
        
        except Exception as e:
            print(f"    ❌ Plot creation failed: {e}")

    def _save_preprocessing_stats(self, audio_files: list, processed_dir: str):
        """前処理統計を保存"""
        import json
    
        stats = {
            "processing_summary": {
                "total_files": len(audio_files),
                "processed_files": len(os.listdir(processed_dir)),
                "success_rate": len(os.listdir(processed_dir)) / len(audio_files) * 100,
                "timestamp": str(datetime.now())
            },
            "file_details": []
        }
    
        for i, audio_file in enumerate(audio_files):
            processed_file = os.path.join(processed_dir, f"processed_audio_{i+1:03d}.wav")
        
            if os.path.exists(processed_file):
                # ファイルサイズ比較
                orig_size = os.path.getsize(audio_file)
                proc_size = os.path.getsize(processed_file)
            
                # 音声長比較
                try:
                    orig_audio = self.preprocessor.load_audio(audio_file)
                    proc_audio = self.preprocessor.load_audio(processed_file)
                
                    stats["file_details"].append({
                        "index": i + 1,
                        "original_file": os.path.basename(audio_file),
                        "processed_file": os.path.basename(processed_file),
                        "original_size_kb": orig_size / 1024,
                        "processed_size_kb": proc_size / 1024,
                        "size_reduction_percent": (1 - proc_size/orig_size) * 100,
                        "original_duration_sec": len(orig_audio) / self.preprocessor.sample_rate,
                        "processed_duration_sec": len(proc_audio) / self.preprocessor.sample_rate,
                        "duration_reduction_percent": (1 - len(proc_audio)/len(orig_audio)) * 100 if len(orig_audio) > 0 else 0
                    })
                except:
                    stats["file_details"].append({
                        "index": i + 1,
                        "original_file": os.path.basename(audio_file),
                        "error": "Could not analyze audio details"
                    })
    
        # 統計を保存
        stats_file = os.path.join(self.processed_path, "preprocessing_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
        print(f"\n✓ Preprocessing statistics saved: {stats_file}")
        print(f"  Success rate: {stats['processing_summary']['success_rate']:.1f}%")

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
    
    def synthesize_speech(self, text: str, output_path: str = None):
        """音声合成（出力長制御を改善）"""
        if self.model is None:
            print("❌ モデルが読み込まれていません")
            return
        
        try:
            import torch
            import torchaudio
            import os
            
            # テキスト前処理
            text_sequence = self.text_processor.text_to_sequence(text)
            text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(text_sequence)]).to(self.device)
            
            print(f"テキスト長: {len(text_sequence)} → 期待される音声長: {len(text_sequence) * 0.1:.1f}秒")
            
            # モデル推論（長さ制御を追加）
            self.model.eval()
            with torch.no_grad():
                # 最大出力長を設定（テキスト長 × 倍率）
                max_length = max(len(text_sequence) * 15, 100)  # 最低100フレーム
                
                mel_outputs, stop_outputs = self._generate_with_length_control(
                    text_tensor, text_lengths, max_length
                )
            
            mel_spec = mel_outputs.squeeze(0).cpu()
            
            print(f"生成されたメルスペクトログラム:")
            print(f"  フレーム数: {mel_spec.shape[0]}")
            print(f"  予想音声長: {mel_spec.shape[0] * 256 / self.preprocessor.sample_rate:.2f}秒")
            
            # 出力パス設定
            if output_path is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_path, f"synthesized_{timestamp}.wav")
            else:
                if not output_path.endswith('.wav'):
                    output_path += '.wav'
                output_path = os.path.join(self.output_path, output_path)
            
            # ボコーダー選択と音声生成
            print("音声合成中...")
            
            # 改善ボコーダーを優先使用
            try:
                audio = self._improved_vocoder(mel_spec)
                vocoder_used = "改善ボコーダー"
            except Exception as e:
                print(f"改善ボコーダーエラー: {e}")
                try:
                    audio = self._simple_vocoder(mel_spec)
                    vocoder_used = "簡易ボコーダー"
                except Exception as e2:
                    print(f"簡易ボコーダーエラー: {e2}")
                    audio = self._griffin_lim_synthesis(mel_spec)
                    vocoder_used = "Griffin-Lim"
        
            # 音声品質チェックと修正
            max_amp = torch.max(torch.abs(audio))
            print(f"生成音声の最大振幅: {max_amp:.6f}")
            
            if max_amp > 1.0:
                print(f"⚠️  振幅が大きすぎます。正規化中...")
                audio = audio / max_amp * 0.8
                print(f"正規化後の最大振幅: {torch.max(torch.abs(audio)):.6f}")
            elif max_amp < 0.01:
                print(f"⚠️  振幅が小さすぎます。増幅中...")
                audio = audio / max_amp * 0.5
                print(f"増幅後の最大振幅: {torch.max(torch.abs(audio)):.6f}")
            
            # 音声保存
            sample_rate = self.preprocessor.sample_rate
            torchaudio.save(output_path, audio.unsqueeze(0), sample_rate)
            
            # 結果表示
            duration = len(audio) / sample_rate
            print(f"✓ 音声合成完了!")
            print(f"  入力テキスト: '{text}'")
            print(f"  出力ファイル: {output_path}")
            print(f"  音声長: {duration:.2f}秒")
            print(f"  使用ボコーダー: {vocoder_used}")
            print(f"  最終振幅範囲: [{torch.min(audio):.3f}, {torch.max(audio):.3f}]")
            
        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")
            import traceback
            traceback.print_exc()

    def _generate_with_length_control(self, text_tensor, text_lengths, max_length):
        """長さ制御付きの音声生成"""
        import torch
    
        # 基本的な推論
        mel_outputs, stop_outputs = self.model(text_tensor, text_lengths)
    
        # 出力が短すぎる場合の対処
        if mel_outputs.shape[1] < 50:  # 50フレーム（約1.1秒）未満の場合
            print(f"⚠️  出力が短すぎます（{mel_outputs.shape[1]}フレーム）。拡張中...")
            
            # パディングで長さを拡張
            target_length = max(50, len(text_tensor[0]) * 10)
            
            if mel_outputs.shape[1] < target_length:
                # 最後のフレームを繰り返して拡張
                last_frame = mel_outputs[:, -1:, :]  # 最後のフレーム
                repeat_count = target_length - mel_outputs.shape[1]
                
                # ノイズを追加して単調さを軽減
                noise = torch.randn_like(last_frame.repeat(1, repeat_count, 1)) * 0.1
                padding = last_frame.repeat(1, repeat_count, 1) + noise
                
                mel_outputs = torch.cat([mel_outputs, padding], dim=1)
                
                print(f"✓ {target_length}フレームに拡張しました")
    
        return mel_outputs, stop_outputs

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

    def _griffin_lim_synthesis(self, mel_spec):
        """Griffin-Lim アルゴリズムを修正"""
        import torch
        import torchaudio
    
        try:
            # メルスペクトログラムの次元を確認
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.T  # (time, mel) → (mel, time)
        
            print(f"Griffin-Lim入力: {mel_spec.shape}")
        
            # メルスペクトログラムを線形スペクトログラムに変換
            # 簡易的にメルビンを周波数ビンにマッピング
            n_fft = 1024
            n_mels = mel_spec.shape[0]
            n_freq = n_fft // 2 + 1  # 513
        
            # メルスペクトログラムを線形スペクトログラムのサイズに拡張
            linear_spec = torch.zeros(n_freq, mel_spec.shape[1])
        
            # 簡易的なマッピング（メル80次元 → 線形513次元）
            for i in range(n_mels):
                # メルスケールを線形スケールにマッピング
                freq_idx = int(i * (n_freq - 1) / (n_mels - 1))
                linear_spec[freq_idx] = mel_spec[i]
        
            # 中間値を補間
            for i in range(1, n_freq - 1):
                if linear_spec[i].sum() == 0:  # 空の場合
                    linear_spec[i] = (linear_spec[i-1] + linear_spec[i+1]) / 2
        
            # dBから線形スケールに変換
            linear_spec = torch.exp(linear_spec / 20.0)
        
            # Griffin-Lim変換
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=n_fft,
                hop_length=256,
                n_iter=32,
                power=1.0
            )
        
            # (freq, time) → (1, freq, time) for Griffin-Lim
            audio = griffin_lim(linear_spec.unsqueeze(0))
        
            return audio.squeeze(0)
        
        except Exception as e:
            print(f"Griffin-Lim合成エラー: {e}")
            # フォールバック: 簡易ボコーダーを使用
            return self._simple_vocoder(mel_spec)

    def _simple_vocoder(self, mel_spec):
        """簡易ボコーダー（音声振幅を正規化）"""
        import torch
        import numpy as np
    
        # メルスペクトログラムを時間軸で処理
        mel_np = mel_spec.numpy() if hasattr(mel_spec, 'numpy') else mel_spec.detach().cpu().numpy()
    
        # 各フレームをサイン波の和で近似
        sample_rate = self.preprocessor.sample_rate
        hop_length = 256  # フレーム間のサンプル数
    
        audio_length = mel_np.shape[0] * hop_length
        audio = np.zeros(audio_length)
    
        # 周波数ビンを実際の周波数にマッピング
        mel_frequencies = np.linspace(0, sample_rate // 2, mel_np.shape[1])
    
        for frame_idx in range(mel_np.shape[0]):
            frame_start = frame_idx * hop_length
            frame_end = min(frame_start + hop_length, audio_length)
        
            for freq_bin in range(mel_np.shape[1]):
                magnitude = mel_np[frame_idx, freq_bin]
                frequency = mel_frequencies[freq_bin]
                
                if magnitude > -40:  # 閾値以上の成分のみ使用
                    t = np.arange(frame_end - frame_start) / sample_rate
                    phase = np.random.random() * 2 * np.pi  # ランダム位相
                
                    # サイン波生成（振幅を制限）
                    amplitude = min(np.exp(magnitude / 20), 0.1)  # 振幅制限
                    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                
                    audio[frame_start:frame_end] += sine_wave
    
        # 音声を正規化（重要！）
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8  # 最大振幅を0.8に制限
    
        return torch.from_numpy(audio.astype(np.float32))

    def _improved_vocoder(self, mel_spec):
        """改善されたボコーダー（長さ制御を改善）"""
        import torch
        import numpy as np

        try:
            # メルスペクトログラムの形状確認
            if mel_spec.dim() == 2 and mel_spec.shape[1] == 80:
                # (time, mel) 形式の場合
                mel_np = mel_spec.numpy()
            else:
                # (mel, time) 形式の場合
                mel_np = mel_spec.T.numpy()

            print(f"改善ボコーダー入力: {mel_np.shape}")

            sample_rate = self.preprocessor.sample_rate
            hop_length = 256
            
            # 音声長を計算
            audio_length = mel_np.shape[0] * hop_length
            audio = np.zeros(audio_length)

            print(f"予想音声長: {audio_length} samples ({audio_length/sample_rate:.2f}秒)")

            # 基本周波数の推定（改善）
            f0 = np.zeros(mel_np.shape[0])
            for i in range(mel_np.shape[0]):
                # 低域から中域の最大値から基本周波数を推定
                low_freq_power = mel_np[i, :30]  # 低域-中域成分
                if np.max(low_freq_power) > -35:  # 閾値を調整
                    # より自然な基本周波数範囲
                    f0[i] = 80 + np.argmax(low_freq_power) * 8  # 80-320Hz範囲
                    
                    # 基本周波数の平滑化
                    if i > 0 and f0[i-1] > 0:
                        # 前のフレームとの差が大きすぎる場合は平滑化
                        if abs(f0[i] - f0[i-1]) > 50:
                            f0[i] = (f0[i] + f0[i-1]) / 2
                else:
                    f0[i] = 0  # 無音

            # より豊かな音声合成
            for frame_idx in range(mel_np.shape[0]):
                frame_start = frame_idx * hop_length
                frame_end = min(frame_start + hop_length, audio_length)
                
                if f0[frame_idx] > 0:  # 有音フレーム
                    t = np.arange(frame_end - frame_start) / sample_rate
                    
                    # 基本波とその調波を生成（より多くの調波）
                    for harmonic in range(1, 12):  # 1-11次調波
                        freq = f0[frame_idx] * harmonic
                        if freq < sample_rate / 2:  # ナイキスト周波数以下
                            freq_bin = min(int(freq * mel_np.shape[1] * 2 / sample_rate), mel_np.shape[1] - 1)
                            magnitude = mel_np[frame_idx, freq_bin]
                            
                            if magnitude > -45:  # 閾値を緩和
                                # 調波の振幅（高次ほど減衰、但し自然な減衰カーブ）
                                amplitude = np.exp(magnitude / 25) / (harmonic ** 0.7) * 0.15
                                
                                # 位相の連続性を考慮
                                phase = np.random.random() * 2 * np.pi
                                if frame_idx > 0:  # 前のフレームとの位相連続性
                                    phase *= 0.5  # 位相変化を抑制
                                
                                sine_wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
                                audio[frame_start:frame_end] += sine_wave
                
                # 無音フレームでもわずかなノイズを追加（自然さのため）
                noise_level = np.mean(np.abs(mel_np[frame_idx, 40:])) / 200
                if noise_level > 0:
                    noise = np.random.randn(frame_end - frame_start) * noise_level
                    audio[frame_start:frame_end] += noise

            # 音声を正規化
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.7

            # エンベロープの適用（より自然な音声）
            envelope_length = len(audio)
            fade_in = int(envelope_length * 0.05)  # 5%フェードイン
            fade_out = int(envelope_length * 0.05)  # 5%フェードアウト
            
            # フェードイン
            if fade_in > 0:
                audio[:fade_in] *= np.linspace(0, 1, fade_in)
            # フェードアウト
            if fade_out > 0:
                audio[-fade_out:] *= np.linspace(1, 0, fade_out)
            
            return torch.from_numpy(audio.astype(np.float32))
            
        except Exception as e:  # ← この行のインデントを修正（tryと同じレベル）
            print(f"改善ボコーダーエラー: {e}")
            import traceback
            traceback.print_exc()
            # フォールバック: 簡易ボコーダーを使用
            return self._simple_vocoder(mel_spec)

    def _neural_vocoder(self, mel_spec):
        """ニューラルネットワークベースのボコーダー（改善版）"""
        import torch
        import numpy as np
        
        try:
            # メルスペクトログラムの前処理
            if mel_spec.dim() == 2 and mel_spec.shape[1] == 80:
                mel_np = mel_spec.numpy()
            else:
                mel_np = mel_spec.T.numpy()
        
            print(f"高品質ボコーダー入力: {mel_np.shape}")
        
            sample_rate = self.preprocessor.sample_rate
            hop_length = 256
            window_length = 1024
            
            # 音声長を計算
            audio_length = mel_np.shape[0] * hop_length
            audio = np.zeros(audio_length)
        
            # メルフィルターバンクの周波数
            mel_frequencies = np.linspace(0, sample_rate // 2, mel_np.shape[1])
        
            # 高品質音声合成
            for frame_idx in range(mel_np.shape[0]):
                frame_start = frame_idx * hop_length
                frame_end = min(frame_start + hop_length, audio_length)
                frame_length = frame_end - frame_start
            
                if frame_length <= 0:
                    continue
            
                # 時間軸
                t = np.arange(frame_length) / sample_rate
                frame_audio = np.zeros(frame_length)
            
                # 各周波数帯域の処理
                for freq_idx in range(mel_np.shape[1]):
                    magnitude = mel_np[frame_idx, freq_idx]
                
                    if magnitude > -45:  # 閾値以上の成分のみ
                        frequency = mel_frequencies[freq_idx]
                        
                        # 振幅の計算（対数スケールから線形スケールへ）
                        amplitude = np.exp((magnitude + 30) / 10) * 0.01
                        
                        # 基本周波数の推定（低域の最大値から）
                        if freq_idx < 20:  # 低域
                            fundamental_freq = 100 + freq_idx * 10  # 100-290Hz
                        else:
                            fundamental_freq = frequency
                        
                        # より自然な波形生成
                        if freq_idx < 30:  # 低域-中域：調波構造
                            # 基本波
                            phase = np.random.random() * 2 * np.pi
                            wave = amplitude * np.sin(2 * np.pi * fundamental_freq * t + phase)
                            
                            # 調波を追加（自然な音声の特徴）
                            for harmonic in range(2, 6):
                                harmonic_freq = fundamental_freq * harmonic
                                if harmonic_freq < sample_rate / 2:
                                    harmonic_amp = amplitude / (harmonic ** 1.5)
                                    harmonic_phase = phase + np.random.random() * np.pi
                                    wave += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t + harmonic_phase)
                        
                            frame_audio += wave
                            
                        else:  # 高域：ノイズ成分
                            # 帯域制限ノイズ
                            noise = np.random.randn(frame_length)
                            
                            # バンドパスフィルタの近似
                            freq_low = frequency - 50
                            freq_high = frequency + 50
                            
                            # 簡易フィルタリング
                            filtered_noise = noise * amplitude * 0.3
                            frame_audio += filtered_noise
            
                # フレーム音声を全体に追加
                audio[frame_start:frame_end] += frame_audio
            
                # フレーム間の平滑化
                if frame_idx > 0:
                    overlap_start = max(0, frame_start - hop_length // 4)
                    overlap_end = frame_start + hop_length // 4
                    if overlap_end <= len(audio):
                        # クロスフェード
                        fade_length = min(hop_length // 4, len(audio) - overlap_start)
                        if fade_length > 0:
                            fade_in = np.linspace(0.5, 1.0, fade_length)
                            audio[overlap_start:overlap_start + fade_length] *= fade_in
        
            # 後処理
            if np.max(np.abs(audio)) > 0:
                # 正規化
                audio = audio / np.max(np.abs(audio)) * 0.8
                
                # 低域フィルタ（ノイズ除去）
                from scipy import signal
                # 8kHz以上をカット
                nyquist = sample_rate / 2
                cutoff = 8000 / nyquist
                b, a = signal.butter(4, cutoff, btype='low')
                audio = signal.filtfilt(b, a, audio)
                
                # 再正規化
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.7
        
            # エンベロープ処理
            envelope_length = len(audio)
            fade_samples = int(envelope_length * 0.02)  # 2%フェード
        
            if fade_samples > 0:
                # フェードイン
                audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # フェードアウト
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
            print(f"高品質音声生成完了: {len(audio)} samples")
            return torch.from_numpy(audio.astype(np.float32))
        
        except Exception as e:
            print(f"高品質ボコーダーエラー: {e}")
            import traceback
            traceback.print_exc()
            # フォールバック: 簡易ボコーダーを使用
            return self._simple_vocoder(mel_spec)

def synthesize_speech(self, text, output_filename=None):
    """音声合成（高品質ボコーダー使用）"""
    if self.model is None:
        print("❌ モデルが読み込まれていません。先にモデルを読み込んでください。")
        return
    
    try:
        # テキスト前処理
        text_sequence = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([len(text_sequence)]).to(self.device)
        
        print(f"テキスト長: {len(text_sequence)} → 期待される音声長: {len(text_sequence) * 0.1:.1f}秒")
        
        # モデル推論
        self.model.eval()
        with torch.no_grad():
            mel_outputs, stop_outputs = self.model(text_tensor, text_lengths)
        
        mel_spec = mel_outputs.squeeze(0).cpu()
        
        # 高品質ボコーダーを使用
        print("高品質音声合成中...")
        audio = self._neural_vocoder(mel_spec)
        
        # 音声保存
        if output_filename is None:
            output_filename = "synthesized_speech_hq.wav"
        
        output_path = os.path.join(self.output_path, output_filename)
        torchaudio.save(output_path, audio.unsqueeze(0), self.preprocessor.sample_rate)
        
        print(f"✓ 高品質音声合成完了! 出力ファイル: {output_path}")
        
    except Exception as e:
        print(f"❌ 音声合成エラー: {e}")
        import traceback
        traceback.print_exc()