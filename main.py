"""
音声クローニングシステム - メインプログラム
"""

import sys
import os
import traceback

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core import VoiceCloner
except ImportError as e:
    print(f"モジュールのインポートに失敗しました: {e}")
    print("以下を確認してください：")
    print("1. srcディレクトリが存在するか")
    print("2. 必要なファイルが全て配置されているか")
    print("3. 必要なライブラリがインストールされているか")
    sys.exit(1)

def check_dependencies():
    """依存関係をチェック"""
    missing_packages = []
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        missing_packages.append("torch")
    
    try:
        import torchaudio
        print(f"✓ Torchaudio: {torchaudio.__version__}")
    except ImportError:
        missing_packages.append("torchaudio")
    
    try:
        import soundfile
        print(f"✓ Soundfile: {soundfile.__version__}")
    except ImportError:
        missing_packages.append("soundfile")
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        missing_packages.append("numpy")
    
    if missing_packages:
        print(f"\n❌ 不足しているパッケージ: {', '.join(missing_packages)}")
        print("以下のコマンドでインストールしてください：")
        print(f"conda install {' '.join(missing_packages)}")
        return False
    
    print("✓ 全ての依存関係が満たされています\n")
    return True

def display_system_info(cloner):
    """システム情報を表示"""
    print("\n=== システム情報 ===")
    print(f"Dataset path: {cloner.dataset_path}")
    print(f"Audio files path: {cloner.audio_path}")
    print(f"Text files path: {cloner.meta_path}")
    print(f"Models path: {cloner.models_path}")
    print(f"Output path: {cloner.output_path}")
    print(f"Device: {cloner.device}")
    
    # GPU情報を表示
    if cloner.device.type == 'cuda':
        try:
            import torch
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            print(f"GPU Name: {torch.cuda.get_device_properties(0).name}")
        except:
            pass
    
    if hasattr(cloner.text_processor, 'vocab') and cloner.text_processor.vocab:
        print(f"Vocabulary size: {len(cloner.text_processor.vocab)}")
    else:
        print("Vocabulary: Not built yet")
    
    if cloner.model is not None:
        print("Model: Loaded")
        # モデルのパラメータ数を表示
        try:
            total_params = sum(p.numel() for p in cloner.model.parameters())
            print(f"Model parameters: {total_params:,}")
        except:
            pass
    else:
        print("Model: Not loaded")

def train_model_interactive(cloner):
    """対話式でモデル訓練を実行"""
    print("モデル訓練を開始します...")
    
    # データファイルの確認
    audio_files, text_files = cloner.collect_data_files()
    if len(audio_files) == 0:
        print("❌ データファイルが見つかりません。")
        print("先にデータファイルを配置してください。")
        return
    
    print(f"✓ {len(audio_files)}個のデータペアが見つかりました")
    
    # パラメータ入力
    try:
        epochs_input = input("エポック数を入力 (デフォルト: 100): ").strip()
        epochs = int(epochs_input) if epochs_input else 100
        
        if epochs <= 0:
            print("エポック数は1以上で入力してください")
            return
            
        batch_size_input = input("バッチサイズを入力 (デフォルト: 2): ").strip()
        batch_size = int(batch_size_input) if batch_size_input else 2
        
        if batch_size <= 0 or batch_size > len(audio_files):
            print(f"バッチサイズは1以上{len(audio_files)}以下で入力してください")
            return
        
        learning_rate_input = input("学習率を入力 (デフォルト: 0.001): ").strip()
        learning_rate = float(learning_rate_input) if learning_rate_input else 0.001
        
        print(f"\n訓練設定:")
        print(f"  エポック数: {epochs}")
        print(f"  バッチサイズ: {batch_size}")
        print(f"  学習率: {learning_rate}")
        print(f"  データ数: {len(audio_files)}")
        
        confirm = input("\n訓練を開始しますか？ (y/N): ").strip().lower()
        if confirm != 'y':
            print("訓練をキャンセルしました")
            return
        
        # 訓練実行
        cloner.train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # モデル保存
        save_confirm = input("モデルを保存しますか？ (Y/n): ").strip().lower()
        if save_confirm != 'n':
            cloner.save_model()
            print("✓ モデルが保存されました")
        
    except ValueError as e:
        print(f"❌ 入力値エラー: {e}")
    except Exception as e:
        print(f"❌ 訓練中にエラーが発生しました: {e}")
        traceback.print_exc()

def synthesize_speech_interactive(cloner):
    """対話式で音声合成を実行"""
    if cloner.model is None:
        print("❌ モデルが読み込まれていません。")
        print("先にモデルを読み込むか訓練してください。")
        return
    
    try:
        text = input("合成したいテキストを入力: ").strip()
        if not text:
            print("❌ テキストが入力されませんでした。")
            return
        
        output_path = input("出力ファイル名（空白でdefault）: ").strip()
        if not output_path:
            output_path = None
        
        print(f"音声合成中: '{text}'")
        cloner.synthesize_speech(text, output_path)
        
    except Exception as e:
        print(f"❌ 音声合成エラー: {e}")
        traceback.print_exc()

def display_data_files(cloner):
    """データファイル一覧を表示"""
    audio_files, text_files = cloner.collect_data_files()
    print("\n=== 検出されたデータファイル ===")
    
    if len(audio_files) == 0:
        print("❌ データファイルが見つかりません。")
        print("\n正しいディレクトリ構造:")
        print("dataset/")
        print("├── audio_files/")
        print("│   ├── audio_1.wav")
        print("│   ├── audio_2.wav")
        print("│   └── ...")
        print("└── meta_files/")
        print("    ├── meta_1.txt")
        print("    ├── meta_2.txt")
        print("    └── ...")
        
        print("\n❗ ファイル名は必ず 'audio_N.wav' と 'meta_N.txt' の形式で、")
        print("   Nは同じ番号にしてください。")
    else:
        print(f"✓ {len(audio_files)}個のデータペアが見つかりました:\n")
        for i, (audio, text) in enumerate(zip(audio_files, text_files)):
            print(f"{i+1:2d}. Audio: {os.path.basename(audio)}")
            print(f"     Text:  {os.path.basename(text)}")
            
            # ファイルサイズも表示
            try:
                audio_size = os.path.getsize(audio) / 1024  # KB
                print(f"     Size:  {audio_size:.1f} KB")
            except:
                print(f"     Size:  不明")
            print()

def main():
    """メイン関数"""
    print("=== 音声クローニングシステム ===")
    print("初期化中...")
    
    # 依存関係チェック
    if not check_dependencies():
        return
    
    try:
        # 音声クローナーを初期化
        cloner = VoiceCloner()
        print("✓ システム初期化完了\n")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        traceback.print_exc()
        return
    
    # メインループ
    while True:
        print("=== メニュー ===")
        print("1. データセットの前処理とモデル訓練")
        print("2. 既存モデルの読み込み")
        print("3. 音声合成")
        print("4. 新しいデータの追加")
        print("5. データファイル確認")
        print("6. システム情報表示")
        print("0. 終了")
        
        try:
            choice = input("\n選択してください (0-6): ").strip()
            
            if choice == "0":
                print("システムを終了します。")
                break
                
            elif choice == "1":
                train_model_interactive(cloner)
                
            elif choice == "2":
                # モデル読み込み
                model_path = input("モデルファイルのパス（空白でデフォルト）: ").strip()
                if not model_path:
                    model_path = None
                
                try:
                    cloner.load_model(model_path)
                    print("✓ モデル読み込み完了")
                except FileNotFoundError:
                    print("❌ モデルファイルが見つかりません")
                    if model_path is None:
                        print("デフォルトパス: models/voice_clone_model.pth")
                except Exception as e:
                    print(f"❌ モデル読み込みエラー: {e}")
                    traceback.print_exc()
            
            elif choice == "3":
                synthesize_speech_interactive(cloner)
                
            elif choice == "4":
                # 新しいデータ追加
                print("新しいデータの追加")
                audio_path = input("新しい音声ファイルのパス: ").strip()
                text_path = input("新しいテキストファイルのパス: ").strip()
                
                if not os.path.exists(audio_path):
                    print(f"❌ 音声ファイルが見つかりません: {audio_path}")
                    continue
                    
                if not os.path.exists(text_path):
                    print(f"❌ テキストファイルが見つかりません: {text_path}")
                    continue
                
                try:
                    cloner.add_new_data(audio_path, text_path)
                except Exception as e:
                    print(f"❌ データ追加エラー: {e}")
                    traceback.print_exc()
            
            elif choice == "5":
                display_data_files(cloner)
            
            elif choice == "6":
                display_system_info(cloner)
            
            else:
                print("❌ 無効な選択です。0-6の数字を入力してください。")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  キーボード割り込みが発生しました。システムを終了します。")
            break
        except Exception as e:
            print(f"❌ 予期しないエラーが発生しました: {e}")
            traceback.print_exc()
            print("\n続行しますか？ (y/N): ", end="")
            try:
                if input().lower() != 'y':
                    break
            except KeyboardInterrupt:
                print("\nシステムを終了します。")
                break

if __name__ == "__main__":
    main()