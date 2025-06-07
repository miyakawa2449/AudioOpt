# AudioOpt - 音声クローニングシステム

PyTorchとTorchaudioを使用したエンドツーエンドの音声クローニング・音声合成システムです。

## 🎯 概要

AudioOptは、少量の音声データから音声クローニングを行うことができるPythonベースのディープラーニングシステムです。テキストから音声を合成し、特定の話者の声質を再現することを目指しています。

## ✨ 特徴

- **エンドツーエンドの音声合成**: テキストから直接音声を生成
- **音声前処理**: ノイズ除去、無音部分除去、正規化
- **データ拡張**: 速度変更、ピッチ変更による学習データの拡張
- **複数の音声合成手法**: Griffin-Lim、改善されたボコーダー、簡易ボコーダー
- **対話式インターフェース**: メニュー形式での簡単操作
- **GPU対応**: CUDAによる高速学習・推論
- **モジュラー設計**: 各コンポーネントが独立して動作

## 🛠️ システム要件

### 必須環境
- Python 3.8+
- CUDA対応GPU（推奨）
- 4GB以上のRAM

### 依存関係
```bash
torch>=1.12.0
torchaudio>=0.12.0
numpy>=1.20.0
soundfile>=0.10.0
matplotlib>=3.5.0
```

## 📦 インストール
1. リポジトリのクローン

```
git clone https://github.com/miyakawa2449/AudioOpt.git
cd AudioOpt
```
2. Conda環境の作成
```
conda create -n voice-clone python=3.10
conda activate voice-clone
```
3. 依存関係のインストール
```
# PyTorchとTorchaudio（CUDA対応）
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# その他の依存関係
conda install numpy matplotlib soundfile -c conda-forge
```
## 📁 プロジェクト構造
```
AudioOpt/
├── src/
│   ├── audio/
│   │   └── preprocessor.py      # 音声前処理
│   ├── text/
│   │   └── text_processor.py    # テキスト処理
│   ├── model/
│   │   ├── dataset.py           # データセット処理
│   │   └── voice_clone_model.py # 音声クローニングモデル
│   └── core/
│       └── voice_cloner.py      # メインシステム
├── dataset/
│   ├── audio_files/             # 音声ファイル (.wav)
│   ├── meta_files/              # テキストファイル (.txt)
│   └── processed/               # 前処理済みデータ
├── models/                      # 訓練済みモデル
├── output/                      # 合成音声出力
├── main.py                      # メインプログラム
└── README.md
```

## 🚀 使用方法
1. データの準備
音声ファイルとテキストファイルを対応する形で配置：
```
dataset/audio_files/audio_1.wav
dataset/audio_files/audio_2.wav
...

dataset/meta_files/meta_1.txt
dataset/meta_files/meta_2.txt
...
```
重要: ファイル名は audio_N.wav と meta_N.txt の形式で、同じ番号Nを使用してください。

2. システムの起動
```
python main.py
```
3. 基本的な使用フロー
   1. データセットの確認 (メニュー 5)
   2. モデルの訓練 (メニュー 1)
   3. 音声合成 (メニュー 3)


🎵 音声合成の例
```
# プログラム内での使用例
cloner = VoiceCloner()
cloner.load_model("models/voice_clone_model.pth")
cloner.synthesize_speech("こんにちは、世界！", "output/hello_world.wav")
```

## ⚙️ 設定とカスタマイズ
### 学習パラメータ
- エポック数: 50-200 (データ量による)
- バッチサイズ: 1-4 (GPU メモリによる)
- 学習率: 0.0001-0.001

### 音声パラメータ
- サンプルレート: 22050 Hz
- メルスペクトログラム: 80 bins
- STFT設定: n_fft=2048, hop_length=512

## 🔧 高度な機能

### データ拡張
```
# データ拡張の実行
cloner.augment_dataset()
```

### 改善された音声合成
```
# より高品質な音声合成
cloner.synthesize_speech_improved("テキスト", "output/improved.wav")
```

### ボコーダーベース合成
```
# ボコーダーを使用した合成
cloner.synthesize_speech_with_vocoder("テキスト", "output/vocoder.wav")
```

## 📊 パフォーマンス
### 推奨データ量
- 最低: 17+ サンプル (基本動作確認)
- 実用: 100-500 サンプル
- 高品質: 1000+ サンプル

### 学習時間（例）
- 100サンプル: 約10-30分 (GPU)
- 500サンプル: 約30-60分 (GPU)

## 🐛 トラブルシューティング
よくある問題

1. ライブラリインポートエラー
```
#　依存関係の再確認
python -c "import torch, torchaudio, soundfile; print('All dependencies OK')"
```
2. CUDA関連エラー
```
# CPU版での実行
export CUDA_VISIBLE_DEVICES=""
python main.py
```

3. 音声ファイル読み込みエラー
- ファイル形式がWAVであることを確認
- ファイル名が正しい命名規則に従っていることを確認

4. 合成音声がノイズのみ
- 学習データが不足している可能性
- より多くのデータで再学習を実行

## 🛣️ 今後の計画
- <input disabled="" type="checkbox"> より高品質なボコーダーの実装
- <input disabled="" type="checkbox"> リアルタイム音声合成
- <input disabled="" type="checkbox"> 多話者対応
- <input disabled="" type="checkbox"> Web インターフェース
- <input disabled="" type="checkbox"> 音声品質評価指標の追加

## 🤝 コントリビューション
1. このリポジトリをフォーク
2. 新しいブランチを作成 (git checkout -b feature/amazing-feature)
3. 変更をコミット (git commit -m 'Add amazing feature')
4. ブランチにプッシュ (git push origin feature/amazing-feature)
5. プルリクエストを作成

## 📄 ライセンス
このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞
- PyTorchチーム
- Torchaudioチーム
- 音声処理コミュニティ

📞 お問い合わせ
Email: t.miyakawa244@gmail.com
⭐ このプロジェクトが役に立った場合は、スターを付けていただけると嬉しいです！

---

# 技術詳細

## 1. アーキテクチャ

* モデル構造

  * エンコーダ: LSTM ベースのテキストエンコーダ
  * デコーダ: Attention付きLSTMデコーダ
  * 出力: 80次元メルスペクトログラム

## 2. 音声合成パイプライン

1. テキスト → 数値序列変換
2. モデル推論 → メルスペクトログラム生成
3. ボコーダー → 音声波形生成

## 3. パフォーマンス最適化

* GPU使用時

  * バッチサイズ: 2～4 推奨
  * メモリ使用量: 約2～4GB
  * 学習速度: 100エポックあたり約15～30分
* CPU使用時

  * バッチサイズ: 1 推奨
  * 学習速度: 100エポックあたり約1～2時間

## 4. データ前処理詳細

* 音声前処理

  * サンプリングレート正規化: 22050Hz
  * ノイズ除去: スペクトル減算法
  * 無音除去: RMSベース検出
  * 正規化: RMS正規化
* テキスト前処理

  * 文字正規化
  * 特殊文字処理
  * 語彙構築

---

この形式ならエディタでもMarkdownとしてきれいに見えます。
さらにシンプルや、各項目をより詳細に分けたい場合も調整できますのでご希望があれば教えてください。
