import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SimpleVoiceCloneModel(nn.Module):
    """シンプルな音声クローニングモデル"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 256, 
                 mel_bins: int = 80):
        super(SimpleVoiceCloneModel, self).__init__()
        
        # テキストエンコーダ
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_lstm = nn.LSTM(embedding_dim, hidden_size, 1, batch_first=True)
        
        # デコーダ
        self.prenet = nn.Sequential(
            nn.Linear(mel_bins, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.decoder_lstm = nn.LSTM(hidden_size * 2, hidden_size, 2, batch_first=True)
        
        # 出力層
        self.mel_projection = nn.Linear(hidden_size, mel_bins)
        self.stop_projection = nn.Linear(hidden_size, 1)
        
        # アテンション（簡易版）
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor, 
                target_mels: Optional[torch.Tensor] = None, max_length: int = 1000):
        batch_size = text.size(0)
        device = text.device
        
        # テキストエンコーディング
        embedded = self.embedding(text)
        text_output, _ = self.text_lstm(embedded)
        
        # デコーダ
        if target_mels is not None:
            # 訓練時
            max_decoder_steps = target_mels.size(1)
            mel_input = torch.cat([torch.zeros(batch_size, 1, target_mels.size(2), device=device), 
                                 target_mels[:, :-1]], dim=1)
        else:
            # 推論時
            max_decoder_steps = max_length
            mel_input = torch.zeros(batch_size, 1, 80, device=device)
        
        mel_outputs = []
        stop_outputs = []
        hidden = None
        
        for i in range(max_decoder_steps):
            if target_mels is not None:
                # 訓練時：正解データを使用
                current_mel = mel_input[:, i:i+1]
            else:
                # 推論時：前の出力を使用
                if i == 0:
                    current_mel = torch.zeros(batch_size, 1, 80, device=device)
                else:
                    current_mel = mel_outputs[-1].unsqueeze(1)
            
            # Prenet
            prenet_out = self.prenet(current_mel.squeeze(1))
            
            # 簡易アテンション（全体の平均）
            context = torch.mean(text_output, dim=1)
            
            # デコーダ入力
            decoder_input = torch.cat([prenet_out, context], dim=-1).unsqueeze(1)
            decoder_output, hidden = self.decoder_lstm(decoder_input, hidden)
            
            # 出力
            mel_out = self.mel_projection(decoder_output.squeeze(1))
            stop_out = torch.sigmoid(self.stop_projection(decoder_output.squeeze(1)))
            
            mel_outputs.append(mel_out)
            stop_outputs.append(stop_out)
            
            # 推論時の停止判定
            if target_mels is None and stop_out.mean() > 0.5:
                break
        
        mel_outputs = torch.stack(mel_outputs, dim=1)
        stop_outputs = torch.stack(stop_outputs, dim=1)
        
        return mel_outputs, stop_outputs

# エイリアス（後方互換性）
VoiceCloneModel = SimpleVoiceCloneModel