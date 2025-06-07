import re
from typing import List, Dict

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
    
    def build_vocab(self, texts: List[str]) -> None:
        """語彙を構築"""
        for text in texts:
            self.vocab.update(text.lower())
        
        # 特殊トークンを追加
        self.vocab.add('<PAD>')
        self.vocab.add('<START>')
        self.vocab.add('<END>')
        self.vocab.add('<UNK>')
        
        # 文字とインデックスのマッピングを作成
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(self.vocab))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        print(f"Built vocabulary with {len(self.vocab)} characters")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """テキストを数値序列に変換"""
        text = self.clean_text(text.lower())
        sequence = [self.char_to_idx.get('<START>', 0)]
        
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0))
            sequence.append(idx)
        
        sequence.append(self.char_to_idx.get('<END>', 0))
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """数値序列をテキストに変換"""
        chars = []
        for idx in sequence:
            char = self.idx_to_char.get(idx, '<UNK>')
            if char in ['<START>', '<END>', '<PAD>']:
                continue
            if char == '<UNK>':
                char = '?'
            chars.append(char)
        
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        """語彙サイズを取得"""
        return len(self.vocab)
    
    def save_vocab(self, filepath: str) -> None:
        """語彙を保存"""
        import json
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
            'vocab': list(self.vocab)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to: {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """語彙を読み込み"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        self.vocab = set(vocab_data['vocab'])
        
        print(f"Vocabulary loaded from: {filepath}")