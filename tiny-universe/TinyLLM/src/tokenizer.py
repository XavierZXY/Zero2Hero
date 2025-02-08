import os
import struct
from typing import List

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "../data/tok4096.model"


class Tokenizer:
    def __init__(self, tokenizer_model=None):
        """初始化分词器，加载预训练的SentencePiece模型

        Args:
            tokenizer_model (_type_, optional): _description_. Defaults to None.
        """
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.exists(model_path), (
            f"Tokenizer model {model_path} not found"
        )

        # 加载 SentencePiece 模型
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path
        # 获取分词器的特殊token以及词汇表的大小
        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()  # 句子的开始
        self.eos_id = self.sp_model.eos_id()  # 句子的结束
        self.pad_id = self.sp_model.pad_id()  # 填充
        # 验证分词器的词汇表大小是否正确
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size(), (
            "Vocab size mismatch"
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """将字符串编码为词元

        Args:
            s (str): _description_
            bos (bool): _description_
            eos (bool): _description_

        Returns:
            List[int]: _description_
        """
        assert isinstance(s, str), "Input must be a string"
        # 将字符串分词为词元
        t = self.sp_model.encode(s)
        # 添加特殊token
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, tokens: List[int]) -> str:
        """decode

        Args:
            tokens (List[int]): _description_

        Returns:
            str: _description_
        """
        return self.sp_model.decode(tokens)


if __name__ == "__main__":
    # 测试 Tokenizer
    enc = Tokenizer(TOKENIZER_MODEL)  # 加载分词器
    text = "IU-2025"  # 测试文本
    print(enc.encode(text, bos=True, eos=True))  # 编码文本
    print(enc.decode(enc.encode(text, bos=True, eos=True)))  # 解码文本
