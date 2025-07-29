import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from tokenizer import Tokenizer
from tqdm import tqdm

DATA_CACHE_DIR = "../data"
TOKENIZER_MODEL = "../data/tok4096.model"


def process_shard(args, vocab_size, tokenizer_model):
    """处理数据分片，将其中的文本进行分词并保存为二进制文件

    Args:
        args (_type_): _description_
        vocab_size (_type_): _description_
        tokenizer_model (_type_): _description_
    """
    print(f"Processing shard {args}")
    # 提取分片id和文件名
    shard_id, shard = args
    # 加载分词器
    enc = Tokenizer(tokenizer_model)
    # 读取分片中的文本
    with open(shard, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)

    # 将所有的token转换为uint16类型
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # 根据词汇表大小确定输出文件名
    if vocab_size == 0:
        # 默认使用llama2的词汇表
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

    # 将token保存为二进制文件
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    # 计算平均序列长度
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(
        f"Shard {shard_id} has {all_tokens.size} tokens, avg seq len {avg_seq_len:.2f}"
    )


def pretokenize(vocab_size):
    """预处理所有的数据分片，并将分词后的数据保存为二进制数据文件

    Args:
        vocab_size (_type_): _description_
    """
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    assert len(shard_filenames) > 0, "No data found"
    # 如果词汇表大小大于0，则创建对应的保存目录
    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # 使用partial函数将vocab_size绑定到process_shard函数
    fun = partial(
        process_shard,
        vocab_size=vocab_size,
        tokenizer_model=TOKENIZER_MODEL,
    )

    # 使用进程池并行处理每个分片
    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(fun, enumerate(shard_filenames))

    print("Pretokenization complete")


class PretokDataset(torch.utils.data.IterableDataset):
    """从磁盘中加载已经预处理的分词数据，并将其以pytorch张量的形式返回"""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        """初始化数据集

        Args:
            split (_type_): 数据集的切分方式
            max_seq_len (_type_): 最大序列长度
            vocab_size (_type_): 词表大小
            vocab_source (_type_): 词表来源
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        """迭代器，按批次加载数据并生成模型输入/输出"""
        # 获取dataloader的worker信息，用于并行数据加载
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # 获取分布式训练的rank信息
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 基于worker_id和rank计算随机种子
        seed = 516 + worker_id + rank * 1337
        rng = random.Random(seed)
        print(f"Created a PretokDataset with seed {seed}")

        # 根据词汇表来源决定数据路径
        if self.vocab_source == "llama2":
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.json")))
        elif self.vocab_source == "custom":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # 训练集使用所有分片数据，测试集只使用第一个分片
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        assert len(shard_filenames) > 0, "No data found"

        while True:
            # 打乱分片顺序
            rng.shuffle(shard_filenames)
            for shard_filename in shard_filenames:
                # 使用memmap读取文件，使得数据保留在磁盘上
                m = np.memmap(shard_filename, dtype=np.uint16, mode="r")
                # 计算该分片中的批次数量
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1
                assert num_batches > 0, "Shard is too small"
                # 随机打乱批次索引
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                # 对每个批次生成输入x和目标输出y
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # 将数据转换为numpy数据并拷贝到RAM中
                    chunk = torch.from_numpy(m[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        """返回一个迭代器，用于生成数据批次

        Args:
            batch_size (_type_): _description_
            device (_type_): _description_
            num_workers (int, optional): _description_. Defaults to 0.

        Yields:
            _type_: _description_
        """
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    pretokenize(vocab_size=4096)
