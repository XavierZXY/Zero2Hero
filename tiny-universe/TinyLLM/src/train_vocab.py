import argparse
import glob
import json
import os

import requests
import sentencepiece as spm
from tqdm import tqdm

DATA_CACHE_DIR = "../data"


def download_file(url: str, fname: str, chunk_size: int = 1024):
    """发送HTTP GET请求以流失方式获取文件

    Args:
        url (str): url地址
        fname (str): 文件名
        chunk_size (int, optional): chunk size. Defaults to 1024.
    """
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        # 获取文件大小
        total = int(resp.headers.get("content-length", 0))
        # 以二进制模型打开一个文件来保存下载内容
        with (
            open(fname, "wb") as file,
            tqdm(
                desc=fname,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)


def download():
    """执行download"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # 下载数据集
    data_url = "https://www.modelscope.cn/datasets/AI-ModelScope/TinyStories/resolve/master/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    # 检查文件是否存在
    if not os.path.exists(data_filename):
        download_file(data_url, data_filename)
    else:
        print(f"File {data_filename} already exists!")

    # 解压数据集
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)  # 创建数据目录
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xvf {data_filename} -C {data_dir}")
    else:
        print(f"Directory {data_dir} already exists!")

    # 查找解压后的所有json文件
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    # 打开第一个json文件
    with open(shard_filenames[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    # 下载完成信息
    print("Download completed!")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"First shard: {data[0]}")


def load_text_from_files(path):
    """加载文本文件

    Args:
        path (_type_): _description_
    """
    path_list = glob.glob(path)
    text_data = []
    for file_path in path_list:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data.extend(f.readlines())
    return text_data


def batch_iterator(text_data, batch_size=648):
    """批量迭代器

    Args:
        text_data (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 648.
    """
    for i in range(0, len(text_data), batch_size):
        yield text_data[i : i + batch_size]


def train_vocab(vocab_size: int = 32000, num_shards: int = 20):
    """训练词表

    Args:
        vocab_size (int, optional): Defaults to 32000.
        num_shards (int, optional): 用于加快词汇表训练的效率，指定要处理的分片数量. Defaults to 20.
    """
    assert vocab_size > 0, "vocab_size must be greater than 0"

    # SentencePice 模型的前缀路径，将用于保存分词器
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # 1> 将多个分片中的文本导出为单个文本文件 tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # 创建tiny.txt文件，并写入指定数量的分片中的文本
    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    if not os.path.exists(tiny_file):
        with open(tiny_file, "w", encoding="utf-8") as f:
            for shard in tqdm(shard_filenames[:num_shards]):
                with open(shard, "r", encoding="utf-8") as f_shard:
                    data = json.load(f_shard)
                    for exapmle in data:
                        text = exapmle["story"]
                        text = text.strip()
                        f.write(text + "\n")

    # 输出tiny.txt文件的大小
    print(
        f"Temporary file {tiny_file} has {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB"
    )

    # 2> 训练 SentencePiece 模型
    print(f"Training SentencePiece model with {vocab_size} tokens...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,  # tiny.txt
        model_prefix=prefix,  # 保存模型的前缀
        model_type="bpe",  # 使用BPE算法
        vocab_size=vocab_size,  # 词表大小
        self_test_sample_size=0,  # 自我测试样本大小
        input_format="text",  # 输入格式
        character_coverage=1.0,  # 字符覆盖率
        num_threads=8,  # 线程数
        split_digits=True,  # 分割数字
        allow_whitespace_only_pieces=True,  # 是否允许空白字符
        byte_fallback=True,  # 字节回退
        unk_surface=r" \342\201\207",  # 未知标记
        normalization_rule_name="identity",  # 规范化规则名称
    )

    # 3> 删除tiny.txt文件
    dec = input(f"Delete temporary file {tiny_file}? [y/n]")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    # 4> 输出模型文件的位置
    print(f"Model file is saved at {prefix}.model \n DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece model")
    parser.add_argument(
        "--download",
        type=bool,
        default=True,
        help="Download the TinyStories dataset",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=4096,
        help="Vocabulary size for the SentencePiece model",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=20,
        help="Number of shards to use for training the SentencePiece model",
    )

    args = parser.parse_args()

    if args.download:
        download()
    train_vocab(args.vocab_size, args.num_shards)
