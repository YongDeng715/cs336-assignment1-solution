# copy from: https://github.com/datawhalechina/diy-llm/blob/main/coursework/assignment1-basics/CS336_Assignment1_BPE.ipynb
import os
import time
import psutil # 监控计算资源占用情况

from typing import BinaryIO
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC


# %% 分块读取工具函数

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分为可以独立计数的块（Chunk）。
    如果边界发生重叠（例如文件太小或分隔符太稀疏），返回的块数量可能会少于预期。
    """
    # 确保传入的分隔符是字节串类型，因为文件是以二进制模式读取的
    assert isinstance(split_special_token, bytes), "必须使用字节串（bytes）表示特殊 Token"

    # 获取文件的总字节大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0) # 回到文件开头

    # 计算初步的理想块大小（字节数）
    chunk_size = file_size // desired_num_chunks

    # 初始的边界猜测：根据块大小进行均匀分布
    # 边界数组包含起始位置 0 和结束位置 file_size
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size # 确保最后一个边界精确指向文件末尾

    mini_chunk_size = 4096  # 每次向后搜索的缓冲区大小4k字节

    # 遍历除了开头和结尾之外的所有中间边界点
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 跳转到初步猜测的边界位置
        
        while True:
            # 读取一小块数据进行扫描
            mini_chunk = file.read(mini_chunk_size)

            # 如果读到了文件末尾（EOF），说明后面没有分隔符了，直接设为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在当前小块数据中查找指定的分隔符
            found_at = mini_chunk.find(split_special_token)
            
            if found_at != -1:
                # 如果找到了分隔符，将边界调整到该分隔符的确切位置
                chunk_boundaries[bi] = initial_position + found_at
                break
            
            # 如果没找到，继续向后移动指针进行下一轮搜索
            initial_position += mini_chunk_size

    # set()：去除重复的边界，防止多个猜测点指向同一个分隔符
    # sorted()：确保边界按从小到大的顺序排列
    return sorted(set(chunk_boundaries))


#%%  按块读取文本
def iter_text_chunks_with_monitor(
    file_path: str,
    chunk_size: int = 1_000_000,  # 1MB
    log_every: int = 5,          # 每N个chunk打印一次
):
    start_time = time.time()
    bytes_processed = 0
    chunk_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        buffer = []
        buffer_size = 0

        for line in f:
            buffer.append(line)
            buffer_size += len(line)
            bytes_processed += len(line)

            if buffer_size >= chunk_size:
                yield "".join(buffer)
                buffer = []
                buffer_size = 0
                chunk_count += 1

                if chunk_count % log_every == 0:
                    log_status(
                        prefix="📘 分词器流式处理",
                        bytes_processed=bytes_processed,
                        start_time=start_time,
                    )

        if buffer:
            yield "".join(buffer)

# Step 1: 在不修改tokenizer内部实现的前提下，实时监控内存占用与数据吞吐量，理解tokenizer训练的真实系统行为。
def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 日志状态函数输出（内存占用、处理数据量、处理速度）
def log_status(prefix, bytes_processed, start_time):
    elapsed = time.time() - start_time  
    mb = bytes_processed / 1024 / 1024
    throughput = mb / elapsed if elapsed > 0 else 0.0 # 计算吞吐量即每秒处理的数据量
    mem = get_memory_mb() 

    print(
        f"{prefix} | "
        f"mem={mem:7.1f} MB | "
        f"data={mb:8.1f} MB | "
        f"speed={throughput:6.2f} MB/s"
    )
    
    
# %% Step 2: 训练BPE Tokenizer
def train_bpe_tokenizer(
    train_file: str,
    val_file: str | None = None,
    vocab_size: int = 50257,
    num_chunks: int = 8,
    output_dir: str = "./bpe_tokenizer",
):
    os.makedirs(output_dir, exist_ok=True)

    special_tokens = [
        "<|endoftext|>",
        "<|unk|>",
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
    ]

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = NFKC()

    # 设计GPT-2风格的BPE Tokenizer
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True, 
    )

    def text_iterator():
        # 训练集
        for chunk in iter_text_chunks_with_monitor(
            train_file,
            chunk_size=1_000_000,    # 每次处理大小约为1MB的原始数据，处理完就会清除数据方便继续处理
            log_every=20,
        ):
            yield chunk

        # 验证集（可选）
        if val_file is not None:
            for chunk in iter_text_chunks_with_monitor(
                val_file,
                chunk_size=1_000_000,
                log_every=10,
            ):
                yield chunk


    print("🚀 开始训练BPE Tokenizer...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    print("✅ BPE Tokenizer训练完成")

    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"💾 分词器已保存至{output_dir}/tokenizer.json")

    return tokenizer

# 验证训练的BPE Tokenizer
def test_main(tokenizer: Tokenizer):
    encoded = tokenizer.encode(" Hello, world! <|endoftext|>")
    print(encoded.tokens)    # 打印编码后的token序列
    print(encoded.ids)       # 打印编码后的ID序列
    # print(tokenizer.decode([1501])) # 输出应该是" world"（前面带个空格）

    # 将Ġ替换回空格
    clean_tokens = [t.replace('Ġ', ' ') for t in encoded.tokens]
    print(clean_tokens)

def train_main():
    train_path = "./data/TinyStoriesV2-GPt4-train.txt"
    val_path = "./data/TinyStoriesV2-GPt4-valid.txt"
    tokenizer = train_bpe_tokenizer(
        train_file=train_path,
        val_file=val_path,
        vocab_size=50257,     # 词表大小（通常设为32000或50257等）
        num_chunks=16,        # 读取文件时的分块数量（建议根据内存大小调整）
        output_dir="./bpe_tokenizer",
    )
    return tokenizer

# Token统计函数
def analyze_tokenizer(tokenizer, texts):
    lengths = [len(tokenizer.encode(t).ids) for t in texts]
    return {
        "avg_tokens": sum(lengths) / len(lengths), # 平均处理token数
        "max_tokens": max(lengths),                # 最大token数，用于设置最大处理序列长度（决定是否截断序列处理）
    }

import random  
def count_tokens(tokenizer, file_path, num_samples=10, name="Train"):
    def load_stories(file_path, num_samples=None):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        stories = [s.strip() for s in content.split("<|endoftext|>") if s.strip()]
        
        if num_samples is not None:
            n = min(num_samples, len(stories))
            stories = random.sample(stories, k=n)
        return stories

    test_texts = load_stories(file_path, num_samples=num_samples)
    print(f"随机抽取的 {name}样本数: {len(test_texts)}")

    # 分析训练集和验证集的token统计
    token_stats = analyze_tokenizer(tokenizer, test_texts)
    print(f"{name} 集统计:", token_stats)
    
#%%     
if __name__ == "__main__":
    tokenizer = train_main()
    
    test_main(tokenizer)
    
    count_tokens(tokenizer, "./data/TinyStoriesV2-GPt4-train.txt", num_samples=10, name="Train")
    count_tokens(tokenizer, "./data/TinyStoriesV2-GPt4-valid.txt", num_samples=10, name="Val")