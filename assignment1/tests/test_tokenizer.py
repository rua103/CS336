# =============================================================================
# test_tokenizer.py — BPE Tokenizer 测试套件
# =============================================================================
# 本文件测试你的 Tokenizer 实现是否正确，所有测试都跟 tiktoken(gpt2) 做比对。
# 测试通过 adapters.py 里的 get_tokenizer() 间接调用你的代码 —— 你不需要改这里。
# =============================================================================

# ---------- 1. 导入 ----------

# 启用"延迟注解求值"(PEP 563), 类型注解中的名字可以先不 import, 运行时才解析
from __future__ import annotations

import json         # 读取 GPT-2 的 vocab JSON 文件
import os           # 操作系统接口 (获取进程 ID 等)
import sys          # 系统参数 (判断操作系统平台)

# resource 是 Unix 专属模块, Windows 上没有
try:
    import resource     # Unix 资源限制库 (RLIMIT_AS: 限制进程可用虚拟内存)
except ImportError:
    resource = None     # Windows 上不存在, 设为 None

import psutil       # 跨平台进程/系统监控 (获取当前进程内存使用量)
import pytest       # Python 测试框架
import tiktoken     # OpenAI 官方 tokenizer 库 —— 我们用它做"参考答案"

# adapters.py 里的 get_tokenizer 函数 —— 测试通过它调用 *你的* Tokenizer 实现
from .adapters import get_tokenizer
# common.py 里的辅助: FIXTURES_PATH = 测试数据目录; gpt2_bytes_to_unicode = GPT-2 字节映射表
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

# 测试用的 GPT-2 官方词汇表文件路径 (JSON 格式)
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
# 测试用的 GPT-2 官方合并规则文件路径 (文本格式, 每行一个 merge)
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

# =============================================================================
# 2. 内存限制装饰器
# =============================================================================


def memory_limit(max_mem):
    """装饰器: 限制被装饰函数只能额外使用 max_mem 字节的内存
    max_mem (int): 允许的额外内存上限, 单位是字节

    用法:
        @memory_limit(1_000_000)     # 限制额外 1MB
        def my_func(...): ...
    """
    def decorator(f):                    # f: 要被限制内存的原始函数
        def wrapper(*args, **kwargs):    # wrapper: 实际被调用的包装函数
            # 获取当前进程对象
            process = psutil.Process(os.getpid())
            # 保存当前内存限制, 以便测试后恢复 (不影响其他测试)
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            # RLIMIT_AS = Address Space limit: 进程可用的最大虚拟内存
            # rss = Resident Set Size: 当前进程实际占用的物理内存
            # 新限制 = 当前已用内存 + 允许的额外内存 max_mem
            # -1 表示"硬限制不设上限", 只有软限制生效
            resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))
            try:
                result = f(*args, **kwargs)   # 执行被装饰的函数
                return result
            finally:
                # 无论函数成功还是失败 (包括内存超限崩溃),
                # 都要恢复原来的内存限制, 否则会影响后续测试
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)

        return wrapper

    return decorator

# =============================================================================
# 3. 辅助函数: 从 GPT-2 文件构建 tokenizer 对象
# =============================================================================


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    """
    从 GPT-2 格式的 vocab.json + merges.txt 构建 tokenizer 对象。
    这是所有测试的公共入口 —— 它读取 GPT-2 文件 → 转成原始 bytes 格式
    → 调用你的 get_tokenizer() (在 adapters.py 里)。

    为什么要把 GPT-2 编码转成原始 bytes?
        GPT-2 用了一套特殊的 Unicode 映射来可视化不可打印字节,
        但我们不需要学这套映射, 所以转回原始 bytes 再传给你。
    """
    # gpt2_bytes_to_unicode() 返回 {byte_int: unicode_char} 的映射
    # 这里反转一下: {unicode_char: byte_int}, 用来把 GPT-2 token 转回原始字节
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    # ---- 读取词汇表 ----
    with open(vocab_path, encoding="utf-8") as vocab_f:
        gpt2_vocab = json.load(vocab_f)     # {"token_str": token_id} — GPT-2 的 Unicode 编码形式

    # ---- 读取 BPE 合并规则 ----
    gpt2_bpe_merges = []                     # 将存储 [(token1_str, token2_str), ...]
    with open(merges_path, encoding="utf-8") as f:
        for line in f:                       # 每行格式: "token1 token2"
            cleaned_line = line.rstrip()     # 去掉行尾换行符
            # 合法的 merge 行包含两个 token, 用空格分隔
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

    # ---- 把 GPT-2 的 Unicode 编码转回原始 bytes ----
    # GPT-2 用 Ġ 表示空格, Ā 表示 \x00 等 —— 这里还原为真实的 bytes
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    # ---- 把特殊 token 追加到词汇表 ----
    # 特殊 token (如 <|endoftext|>) 可能在 GPT-2 vocab 里没有,
    # 或者已经在里面但我们要确保它存在
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")  # str → bytes
            # 只有当这个特殊 token 还不在 vocab 里时才追加
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token   # 分配新的 token ID

    # ---- 把 merges 也转回原始 bytes ----
    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    # ---- 调用你的 get_tokenizer (在 adapters.py 里) ----
    return get_tokenizer(vocab, merges, special_tokens)


# =============================================================================
# 4. 测试用例 — Roundtrip 测试 (encode → decode 必须完全还原原文)
# =============================================================================
# "Roundtrip" 意思是: 原文 → encode → 得到 ID 列表 → decode → 还原字符串
# 最基础的测试: 空字符串 / 单字符 / ASCII 句子 / Unicode 句子 / 含特殊 token 的句子


def test_roundtrip_empty():
    """测试空字符串的往返: encode("") 应该得到空列表, decode([]) 应该得到 "" """
    tokenizer = get_tokenizer_from_vocab_merges_path(  # 用 GPT-2 的 vocab + merges 构建 tokenizer
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""                                   # 空字符串
    encoded_ids = tokenizer.encode(test_string)        # 编码: str → list[int]
    decoded_string = tokenizer.decode(encoded_ids)     # 解码: list[int] → str
    assert test_string == decoded_string               # 必须完全还原


def test_empty_matches_tiktoken():
    """测试空字符串的编码结果与 tiktoken 的 GPT-2 编码完全一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")  # 参考答案: OpenAI 的 GPT-2 tokenizer
    tokenizer = get_tokenizer_from_vocab_merges_path(     # 你的实现
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""

    reference_ids = reference_tokenizer.encode(test_string)  # 参考答案的编码结果
    ids = tokenizer.encode(test_string)                      # 你的编码结果
    assert ids == reference_ids                              # 必须一模一样

    # 额外检查: 逐个 token decode, 确保每个 ID 都能单独解码
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == []                        # 空字符串 → 空列表

    # 再确认一次整体 roundtrip
    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# ---------- 单字符测试 ----------

def test_roundtrip_single_character():
    """单 ASCII 字符 's' 的往返测试"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"                                   # 单个 ASCII 字符
    encoded_ids = tokenizer.encode(test_string)         # 编码会得到什么? 可能就是一个 token ID
    decoded_string = tokenizer.decode(encoded_ids)      # 解码回去
    assert test_string == decoded_string                # "s" → encode → decode → "s"


def test_single_character_matches_tiktoken():
    """单字符 's' 需要和 tiktoken 的 gpt2 输出完全一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"

    reference_ids = reference_tokenizer.encode(test_string)   # tiktoken 的结果
    ids = tokenizer.encode(test_string)                       # 你的结果
    assert ids == reference_ids                               # 必须完全匹配

    # 逐个 token 解码: 每个 ID 单独 decode 应该得到对应的文本片段
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["s"]                          # 应该只有 "s" 一个 token

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# ---------- 单 Unicode 字符测试 ----------

def test_roundtrip_single_unicode_character():
    """单个 Unicode 字符 (emoji) 的往返测试 —— 确保非 ASCII 字符也能正确编解码"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"                                   # 一个 emoji (多于 1 个字节)
    encoded_ids = tokenizer.encode(test_string)          # emoji 可能被拆成多个 token
    decoded_string = tokenizer.decode(encoded_ids)       # 但 decode 后必须还原
    assert test_string == decoded_string


def test_single_unicode_character_matches_tiktoken():
    """单 Unicode 字符必须和 tiktoken 结果完全一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids                         # emoji 的编码也要一致

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# ---------- ASCII 多词句子测试 ----------

def test_roundtrip_ascii_string():
    """普通英文句子 roundtrip"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_ascii_string_matches_tiktoken():
    """普通英文句子 vs tiktoken — 注意这个测试没有 assert ids == reference_ids!"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    # assert ids == reference_ids  ← 被注释掉了! 因为这个 case 可能因 special_tokens 而有差异

    # 但逐个 token 解码后必须拆成: ["Hello", ",", " how", " are", " you", "?"]
    # 注意 " how" 保留了前导空格 — 这就是 GPT-2 的预处理方式
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["Hello", ",", " how", " are", " you", "?"]

    # Roundtrip 必须通过
    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# ---------- Unicode 多词句子测试 ----------

def test_roundtrip_unicode_string():
    """带重音字母 + emoji 的 Unicode 句子 roundtrip"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Héllò hôw are ü? 🙃"          # 法语式字母 + emoji
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_matches_tiktoken():
    """Unicode 句子 vs tiktoken — 这个会严格比对 IDs"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids                     # 必须完全匹配

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string

# ---------- 含特殊 token 的字符串测试 ----------

def test_roundtrip_unicode_string_with_special_tokens():
    """测试文本中混入了 <|endoftext|> 特殊 token 时能否正确往返"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    # 逐个 token 解码后检查: 特殊 token 不能被拆散
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # 原文有 3 个 <|endoftext|>: 两个紧挨的 + 末尾一个
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_with_special_tokens_matches_tiktoken():
    """含特殊 token 的 Unicode 句子 vs tiktoken"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    # tiktoken 需要显式声明 allowed_special, 否则会报错 (GPT-2 原生不支持 <|endoftext|>)
    reference_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_overlapping_special_tokens():
    """测试重叠/嵌套的特殊 token — 长匹配优先!
    如果同时有 "<|endoftext|>" 和 "<|endoftext|><|endoftext|>" 两个特殊 token,
    编码器应该优先匹配更长的那个 (贪婪最长匹配).
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    # 两个紧挨的 <|endoftext|><|endoftext|> 应该作为一个整体 token, 不被拆成两个
    assert tokenized_string.count("<|endoftext|>") == 1              # 末尾那一个
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1 # 开头的双 token 整体
    # 往返必须还原
    assert tokenizer.decode(ids) == test_string


# =============================================================================
# 5. 真实文件测试 — 在整个文本文件上验证 encode/decode
# =============================================================================
# 测试逐步升级: address.txt（英文地址）→ german.txt（德语, 含非 ASCII 字符）
# → tinystories_sample.txt（儿童故事, 含特殊 token）

# ---------- address.txt ----------

def test_address_roundtrip():
    """英文地址文件的往返测试"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "address.txt", encoding="utf-8") as f:
        corpus_contents = f.read()                 # 读入整个地址文件

    ids = tokenizer.encode(corpus_contents)        # 编码
    assert tokenizer.decode(ids) == corpus_contents # 解码回来必须完全一致


def test_address_matches_tiktoken():
    """地址文件必须和 tiktoken 输出完全一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")  # tiktoken 参考答案
    tokenizer = get_tokenizer_from_vocab_merges_path(     # 你的实现
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "address.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)  # 参考答案
    ids = tokenizer.encode(corpus_contents)                      # 你的答案
    assert ids == reference_ids                                  # 必须和 gpt2 一模一样

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents

# ---------- german.txt ----------

def test_german_roundtrip():
    """德语文件往返: 含大量非 ASCII 字符 (ö, ü, ß...)"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "german.txt", encoding="utf-8") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_german_matches_tiktoken():
    """德语文件 vs tiktoken — IDs 必须一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "german.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids                     # 德语也要和 gpt2 一致!

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents

# ---------- tinystories_sample.txt ----------

def test_tinystories_sample_roundtrip():
    """TinyStories 样本文件往返 — 这是一个更长的文件 (含特殊 token)"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_tinystories_matches_tiktoken():
    """TinyStories vs tiktoken — 需要处理 <|endoftext|> 特殊 token"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    # tiktoken 默认不认识 <|endoftext|>, 必须通过 allowed_special 放行
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids                     # 含特殊 token 的文本也要一致

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# =============================================================================
# 6. 特殊 token 边界情况测试
# =============================================================================
# 测试特殊 token 后面跟着各种奇奇怪怪的字符时的行为:
# - 换行符
# - 连续换行
# - 非空白字符紧跟在特殊 token 后面


def test_encode_special_token_trailing_newlines():
    """特殊 token 后面跟换行符: <|endoftext|>\n"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_trailing_newlines.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    # tiktoken 不认识 <|endoftext|>, 必须 allowed_special 放行
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids                 # 特殊 token + 换行的组合也要一致

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_double_newline_non_whitespace():
    """特殊 token 跟连续换行 + 非空白字符: <|endoftext|>\n\nabc"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_double_newlines_non_whitespace.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids                 # 必须和 gpt2 结果一致

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# =============================================================================
# 7. encode_iterable 测试 — 流式编码 (边读文件边产出 token, 不一次性加载)
# =============================================================================
# encode_iterable 和 encode 的区别:
#   encode(text):         输入是整个字符串, return 整个 ID 列表 → 内存占用大
#   encode_iterable(f):   输入是文件对象 (迭代器), 逐个 yield token ID → 内存友好
# 核心: encode_iterable 必须在 1MB 内存限制下正常工作, 而 encode 注定会超限!


def test_encode_iterable_tinystories_sample_roundtrip():
    """流式编码的 roundtrip: 读文件 → 逐个编码 → 收集 ID → decode 必须还原"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    all_ids = []                                       # 收集所有的 token ID
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        for _id in tokenizer.encode_iterable(f):       # 逐个 yield, 不一次性算完
            all_ids.append(_id)
    # 重新读文件 (因为 f 已经被消耗完了)
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    assert tokenizer.decode(all_ids) == corpus_contents  # roundtrip 必须通过


def test_encode_iterable_tinystories_matches_tiktoken():
    """流式编码的结果必须和 tiktoken 一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    # 先用传统方式得到参考答案 (一次加载)
    with open(corpus_path, encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    # 流式 encode, 逐个收集
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        for _id in tokenizer.encode_iterable(f):       # 边读边 yield
            all_ids.append(_id)
    assert all_ids == reference_ids                    # 结果必须和 tiktoken 一模一样

    assert tokenizer.decode(all_ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
def test_encode_iterable_memory_usage():
    """内存限制测试: encode_iterable 必须在 1MB 额外内存内完成!
    这个测试用 @memory_limit(1e6) 装饰 _encode_iterable, 如果内存超限就会崩溃.
    只在 Linux 上运行, 因为 Windows/Mac 对 rlimit 支持有限.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", encoding="utf-8") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):    # _encode_iterable 被 memory_limit 装饰
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
@pytest.mark.xfail(reason="Tokenizer.encode is expected to take more memory than allotted (1MB).")
def test_encode_memory_usage():
    """encode() 的内存测试 — 这个测试 *注定失败* (xfail)!
    encode() 会把所有 token ID 攒在一个 list 里返回, 对于 5MB 的文件必然超过 1MB 内存.
    这就是为什么我们需要 encode_iterable — 流式处理不占内存.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", encoding="utf-8") as f:
        contents = f.read()                            # 5MB 全加载到内存
        _ = _encode(tokenizer, contents)               # 再编码成 ID 列表 → 注定内存超限


# =============================================================================
# 8. 辅助包装函数 — 配合 @memory_limit 装饰器使用
# =============================================================================
# 为什么需要这些包装函数?
# 因为 @memory_limit 需要作用在"独立的函数"上才能正确限制内存。
# 如果直接装饰 test 函数, pytest 框架本身的内存也会被限制, 导致奇怪的问题。


@memory_limit(int(1e6))                                # int(1e6) = 1,000,000 字节 = ~1MB
def _encode_iterable(tokenizer, iterable):
    """
    把 tokenizer.encode_iterable 包一层, 让 @memory_limit 能限制它的内存。
    函数体里用 yield from 直接把 generator 的 yield 代理出去。
    """
    yield from tokenizer.encode_iterable(iterable)     # 逐个 yield, 内存几乎为 0


@memory_limit(int(1e6))
def _encode(tokenizer, text):
    """
    把 tokenizer.encode 包一层, 让 @memory_limit 能限制它的内存。
    这个函数 *注定* 会因内存超限而崩溃 — 因为 encode 返回完整列表。
    """
    return tokenizer.encode(text)                      # 返回整个 ID 列表 → 内存爆炸!
