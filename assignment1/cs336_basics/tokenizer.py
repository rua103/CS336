import regex as re
import unicodedata
from collections import Counter
from typing import List


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def get_stats(ids,counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def merge_tuple(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return tuple(new_ids)

#the helpers of the safe visulization of the token on the terminal
def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode('utf-8',errors="replace")
    s = replace_control_characters(s)
    return s

#basic Tokenizer
class Tokenizer:
    """Basic class for Tokenizer"""
    def __init__(self):
        self.merges = {}
        self.pattern = "" #for the pat
        self.special_tokens = {} #str -> int,e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() #int -> bytes
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, text):
        raise NotImplementedError
    
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
class RegexTokenizer(Tokenizer):
    def __init__(self,pattern=None):
        super().__init__()
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        # 允许从外部赋值
        self.merges = {} # (int, int) -> int
        self.vocab = {} # int -> bytes
        self.bytes_to_ids = {}

    def _load_from_raw_data(self, vocab: dict[int, bytes], merges_list: list[tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = {}
        self.byte_to_id = {v[0]: k for k, v in vocab.items() if len(v) == 1}

        bytes_to_id = {v: k for k, v in vocab.items()}

        for i, (p0, p1) in enumerate(merges_list):
            if p0 in bytes_to_id and p1 in bytes_to_id:
                id0 = bytes_to_id[p0]
                id1 = bytes_to_id[p1]
                new_id = 256 + i
                self.merges[(id0, id1)] = new_id

    def train(self, text, vocab_size,verbose=False, special_tokens=None):        
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        chunks = [text]
        if special_tokens:
            special_patterns = "(" + "|".join(re.escape(k) for k in sorted(special_tokens, key=len, reverse=True)) + ")"
            chunks = [part for part in re.split(special_patterns, text) if part not in special_tokens]

        #first, split the text into the chunks
        word_counts = Counter()
        for chunk in chunks:
            for match in re.finditer(self.compiled_pattern, chunk):
                word_counts[tuple(match.group().encode("utf-8"))] += 1

        pair_counts = {}
        pair_to_words = {}
        for chunk_ids, count in word_counts.items():
            pairs = list(zip(chunk_ids, chunk_ids[1:]))
            for pair in pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + count
            for pair in set(pairs):
                pair_to_words.setdefault(pair, set()).add(chunk_ids)

        merges = {} #(int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            if not pair_counts:
                break
            pair = max(pair_counts, key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))
            pair_count = pair_counts[pair]
            idx = 256 + i

            for chunk_ids in list(pair_to_words.get(pair, ())):
                count = word_counts.pop(chunk_ids, 0)
                if count == 0:
                    continue

                old_pairs = list(zip(chunk_ids, chunk_ids[1:]))
                for old_pair in old_pairs:
                    new_count = pair_counts[old_pair] - count
                    if new_count > 0:
                        pair_counts[old_pair] = new_count
                    else:
                        del pair_counts[old_pair]
                for old_pair in set(old_pairs):
                    words = pair_to_words.get(old_pair)
                    if words is not None:
                        words.discard(chunk_ids)
                        if not words:
                            del pair_to_words[old_pair]

                merged_ids = merge_tuple(chunk_ids, pair, idx)
                word_counts[merged_ids] = word_counts.get(merged_ids, 0) + count

                new_pairs = list(zip(merged_ids, merged_ids[1:]))
                for new_pair in new_pairs:
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count
                for new_pair in set(new_pairs):
                    pair_to_words.setdefault(new_pair, set()).add(merged_ids)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}:{pair} -> {idx} ({vocab[idx]}) had {pair_count} occurrences")

            
        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k,v in special_tokens.items()}

    def decode(self, ids):
        #given ids(list pf integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text_bytes):
        ids = [self.byte_to_id[b] for b in text_bytes]
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p : self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):
        #specific for the normal text without the special tokens
        text_chunks_iter = re.finditer(self.compiled_pattern,text)

        ids = []
        for chunk_iter in text_chunks_iter:
            chunk = chunk_iter.group()
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids
    
    def encode(self, text, allowed_special="all"):
        """Sometime we need to consider the special token may be used to attack our
        system,so it's neccessary to consider the safe ot the input"""
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError("Invalid input allowed_special")
        if not special:
            return self.encode_ordinary(text)
        
        special_patterns = "(" + "|".join(re.escape(k) for k in sorted(special, key=len, reverse=True)) + ")"
        special_chunks = re.split(special_patterns, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    # ===================Memory-Efficient Streaming Method========================
    def encode_iterable(self,iterable,allowed_special="all"):
        """
        Streaming encoding for massive datasets or files that can't fit in memory.
        iterable: An interable object(e.g., a file handler) that yields chunks/lines of string.
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
        elif isinstance(allowed_special,set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if special:
            special_patterns = "(" + "|".join(re.escape(k) for k in sorted(special, key=len, reverse=True)) + ")"

        for text in iterable:
            if allowed_special == "none_raise":
                assert all(token not in text for token in self.special_tokens)

            if not special:
                for chunk_iter in re.finditer(self.compiled_pattern, text):
                    chunk = chunk_iter.group()
                    chunk_bytes = chunk.encode("utf-8")
                    chunk_ids = self._encode_chunk(chunk_bytes)
                    yield from chunk_ids

            else:
                special_chunks = re.split(special_patterns, text)
                for part in special_chunks:
                    if part in special:
                        yield special[part]
                    else:
                        for chunk_iter in re.finditer(self.compiled_pattern, part):
                            chunk = chunk_iter.group()
                            chunk_bytes = chunk.encode("utf-8")
                            chunk_ids = self._encode_chunk(chunk_bytes)
                            yield from chunk_ids      
