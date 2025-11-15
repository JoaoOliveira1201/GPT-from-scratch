import logging
from collections import Counter
from functools import lru_cache
import regex as re
from numba import jit

logger = logging.getLogger(__name__)


class BPE:
    def __init__(self):
        self.gpt_split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.gpt_split_pattern = re.compile(self.gpt_split_pattern)
        self.merges = {}  # (int,int) -> int
        self.vocab = {}  # id -> bytes
        self._sorted_merges = None

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        logger.info(
            f"Starting BPE training with vocab_size={vocab_size}, text_length={len(text)}"
        )

        assert vocab_size > 256
        num_merges = vocab_size - 256
        chunks = self.gpt_split_pattern.findall(text)
        logger.info(f"Text split into {len(chunks)} chunks")
        
        chunk_ids_list = []
        for chunk in chunks:
            text_bytes = chunk.encode("utf-8")
            ids = list(text_bytes)
            chunk_ids_list.append(ids)
        
        total_bytes = sum(len(ids) for ids in chunk_ids_list)
        logger.info(f"Chunks encoded to {total_bytes} total bytes")

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            
            pair_counts = Counter()
            for ids in chunk_ids_list:
                if len(ids) >= 2:
                    pair_counts.update(zip(ids, ids[1:]))

            if not pair_counts:
                break

            most_common_pair, count = pair_counts.most_common(1)[0]
            idx = 256 + i

            chunk_ids_list = [merge(ids, most_common_pair, idx) for ids in chunk_ids_list]

            merges[most_common_pair] = idx
            vocab[idx] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: Pair: {most_common_pair} -> Id: {idx} ({vocab[idx]}) had {count} occurrences"
                )
            elif i % 1000 == 0:
                logger.debug(
                    f"Training progress: {i + 1}/{num_merges} merges completed"
                )

        self.merges = merges
        self.vocab = vocab
        self._sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        logger.info(f"BPE training completed with {len(self.merges)} merges")

    def save(self, model_file_name: str):
        model_file = model_file_name + ".model"
        logger.info(f"Saving BPE model to {model_file} with {len(self.merges)} merges")

        with open(model_file, "w") as f:
            f.write("Byte pair encoding - João\n")
            for merge_combo_1, merge_combo_2 in self.merges.keys():
                f.write(f"{merge_combo_1} {merge_combo_2}\n")

        logger.info(f"BPE model saved to {model_file}")

    def load(self, model_file: str):
        logger.info(f"Loading BPE model from {model_file}")

        assert model_file.endswith(".model")

        merges = {}
        idx = 256

        with open(model_file, "r", encoding="utf-8") as f:
            signature = f.readline().strip()
            assert signature == "Byte pair encoding - João"

            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.vocab = self._build_vocab()
        self._sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        logger.info(
            f"BPE model loaded with {len(merges)} merges and vocab_size={len(self.vocab)}"
        )

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def encode(self, text: str):
        logger.debug(f"Encoding text of length {len(text)}")
        chunks = self.gpt_split_pattern.findall(text)
        all_ids = []

        for chunk in chunks:
            text_bytes = chunk.encode("utf-8")
            ids = list(text_bytes)

            merge_count = 0
            while len(ids) >= 2:
                stats = get_stats(tuple(ids))

                mergeable_pair = None
                merge_idx = float("inf")

                for pair in stats:
                    if pair in self.merges:
                        pair_idx = self.merges[pair]
                        if pair_idx < merge_idx:
                            merge_idx = pair_idx
                            mergeable_pair = pair

                if mergeable_pair is None:
                    break

                ids = merge(ids, mergeable_pair, self.merges[mergeable_pair])
                merge_count += 1

            all_ids.extend(ids)

        logger.debug(
            f"Encoding completed: {len(chunks)} chunks processed, final token count: {len(all_ids)}"
        )
        return all_ids

    def decode(self, ids: list[int]):
        logger.debug(f"Decoding {len(ids)} tokens")
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        logger.debug(f"Decoding completed: {len(text)} characters")
        return text


@jit(nopython=True)
def merge(ids, pair, idx):
    newids = []
    i = 0
    n = len(ids)

    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


@lru_cache(maxsize=1024)
def get_stats(ids: tuple[int, ...]):
    """
    Given a tuple of integers, return a dictionary of counts of consecutive pairs
    """
    return Counter(zip(ids, ids[1:]))
