import logging
from collections import Counter
from functools import lru_cache
from numba import jit

logger = logging.getLogger(__name__)


class BPE:
    def __init__(self):
        self.merges = {}  # (int,int) -> int
        self.vocab = {}  # id -> bytes
        self._sorted_merges = None  # Cached sorted merges for encoding
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        logger.info(
            f"Starting BPE training with vocab_size={vocab_size}, text_length={len(text)}"
        )

        assert vocab_size > 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        logger.info(f"Text encoded to {len(ids)} bytes")

        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes

        for i in range(num_merges):
            # Use efficient Counter to find most frequent pair
            pair_counts = Counter(zip(ids, ids[1:]))

            if not pair_counts:
                logger.debug(f"No more pairs to merge after {i} merges")
                break  # No more pairs to merge

            # most_common(1) is more efficient than max() for finding top item
            most_common_pair, count = pair_counts.most_common(1)[0]
            idx = 256 + i

            # Perform the merge (JIT compiled for speed)
            ids = merge(ids, most_common_pair, idx)

            merges[most_common_pair] = idx
            vocab[idx] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: Pair: {most_common_pair} -> Id: {idx} ({vocab[idx]}) had {count} occurrences"
                )
            elif i % 1000 == 0:  # Log progress every 1000 merges
                logger.debug(f"Training progress: {i+1}/{num_merges} merges completed")

        self.merges = merges
        self.vocab = vocab
        # Cache sorted merges for efficient encoding (lowest merge index = highest priority)
        self._sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        logger.info(f"BPE training completed with {len(merges)} merges")

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
        # Cache sorted merges for efficient encoding (lowest merge index = highest priority)
        self._sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        logger.info(
            f"BPE model loaded with {len(merges)} merges and vocab_size={len(self.vocab)}"
        )

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def encode(self, text: str):
        logger.debug(f"Encoding text of length {len(text)}")
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        logger.debug(f"Text encoded to {len(ids)} initial bytes")

        merge_count = 0
        while len(ids) >= 2:
            # Get current pair statistics
            stats = get_stats(tuple(ids))

            # Find the highest priority mergeable pair
            mergeable_pair = None
            merge_idx = float('inf')

            for pair in stats:
                if pair in self.merges:
                    pair_idx = self.merges[pair]
                    if pair_idx < merge_idx:  # Lower index = higher priority
                        merge_idx = pair_idx
                        mergeable_pair = pair

            if mergeable_pair is None:
                break

            ids = merge(ids, mergeable_pair, self.merges[mergeable_pair])
            merge_count += 1

        logger.debug(f"Encoding completed: {merge_count} merges applied, final token count: {len(ids)}")
        return ids

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
