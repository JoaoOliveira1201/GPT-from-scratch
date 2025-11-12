import logging

logger = logging.getLogger(__name__)


class BPE:
    def __init__(self):
        self.merges = {}  # (int,int) -> int
        self.vocab = {}  # id -> bytes

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
            stats = get_stats(ids)
            most_common_pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = merge(ids, most_common_pair, idx)

            merges[most_common_pair] = idx
            vocab[idx] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: Pair: {most_common_pair} -> Id: {idx} ({vocab[idx]}) had {stats[most_common_pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab
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
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) > 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing can be merged
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids: list[int]):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


def merge(ids: list[int], pair: tuple[int, int], idx: int):
    newids = []
    i = 0

    while i < len(ids):
        is_a_match = ids[i] == pair[0] and ids[i + 1] == pair[1]
        at_the_end = i < len(ids) - 1
        if not at_the_end and is_a_match:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def get_stats(ids: list[int]):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    """
    counts = {}
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts
