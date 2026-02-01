from collections import ChainMap
import string
from typing import Callable, Dict, List, Set

import pandas as pd


def tokenize(text: str) -> List[str]:
    return [w.lower().strip(string.punctuation) for w in text.split() if w.strip(string.punctuation)]

class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        # TODO: implement this! Expected # of lines: <5
        words = tokenize(text)
        feats: Dict[str, float] = {}
        for w in words:
            if w and w not in self.STOP_WORDS:
                feats[w] = 1.0
        return self.prefix_with_name(feats)


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        n = len(tokenize(text))
        if n < 10:
            ret = {"short": 1.0}
        else:
            ret = {"long": 3.0}
        return self.prefix_with_name(ret)
    
class NegationHandling(FeatureMap):
    name = "neg"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """feature that counts negation words"""
        negators = {"not", "no", "never"}
        tokens = [t.rstrip(".,!?;:\"')(").lower() for t in text.split()]
        ret: Dict[str, float] = {}
        neg_count = 0.0
        scope_left = 0
        SCOPE_K = 3
        for t in tokens:
            if (t in negators) or t.endswith("n't"):
                neg_count += 1.0
                scope_left = SCOPE_K
                continue
            if scope_left > 0 and t and t not in BagOfWords.STOP_WORDS:
                # mark negated token (so it behaves like a separate bow feature)
                ret[f"NEG_{t}"] = ret.get(f"NEG_{t}", 0.0) + 1.0
                scope_left -= 1
        # binary presence of any negation and count
        ret["negation_count"] = float(neg_count)
        if neg_count > 0:
            ret["has_negation"] = 1.0
        return self.prefix_with_name(ret)
    

class PunctuationCount(FeatureMap):
    name = "punct"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        ret = {
            "exclaim": float(text.count("!")),
            "question": float(text.count("?")),
            "has_ellipsis": 1.0 if "..." in text else 0.0,
        }
        ret = {k: v for k, v in ret.items() if v != 0.0}
        return self.prefix_with_name(ret)
    

class SentimentLexicon(FeatureMap):
    name = "lex"
    POS = {"great","excellent","amazing","wonderful","best","love","loved","enjoyed","perfect","fun"}
    NEG = {"bad","terrible","awful","boring","worst","hate","hated","poor","dull","waste"}

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = tokenize(text)
        pos = sum(1 for w in words if w in self.POS)
        neg = sum(1 for w in words if w in self.NEG)
        ret = {
            "pos_count": float(pos),
            "neg_count": float(neg),
            "pos_minus_neg": float(pos - neg),
        }
        # keep sparse-ish:
        ret = {k: v for k, v in ret.items() if v != 0.0}
        return self.prefix_with_name(ret)
    

class Bigrams(FeatureMap):
    name = "bi"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = [w for w in tokenize(text) if w and w not in BagOfWords.STOP_WORDS]
        ret: Dict[str, float] = {}
        for i in range(len(words) - 1):
            bg = f"{words[i]}_{words[i+1]}"
            ret[bg] = 1.0  # binary bigram presence
        return self.prefix_with_name(ret)


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, NegationHandling, PunctuationCount, SentimentLexicon, Bigrams]}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        out: Dict[str, float] = {}
        for fn in featurize_fns:
            d = fn(text)
            for k, v in d.items():
                out[k] = out.get(k, 0.0) + v
        return out

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I don't love this movie!!!"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len", "neg", "punct"})
    print(featurize(text))
