import itertools
from typing import List, Union

import torch


class Tokenizer:
    def __init__(self, vocab: List[str]):
        self.idx2word = vocab
        self.word2idx = {word: i for i, word in enumerate(vocab)}

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]]):
        if type(token_ids) == list:
            return [self.idx2word[id_] for id_ in token_ids]
        elif type(token_ids) == int:
            return self.idx2word[token_ids]
        else:
            raise TypeError(f'Type of ids should be either list or int but is {type(token_ids)}')

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        if type(tokens) == list:
            return [self.word2idx.get(token, 1) for token in tokens]
        elif type(tokens) == str:
            return self.word2idx.get(tokens, 1)
        else:
            raise TypeError(f'Type of ids should be either list or str but is {type(tokens)}')

    def convert_tokens_to_string(self, tokens: List[str]):
        return ' '.join(tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False):
        if skip_special_tokens:
            token_ids = [id_ for id_ in token_ids if id_ >= 4]
        token_ids = list(itertools.takewhile(lambda id_: id_ != 0, token_ids))  # getting rid of pad tokens
        tokens = self.convert_ids_to_tokens(token_ids)
        return self.convert_tokens_to_string(tokens)

    def get_vocab(self):
        return self.word2idx

    def pad_single(self, token_ids: List[int], max_len: int):
        n_pad = max_len - len(token_ids)
        return token_ids + [0 for _ in range(n_pad)]

    def pad(self, token_ids: List[List[int]]):
        max_len = max(len(ids) for ids in token_ids)
        return [self.pad_single(ids, max_len) for ids in token_ids]

    def tokenize(self, text: str, padding=None):
        """Tokenizes a piece of text. Assumes that dots and commas etc. are taken care of before."""
        tokens = text.lower().split()
        token_ids = self.convert_tokens_to_ids(tokens)
        if padding and padding > 0:
            n_pad = padding - len(token_ids)
            token_ids = token_ids + [0 for _ in range(n_pad)]
        return token_ids


def to_device(*args, device=None):
    device = device or torch.device('cuda')
    return [arg.to(device) for arg in args]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts=None):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        text = self.texts[i]
        return self.tokenizer.tokenizer(text)

    def __len__(self, i):
        return len(self.texts)

    def get_tensors(self, texts: Union[str, List[str]], target_styles: Union[int, List[int]], device=None):
        if type(texts) == str:
            texts = [texts]
            target_styles = [target_styles]
        else:
            assert type(target_styles) == list
        assert len(texts) == len(
            target_styles), f'length of texts ({len(texts)}) must be equal to length of target_styles ({len(target_styles)})'
        token_ids = [self.tokenizer.tokenize(text) for text in texts]
        token_ids = self.tokenizer.pad(token_ids)
        token_ids = torch.tensor(token_ids)
        x_mask = (token_ids != 0).long()
        x_len = [len(ts) for ts in token_ids]
        y_neg = torch.tensor(target_styles).reshape(-1, 1)
        y_mask = (y_neg == 0).long()
        y_len = [1 for _ in range(len(y_neg))]
        token_ids, x_mask, y_neg, y_mask = to_device(token_ids, x_mask, y_neg, y_mask)
        return token_ids, x_mask, x_len, y_neg, y_mask, y_len


def translate(model, tokenizer, texts: Union[str, List[str]], target_styles: Union[int, List[int]], beam_size: int = 5):
    dataset = MyDataset(tokenizer)
    tensors = dataset.get_tensors(texts, target_styles)
    hs = model.translate(*tensors, beam_size=beam_size, max_len=300, poly_norm_m=0)
    return list(map(lambda h: tokenizer.decode(h), hs))
