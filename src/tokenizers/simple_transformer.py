from typing import List, NamedTuple, Tuple, Optional

import torch
from transformers import AutoTokenizer


class TokenizationOutput(NamedTuple):
    input_ids: torch.Tensor
    tokens_offsets: List[Tuple[int, int]]


class SimpleTransformerTokenizer:
    def __init__(self, transformer_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model, use_fast=True)

    def tokenize(self, tokens: List[str]) -> Optional[TokenizationOutput]:
        tok_encoding = self.tokenizer.encode_plus(tokens, return_tensors="pt", is_split_into_words=True)
        try:
            tokenization_output = TokenizationOutput(
                input_ids=tok_encoding.input_ids.squeeze(0),
                tokens_offsets=[tuple(tok_encoding.word_to_tokens(wi)) for wi in range(len(tokens))],
            )
        except TypeError:
            print(f"Tokenization failed for tokens: {' | '.join(tokens)}")
            return None

        return tokenization_output

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def max_length(self) -> int:
        return self.tokenizer.model_max_length


if __name__ == "__main__":
    simple_transformer_tokenizer = SimpleTransformerTokenizer("bert-base-cased")
    simple_transformer_tokenizer.tokenize("Hi I am Mario".split(" "))
