from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, List, Tuple, Callable

import torch
from transformers import AutoModel


class TextEncoder(ABC, torch.nn.Module):
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        instances_offsets: List[List[Tuple[int, int]]],
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError


class ClassificationOutput(NamedTuple):
    output_logits: torch.Tensor
    output_probs: torch.Tensor
    output_predictions: torch.Tensor
    loss: Optional[torch.Tensor]


class ClassificationHead(ABC, torch.nn.Module):
    @abstractmethod
    def forward(
        self, encoded_instances: torch.Tensor, labels: Optional[torch.Tensor], **kwargs
    ) -> ClassificationOutput:
        raise NotImplementedError


class Disambiguator(torch.nn.Module):
    def __init__(self, encoder: TextEncoder, classification_head: ClassificationHead):
        super().__init__()
        self.encoder = encoder
        self.classification_head = classification_head

    def forward(
        self,
        input_ids: str,
        attention_mask: str,
        instances_offsets: List[List[Tuple[int, int]]],
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ClassificationOutput:
        encoded_sequence = self.encoder(input_ids, attention_mask, instances_offsets, **kwargs)
        return self.classification_head(encoded_sequence, labels, **kwargs)


class TransformerEncoder(TextEncoder):
    def __init__(
        self,
        transformer_model: str,
        fine_tune: bool,
        bpes_merging_strategy: Callable[[List[torch.Tensor]], torch.Tensor],
    ):
        super().__init__()
        self._encoder = AutoModel.from_pretrained(transformer_model)

        if not fine_tune:
            for param in self._encoder.parameters():
                param.requires_grad = False

        self.bpes_merging_strategy = bpes_merging_strategy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        instances_offsets: List[List[Tuple[int, int]]],
        **kwargs
    ) -> torch.Tensor:
        encoded_bpes = self._encoder(input_ids, attention_mask)[0]
        max_instances = max(map(len, instances_offsets))
        encoded_instances = torch.zeros(
            (input_ids.shape[0], max_instances, encoded_bpes.shape[-1]),
            dtype=encoded_bpes.dtype,
            device=encoded_bpes.device,
        )
        for i, sent_inst_offsets in enumerate(instances_offsets):
            encoded_instances[i, : len(sent_inst_offsets)] = torch.stack(
                [self.bpes_merging_strategy(encoded_bpes[i, sj:ej]) for sj, ej in sent_inst_offsets]
            )

        return encoded_instances

    def hidden_size(self) -> int:
        return self._encoder.config.hidden_size


class LinearClassificationHead(ClassificationHead):
    def __init__(self, hidden_size: int, output_vocab_size: int, hidden_compression: Optional[int] = None):
        super().__init__()
        if hidden_compression is None:
            self.classifier = torch.nn.Linear(hidden_size, output_vocab_size)
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_compression),
                torch.nn.Linear(hidden_compression, output_vocab_size),
            )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self, encoded_instances: torch.Tensor, labels: Optional[torch.Tensor], **kwargs
    ) -> ClassificationOutput:

        forward_out = self.classifier(encoded_instances)
        batch_size, _, output_size = forward_out.shape

        return ClassificationOutput(
            output_logits=forward_out,
            output_probs=torch.softmax(forward_out, dim=-1),
            output_predictions=torch.argmax(forward_out, dim=-1),
            loss=self.criterion(forward_out.view(-1, output_size), labels.view(-1)) if labels is not None else None,
        )


class BPESMergingStrategy(ABC):
    @abstractmethod
    def __call__(self, bpes_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AverageBPES(BPESMergingStrategy):
    def __call__(self, bpes_vectors: torch.Tensor) -> torch.Tensor:
        return torch.mean(bpes_vectors, dim=0)
