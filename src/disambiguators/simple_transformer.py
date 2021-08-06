from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, List, Tuple, Callable

import torch
from transformers import AutoModel, AutoConfig


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
        use_last_n_layers: int = 1,
        layer_merging_strategy: str = 'cat'
    ):
        super().__init__()

        # load encoder
        auto_config = AutoConfig.from_pretrained(transformer_model)
        auto_config.output_hidden_states = True
        self._encoder = AutoModel.from_pretrained(transformer_model, config=auto_config)

        if not fine_tune:
            for param in self._encoder.parameters():
                param.requires_grad = False

        self.bpes_merging_strategy = bpes_merging_strategy
        self.use_last_n_layers = use_last_n_layers
        if layer_merging_strategy not in ['cat', 'sum']:
            raise NotImplementedError(f'layer_merging_strategy {layer_merging_strategy} not implemented')
        self.layer_merging_strategy = layer_merging_strategy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        instances_offsets: List[List[Tuple[int, int]]],
        **kwargs
    ) -> torch.Tensor:

        # todo make it a param
        if self.layer_merging_strategy == 'cat':
            encoded_bpes = torch.cat(self._encoder(input_ids, attention_mask)[2][-self.use_last_n_layers :], dim=-1)
        elif self.layer_merging_strategy == 'sum':
            encoded_bpes = torch.stack(self._encoder(input_ids, attention_mask)[2][-self.use_last_n_layers :], dim=-1).sum(-1)
        else:
            raise NotImplementedError(f'layer_merging_strategy {self.layer_merging_strategy} not implemented')

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
        if self.layer_merging_strategy == 'cat':
            return self._encoder.config.hidden_size * self.use_last_n_layers
        elif self.layer_merging_strategy == 'sum':
            return self._encoder.config.hidden_size
        else:
            raise NotImplementedError(f'layer_merging_strategy {self.layer_merging_strategy} not implemented')


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


class BevilacquaClassificationHead(ClassificationHead):
    def __init__(self, hidden_size: int, output_vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, output_vocab_size, bias=False),
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self, encoded_instances: torch.Tensor, labels: Optional[torch.Tensor], **kwargs
    ) -> ClassificationOutput:

        flattened_encoded_instances = encoded_instances.view(-1, self.hidden_size)
        if flattened_encoded_instances.shape[0] > 1:
            flattened_encoded_instances = self.batch_norm(flattened_encoded_instances)

        flattened_forward_out = self.classifier(flattened_encoded_instances)
        loss = self.criterion(flattened_forward_out, labels.view(-1)) if labels is not None else None
        forward_out = flattened_forward_out.view(encoded_instances.shape[0], encoded_instances.shape[1], -1)

        return ClassificationOutput(
            output_logits=forward_out,
            output_probs=torch.softmax(forward_out, dim=-1),
            output_predictions=torch.argmax(forward_out, dim=-1),
            loss=loss
        )


class BPESMergingStrategy(ABC):
    @abstractmethod
    def __call__(self, bpes_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AverageBPES(BPESMergingStrategy):
    def __call__(self, bpes_vectors: torch.Tensor) -> torch.Tensor:
        return torch.mean(bpes_vectors, dim=0)
