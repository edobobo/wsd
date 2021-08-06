import logging
import random
from typing import List, Optional, Iterable, Dict, Any

import torch

from src.disambiguation_corpora import DisambiguationCorpus
from src.sense_inventories import SenseInventory
from src.tokenizers.simple_transformer import SimpleTransformerTokenizer
from src.utils.base_dataset import BaseDataset, batchify
from src.utils.sense_vocabulary import SenseVocabulary


logger = logging.getLogger(__name__)


class SimpleTransformerDataset(BaseDataset):
    def __init__(
        self,
        disambiguation_corpus: DisambiguationCorpus,
        tokenizer: SimpleTransformerTokenizer,
        sense_vocabulary: SenseVocabulary,
        sense_inventory: SenseInventory,
        tokens_per_batch: int,
        max_batch_size: Optional[int],
        main_field: str,
        section_size: int,
        prebatch: bool,
        shuffle: bool,
        max_length: int,
    ):
        super().__init__(
            None,
            tokens_per_batch,
            max_batch_size,
            main_field,
            None,
            section_size,
            prebatch,
            shuffle,
            max_length,
        )
        self.disambiguation_corpus = disambiguation_corpus
        self.tokenizer = tokenizer
        self.sense_vocabulary = sense_vocabulary
        self.sense_inventory = sense_inventory
        self.__init_fields_batcher()

    def __init_fields_batcher(self) -> None:
        self.fields_batcher = {
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "instances_offsets": None,
            "tokens_offsets": None,
            "labels": lambda lst: batchify(lst, padding_value=-100),  # -100 == cross entropy ignore index,
            "comprehensive_labels": None,
            "sentence_id": None,
            "instance_ids": None,
            "possible_candidates": None,
        }

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:

        for disambiguation_sentence in self.disambiguation_corpus:

            tokenization_output = self.tokenizer.tokenize(tokens=[di.text for di in disambiguation_sentence.instances])

            if tokenization_output is None:
                continue

            input_ids, tokens_offsets = tokenization_output
            attention_mask = torch.ones_like(input_ids)

            labels, encoded_labels = [], []
            comprehensive_labels = []
            instances_offsets = []

            for i, disambiguation_instance in enumerate(disambiguation_sentence.instances):
                # the two different if branches are needed to handle the fact that, at prediction time, .labels will be empty
                if disambiguation_instance.labels is not None:
                    try:
                        _labels = [(l, self.sense_vocabulary.get_index(l)) for l in disambiguation_instance.labels]
                        l, el = random.choice([(l, el) for l, el in _labels if el is not None])
                        labels.append(l)
                        encoded_labels.append(el)
                        comprehensive_labels.append(disambiguation_instance.labels)
                    except IndexError:
                        logger.warning(f'Instance discarded as no label in {disambiguation_instance.labels} was found in sense vocabulary')
                        continue
                if disambiguation_instance.instance_id is not None:
                    instances_offsets.append(tokens_offsets[i])

            if len(instances_offsets) == 0:
                continue

            batch_elem = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "instances_offsets": instances_offsets,
                "tokens_offsets": tokens_offsets,
                "sentence_id": disambiguation_sentence.sentence_id,
                "instance_ids": [di.instance_id for di in disambiguation_sentence.instances],
                "possible_candidates": [
                    self.sense_inventory.get_possible_senses(di.lemma, di.pos) if di.instance_id is not None else None
                    for di in disambiguation_sentence.instances
                ]
            }

            if len(labels) > 0:
                batch_elem["labels"] = torch.tensor(
                    [self.sense_vocabulary.get_index(label) for label in labels], dtype=torch.long
                )
                batch_elem["comprehensive_labels"] = comprehensive_labels

            yield batch_elem
