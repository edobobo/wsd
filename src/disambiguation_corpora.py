import logging
import re
import string

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, List, Iterator, Iterable, Tuple, Union
import collections
from itertools import chain

import numpy as np
import torch

from src.utils.wsd import read_from_raganato, expand_raganato_path, pos_map, WSDInstance


logger = logging.getLogger(__name__)


class DisambiguationInstance(NamedTuple):
    document_id: str
    sentence_id: str
    instance_id: Optional[str]
    text: str
    pos: str
    lemma: str
    labels: Optional[List[str]]


class DisambiguationSentence(NamedTuple):
    text: str
    instances: List[DisambiguationInstance]

    def document_id(self) -> str:
        return self.instances[0].document_id

    @property
    def sentence_id(self) -> str:
        return self.instances[0].sentence_id


class DisambiguationCorpus(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __iter__(self) -> Iterator[DisambiguationSentence]:
        raise NotImplementedError

    @abstractmethod
    def get_neighbours_sentences(
        self, document_id: str, sentence_id: str, prev_sent_num: int, next_sent_num: int
    ) -> Tuple[List[DisambiguationSentence], List[DisambiguationSentence]]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class WordNetCorpus(DisambiguationCorpus):
    def __init__(
        self,
        raganato_path: Union[str, List[str]],
        materialize: bool = False,
        cached: bool = False,
        shuffle: bool = False,
        is_doc_based: bool = True,
        is_train: bool = False,
    ):
        super().__init__()
        self.raganato_path = raganato_path
        self.cached = cached
        self.dataset_store = None
        self.is_doc_based = is_doc_based
        self.is_train = is_train

        if materialize or shuffle:
            self.materialize_dataset()
            if shuffle:
                np.random.shuffle(self.dataset_store)

        # indexing structures
        self.doc2sent_pos = None
        self.doc2sent_order = None
        self.sentences_index = None

    def read_dataset(self) -> Iterable[Tuple[str, str, List[WSDInstance]]]:
        raganato_path = self.raganato_path
        if type(raganato_path) == str:
            raganato_path = [raganato_path]
        return chain(*[read_from_raganato(*expand_raganato_path(rp)) for rp in raganato_path])

    def materialize_dataset(self) -> None:
        self.logger.info("Materializing raganato dataset")

        if self.cached:
            self.dataset_store = torch.load(f"{self.raganato_path}.pickle")
        else:
            self.dataset_store = list(self.read_dataset())

    def __iter__(self) -> Iterator[DisambiguationSentence]:

        raganato_iterator = self.dataset_store if self.dataset_store is not None else self.read_dataset()

        for document_id, sentence_id, wsd_sentence in raganato_iterator:

            disambiguation_instances = [
                DisambiguationInstance(
                    document_id,
                    sentence_id,
                    wi.instance_id
                    if not self.is_train or wi.instance_id is None or (wi.labels is not None and len(wi.labels) > 0)
                    else None,
                    wi.annotated_token.text,
                    pos_map.get(wi.annotated_token.pos, wi.annotated_token.pos),
                    wi.annotated_token.lemma,
                    wi.labels if wi.labels is None or len(wi.labels) > 0 else None,
                )
                for wi in wsd_sentence
            ]

            if re.fullmatch(rf"[{string.punctuation}]*", disambiguation_instances[-1].text) is None:
                disambiguation_instances.append(
                    DisambiguationInstance(
                        document_id,
                        sentence_id,
                        None,
                        ".",
                        pos_map.get("PUNCT", "PUNCT"),
                        ".",
                        None,
                    )
                )
                logger.debug(
                    f"Found sentence with missing trailing punctuation, adding it: "
                    f'{" ".join([di.text for di in disambiguation_instances])}'
                )

            disambiguation_sentence = DisambiguationSentence(
                " ".join(di.text for di in disambiguation_instances),
                disambiguation_instances,
            )

            yield disambiguation_sentence

    def _load_corpus_indexing_structures(self) -> None:
        self.logger.info("Initializing corpus indexing structures")

        if self.dataset_store is None:
            self.materialize_dataset()  # we need to keep references to the sentences

        self.doc2sent_pos = collections.defaultdict(dict)
        self.doc2sent_order = collections.defaultdict(list)
        self.sentences_index = dict()

        for disambiguation_sentence in iter(self):
            self.doc2sent_order[disambiguation_sentence.document_id].append(disambiguation_sentence.sentence_id)
            self.sentences_index[disambiguation_sentence.sentence_id] = disambiguation_sentence

        for document_id in self.doc2sent_order.keys():
            ordered_sentences = sorted(self.doc2sent_order[document_id])
            for sentence_index, sentence_id in enumerate(ordered_sentences):
                self.doc2sent_pos[document_id][sentence_id] = sentence_index
            self.doc2sent_order[document_id] = ordered_sentences

    def get_neighbours_sentences(
        self,
        document_id: str,
        sentence_id: str,
        prev_sent_num: int,
        next_sent_num: int,
    ) -> Tuple[List[DisambiguationSentence], List[DisambiguationSentence]]:

        if not self.is_doc_based:
            return [], []

        if self.doc2sent_order is None:
            self._load_corpus_indexing_structures()

        sentence_position_in_doc = self.doc2sent_pos[document_id][sentence_id]
        doc_sentences_id = self.doc2sent_order[document_id]

        prev_sentences_id = doc_sentences_id[
            max(sentence_position_in_doc - prev_sent_num, 0) : sentence_position_in_doc
        ]
        next_sentences_id = doc_sentences_id[
            sentence_position_in_doc + 1 : sentence_position_in_doc + 1 + next_sent_num
        ]

        prev_sentences = [self.sentences_index[sent_id] for sent_id in prev_sentences_id]
        next_sentences = [self.sentences_index[sent_id] for sent_id in next_sentences_id]

        return prev_sentences, next_sentences

    def __len__(self) -> int:
        if self.dataset_store is None:
            self.materialize_dataset()
        return len(self.dataset_store)


if __name__ == "__main__":

    wordnet_corpus = WordNetCorpus(
        "data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007",
        materialize=True,
        cached=False,
        shuffle=True,
        is_doc_based=True,
        is_train=True,
    )

    for wsd_sentence in wordnet_corpus:
        print(wsd_sentence)
