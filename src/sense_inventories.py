from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional

from nltk.corpus import wordnet as wn
from tqdm import tqdm

from src.utils.collections import flatten
from src.utils.wsd import pos_map


class SenseInventory(ABC):
    @abstractmethod
    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_definition(self, sense: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_all_senses(self) -> List[str]:
        raise NotImplementedError


# WORDNET


@lru_cache(maxsize=None)
def gloss_from_sense_key(sense_key: str) -> str:
    return wn.lemma_from_key(sense_key).synset().definition()


class WordNetSenseInventory(SenseInventory):

    _shared_state = {}

    def __init__(self, wn_candidates_path: str):
        # borg pattern
        if wn_candidates_path not in self._shared_state:
            self.lemmapos2senses = dict()
            self._load_lemmapos2senses(wn_candidates_path)
            self._shared_state[wn_candidates_path] = self.__dict__
        else:
            self.__dict__ = self._shared_state[wn_candidates_path]

    def _load_lemmapos2senses(self, wn_candidates_path: str):
        with open(wn_candidates_path) as f:
            for line in f:
                lemma, pos, *senses = line.strip().split("\t")
                self.lemmapos2senses[(lemma, pos)] = senses

    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        return self.lemmapos2senses.get((lemma, pos), [])

    def get_definition(self, sense: str) -> str:
        return gloss_from_sense_key(sense)

    def get_all_senses(self) -> List[str]:
        return sorted(list(set(flatten(self.lemmapos2senses.values()))))


class BabelNetSenseInventory(SenseInventory):

    _shared_state = {}

    def __init__(self, inventory_path: str, definitions_path: Optional[str] = None):

        # borg pattern

        if (inventory_path, definitions_path) not in self._shared_state:
            self.lemmapos2synsets = dict()
            self._load_inventory(inventory_path)
            self.synset2definition = dict()

            if definitions_path is not None:
                self.synset2definition = dict()
                self._load_synset_definitions(definitions_path)

            self._shared_state[(inventory_path, definitions_path)] = self.__dict__
        else:
            self.__dict__ = self._shared_state[(inventory_path, definitions_path)]

    def _load_inventory(self, inventory_path: str) -> None:
        with open(inventory_path) as f:
            for line in tqdm(f, desc='Inventory: loading lemmapos -> synset mapping'):
                lemmapos, *synsets = line.strip().split("\t")
                lemma, pos = lemmapos.split("#")
                pos = pos_map[pos]
                self.lemmapos2synsets[(lemma, pos)] = synsets

    def _load_synset_definitions(self, definitions_path: str) -> None:
        with open(definitions_path) as f:
            for line in tqdm(f, desc='Inventory: loading definitions'):
                synset, definition = line.strip().split("\t")
                self.synset2definition[synset] = definition

    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        return self.lemmapos2synsets.get((lemma.lower().replace(" ", "_"), pos), [])

    def get_definition(self, sense: str) -> str:
        if self.synset2definition is not None:
            return self.synset2definition[sense]

    def get_all_senses(self) -> List[str]:
        return sorted(list(set(flatten(self.lemmapos2synsets.values()))))
