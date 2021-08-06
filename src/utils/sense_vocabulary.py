from typing import Dict, Any, Optional, List

from tqdm import tqdm

from src.disambiguation_corpora import DisambiguationCorpus
from src.sense_inventories import SenseInventory
from src.utils.wsd import read_from_raganato, expand_raganato_path


class SenseVocabulary:
    @classmethod
    def from_sense_inventory(cls, sense_inventory: SenseInventory):
        return cls.from_sense_inventory_and_disambiguation_corpora(sense_inventory=sense_inventory, disambiguation_corpora=None)

    @classmethod
    def from_datasets(cls, disambiguation_corpora: List[DisambiguationCorpus]):
        return cls.from_sense_inventory_and_disambiguation_corpora(sense_inventory=None, disambiguation_corpora=disambiguation_corpora)

    @classmethod
    def from_sense_inventory_and_disambiguation_corpora(cls, sense_inventory: Optional[SenseInventory], disambiguation_corpora: Optional[List[DisambiguationCorpus]]):
        sense_idx_store = dict()
        # read from inventory
        if sense_inventory is not None:
            senses = sense_inventory.get_all_senses()
            for sense in tqdm(senses, desc='Sense vocabulary construction: reading senses from inventory'):
                assert sense not in sense_idx_store
                sense_idx_store[sense] = len(sense_idx_store)
        # read from datasets
        if disambiguation_corpora is not None:
            for disambiguation_corpus in tqdm(disambiguation_corpora, desc='Sense vocabulary construction: reading senses from disambiguation corpora'):
                for sentence in disambiguation_corpus:
                    for instance in sentence.instances:
                        if instance.labels is not None:
                            for sense in instance.labels:
                                if sense not in sense_idx_store:
                                    sense_idx_store[sense] = len(sense_idx_store)
        # add idk
        sense_idx_store["[$idk$]"] = len(sense_idx_store)
        # return
        return cls(sense_idx_store)

    @classmethod
    def load(cls, fp: str):
        sense_idx_store = dict()
        with open(fp) as f:
            for line in f:
                k, v = line.strip().split('\t')
                sense_idx_store[k] = int(v)
        return cls(sense_idx_store)

    def __init__(self, labels_idx: Dict[Any, int]):
        self.labels2idx = labels_idx
        self.idx2labels = {v: k for k, v in self.labels2idx.items()}

    def get_label(self, index: int) -> Optional[Any]:
        return self.idx2labels.get(index, "[$idk$]")

    def get_index(self, label: Any) -> Optional[int]:
        return self.labels2idx.get(label, None)

    def save(self, fp: str):
        with open(fp, 'w') as f:
            for k, v in self.labels2idx.items():
                f.write(f'{k}\t{v}\n')

    def __len__(self):
        return len(self.labels2idx)
