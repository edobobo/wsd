from typing import Dict, Any, Optional, List

from src.sense_inventories import SenseInventory
from src.utils.wsd import read_from_raganato, expand_raganato_path


class SenseVocabulary:
    @classmethod
    def from_sense_inventory(cls, sense_inventory: SenseInventory):
        senses = sense_inventory.get_all_senses()
        sense_idx_store = dict()
        for sense in senses:
            sense_idx_store[sense] = len(sense_idx_store)
        return cls(sense_idx_store)

    @classmethod
    def from_datasets(cls, raganato_paths: List[str]):
        senses = []
        for raganato_path in raganato_paths:
            for _, _, wsd_sentence in read_from_raganato(*expand_raganato_path(raganato_path)):
                for wsd_instance in wsd_sentence:
                    senses += wsd_instance.labels
        senses = list(set(senses))
        sense_idx_store = dict()
        for sense in senses:
            sense_idx_store[sense] = len(sense_idx_store)
        return cls(sense_idx_store)

    def __init__(self, labels_idx: Dict[Any, int]):
        self.labels2idx = labels_idx
        self.idx2labels = {v: k for k, v in self.labels2idx.items()}

    def get_label(self, index: int) -> Optional[Any]:
        return self.idx2labels.get(index, None)

    def get_index(self, label: Any) -> Optional[int]:
        return self.labels2idx.get(label, None)

    def __len__(self):
        return len(self.labels2idx)
