from typing import Optional, Union, List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.sense_inventories import SenseInventory
from src.utils.sense_vocabulary import SenseVocabulary


class SimpleTransformerPLDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.train_dataset = None
        self.validation_dataset = None
        self.tokenizer = hydra.utils.instantiate(self.conf.tokenizer.simple_transformer_tokenizer)
        self.sense_inventory = hydra.utils.instantiate(self.conf.data.sense_inventory)
        self.sense_vocabulary = hydra.utils.instantiate(self.conf.data.sense_vocabulary)
        self.sense_vocabulary.save('sense_vocabulary.txt')
        self.conf.data.output_vocab_size = len(self.sense_vocabulary)

    def setup(self, stage: Optional[str] = None):

        if stage == "fit":

            self.train_dataset = hydra.utils.instantiate(
                self.conf.data.train_dataset,
                tokenizer=self.tokenizer,
                sense_inventory=self.sense_inventory,
                sense_vocabulary=self.sense_vocabulary,
                max_length=self.tokenizer.max_length,
            )

            self.validation_dataset = hydra.utils.instantiate(
                self.conf.data.validation_dataset,
                tokenizer=self.tokenizer,
                sense_inventory=self.sense_inventory,
                sense_vocabulary=self.sense_vocabulary,
                max_length=self.tokenizer.max_length,
            )

        else:
            raise NotImplementedError

    def get_dataloader(self, dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=None, num_workers=self.conf.data.num_workers)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.get_dataloader(self.validation_dataset)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError
