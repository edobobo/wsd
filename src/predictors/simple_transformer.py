from typing import Union, Optional

import hydra.utils
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from src.datasets.simple_transformer import SimpleTransformerDataset
from src.disambiguation_corpora import DisambiguationCorpus, WordNetCorpus
from src.pl_modules.simple_transformer import SimpleTransformerPLModule
from src.predictors.base_predictor import BasePredictor, PredictorOutput
from src.sense_inventories import WordNetSenseInventory, SenseInventory
from src.tokenizers.simple_transformer import SimpleTransformerTokenizer
from src.utils.sense_vocabulary import SenseVocabulary


class SimpleTransformerPredictor(BasePredictor):
    def __init__(
        self,
        module: Optional[Union[str, SimpleTransformerPLModule]],
        tokenizer: Optional[SimpleTransformerTokenizer],
        sense_inventory: SenseInventory,
        sense_vocabulary: Optional[SenseVocabulary],
        cat_senses: bool,
        tokens_per_batch: int,
        max_batch_size: int = 120,
        main_field: str = "input_ids",
        section_size: int = -1,
        prebatch: bool = True,
        shuffle: bool = False,
        device: int = 0,
        enable_autocast: bool = True,
    ):
        self.sense_inventory = sense_inventory
        self.sense_vocabulary = (
            sense_vocabulary
            if sense_vocabulary is not None
            else SenseVocabulary.from_sense_inventory(self.sense_inventory)
        )
        self.cat_senses = cat_senses
        self.tokens_per_batch = tokens_per_batch
        self.max_batch_size = max_batch_size
        self.main_field = main_field
        self.section_size = section_size
        self.prebatch = prebatch
        self.shuffle = shuffle
        self.device = torch.device(device if device != -1 else "cpu")
        self.enable_autocast = enable_autocast

        self.module = None
        self.tokenizer = None
        if module is not None:
            self.load_module_and_tokenizer(module, tokenizer)

    def load_module_and_tokenizer(self, module, tokenizer):
        self.module = self.load_module(module)
        self.tokenizer = self.load_tokenizer(tokenizer)

    def load_module(self, module: Union[str, SimpleTransformerPLModule]) -> SimpleTransformerPLModule:
        if type(module) == str:
            module = SimpleTransformerPLModule.load_from_checkpoint(module)
            module.to(self.device)
            module.freeze()

        return module

    def load_tokenizer(self, tokenizer: Optional[SimpleTransformerTokenizer]) -> SimpleTransformerTokenizer:
        if tokenizer is None:
            return hydra.utils.instantiate(self.module.hparams.tokenizer.simple_transformer_tokenizer)
        return tokenizer

    def build_dataset(self, disambiguation_corpus: DisambiguationCorpus) -> SimpleTransformerDataset:
        simple_transformer_dataset = SimpleTransformerDataset(
            disambiguation_corpus,
            self.tokenizer,
            self.sense_vocabulary,
            self.sense_inventory,
            self.tokens_per_batch,
            self.max_batch_size,
            self.main_field,
            self.section_size,
            self.prebatch,
            self.shuffle,
            self.tokenizer.max_length,
        )
        return simple_transformer_dataset

    def predict(self, disambiguation_corpus: DisambiguationCorpus) -> PredictorOutput:

        disambiguation_dataset = self.build_dataset(disambiguation_corpus)
        disambiguation_dataloader = DataLoader(disambiguation_dataset, batch_size=None, num_workers=0)

        prediction_keys = dict()

        with autocast(enabled=self.enable_autocast):
            with torch.no_grad():
                for batch in disambiguation_dataloader:
                    batch = self.module.transfer_batch_to_device(batch)
                    forward_out = self.module(**batch)
                    for sentence_instances, sentence_possible_candidates, sentence_probs in zip(
                        batch["instance_ids"], batch["possible_candidates"], forward_out["probabilities"]
                    ):

                        sentence_instances = [si for si in sentence_instances if si is not None]
                        sentence_possible_candidates = [spc for spc in sentence_possible_candidates if spc is not None]

                        assert len(sentence_instances) == len(sentence_possible_candidates)

                        for instance_id, sense_probabilities, instance_possible_candidates in zip(
                            sentence_instances, sentence_probs, sentence_possible_candidates
                        ):
                            if instance_id is None:
                                continue

                            if self.cat_senses:
                                predicted_sense = max(
                                    instance_possible_candidates,
                                    key=lambda ipc: sense_probabilities[self.sense_vocabulary.get_index(ipc)],
                                )
                            else:
                                predicted_index = torch.argmax(sense_probabilities).item()
                                predicted_sense = self.sense_vocabulary.get_label(predicted_index)

                            prediction_keys[instance_id] = predicted_sense

        return PredictorOutput(prediction_keys)
