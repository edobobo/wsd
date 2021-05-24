from typing import List, Tuple, Optional

import hydra

from src.disambiguators.simple_transformer import Disambiguator, ClassificationOutput

import pytorch_lightning as pl
import torch


class SimpleTransformerPLModule(pl.LightningModule):
    disambiguator: Disambiguator

    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.disambiguator: Disambiguator = self.build_disambiguator()
        self.accuracy = pl.metrics.Accuracy()

    def build_disambiguator(self):
        text_encoder = hydra.utils.instantiate(self.hparams.model.text_encoder)
        classification_head = hydra.utils.instantiate(
            self.hparams.model.classification_head,
            hidden_size=text_encoder.hidden_size(),
            output_vocab_size=self.hparams.data.output_vocab_size,
        )
        return Disambiguator(text_encoder, classification_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        instances_offsets: List[List[Tuple[int, int]]],
        labels: Optional[torch.Tensor],
        **kwargs,
    ) -> dict:
        classification_output: ClassificationOutput = self.disambiguator(
            input_ids, attention_mask, instances_offsets, labels
        )

        output_dict = {
            "logits": classification_output.output_logits,
            "probabilities": classification_output.output_probs,
            "predictions": classification_output.output_predictions,
        }

        if classification_output.loss is not None:
            output_dict["loss"] = classification_output.loss

        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output["loss"])
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**batch)
        self.log("val_loss", forward_output["loss"])

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.train.optim, _recursive_=False)(module=self)
