import tempfile
from logging import getLogger
from typing import Dict

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl

from src.disambiguation_corpora import WordNetCorpus
from src.predictors.base_predictor import BasePredictor
from src.utils.wsd import framework_evaluate, expand_raganato_path

logger = getLogger(__name__)


class PredictorsRaganatoEvaluateCallback(pl.Callback):
    def __init__(
        self,
        raganato_path: str,
        wsd_framework_dir: str,
        predictors: Dict[str, DictConfig],
    ):
        self.raganato_path = raganato_path
        self.wsd_framework_dir = wsd_framework_dir
        self.predictors: Dict[str, BasePredictor] = {k: hydra.utils.instantiate(v) for k, v in predictors.items()}

    def on_validation_epoch_start(self, trainer, pl_module):

        logger.info("PredictorsRaganatoEvaluateCallback started")

        for predictor_name, predictor in self.predictors.items():

            logger.info(f"Doing {predictor_name}")

            predictor.load_module_and_tokenizer(pl_module, None)

            with tempfile.TemporaryDirectory() as tmp_dir:
                predictions_file = f"{tmp_dir}/predictions.key.txt"

                predictor.predict_on_file(
                    WordNetCorpus(self.raganato_path, materialize=False, cached=False), predictions_file
                )

                _, _, f1 = framework_evaluate(
                    self.wsd_framework_dir,
                    gold_file_path=expand_raganato_path(self.raganato_path)[1],
                    pred_file_path=predictions_file,
                )

            pl_module.log(f"{predictor_name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
            logger.info(f"{predictor_name}: {f1} f1")
