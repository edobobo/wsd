from typing import Tuple

import hydra.utils
import omegaconf
import torch

from src.predictors.base_predictor import BasePredictor
from src.utils.commons import execute_bash_command
from src.utils.hydra import fix
from src.utils.wsd import xlwsd_framework_evaluate


def load_module(module_class_full_name: str):
    parts = module_class_full_name.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


@hydra.main(config_path="../../../conf/test")
def main(conf: omegaconf.DictConfig) -> None:

    fix(conf)

    module_class = load_module(conf.module_class)

    module = module_class.load_from_checkpoint(conf.module_ckpt)
    module.to(torch.device(conf.device if conf.device != -1 else "cpu"))
    module.freeze()

    predictor: BasePredictor = hydra.utils.instantiate(conf.predictor)
    predictor.load_module_and_tokenizer(module, tokenizer=None)

    disambiguation_corpus = hydra.utils.instantiate(conf.disambiguation_corpus)

    prediction_keys_path = "predictions.gold.key.txt"

    predictor.predict_on_file(disambiguation_corpus, prediction_keys_path)

    result = xlwsd_framework_evaluate(conf.framework_dir, conf.gold_keys_path, prediction_keys_path)
    print(f"XL-WSD Score: {result * 100:.2f}")


if __name__ == "__main__":
    main()
