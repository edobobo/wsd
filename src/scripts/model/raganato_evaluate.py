import hydra.utils
import omegaconf
import torch

from src.predictors.base_predictor import BasePredictor
from src.utils.hydra import fix
from src.utils.wsd import framework_evaluate


def load_module(module_class_full_name: str):
    parts = module_class_full_name.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


@hydra.main(config_path="../../../conf", config_name="root_test")
def main(conf: omegaconf.DictConfig) -> None:

    fix(conf)

    module_class = load_module(conf.test.module_class)

    module = module_class.load_from_checkpoint(conf.test.module_ckpt)
    module.to(torch.device(conf.test.device if conf.test.device != -1 else "cpu"))
    module.freeze()

    predictor: BasePredictor = hydra.utils.instantiate(conf.test.predictor)
    predictor.load_module_and_tokenizer(module, tokenizer=None)

    disambiguation_corpus = hydra.utils.instantiate(conf.test.disambiguation_corpus)

    prediction_keys_path = "predictions.gold.key.txt"

    predictor.predict_on_file(disambiguation_corpus, prediction_keys_path)

    p, r, f1 = framework_evaluate(conf.test.framework_dir, conf.test.gold_keys_path, prediction_keys_path)

    print(f"P: {p} | R: {r} | F1: {f1}")


if __name__ == "__main__":
    main()
