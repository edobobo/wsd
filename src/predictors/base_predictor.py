from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, Any, Optional


from src.disambiguation_corpora import DisambiguationCorpus


class PredictorOutput(NamedTuple):
    prediction_keys: Dict[str, str]
    kwargs: Optional[Dict[str, Any]] = None


class BasePredictor(ABC):
    @abstractmethod
    def load_module_and_tokenizer(self, module, tokenizer):
        """
        returns a tokenizer and a pl_module
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, disambiguation_corpus: DisambiguationCorpus) -> PredictorOutput:
        raise NotImplementedError

    def predict_on_file(self, disambiguation_corpus: DisambiguationCorpus, output_file_path: str) -> None:
        predictor_output = self.predict(disambiguation_corpus)
        with open(output_file_path, "w") as f:
            for instance_id, sense_key in predictor_output.prediction_keys.items():
                f.write(f"{instance_id} {sense_key}\n")
