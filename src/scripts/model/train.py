import omegaconf
import hydra

import pytorch_lightning as pl

from src.utils.hydra import fix


def train(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = hydra.utils.instantiate(conf.data.pl_data_module, conf)

    # main module declaration
    pl_module = hydra.utils.instantiate(conf.model.pl_module, conf)

    # callbacks declaration
    callbacks_store = []

    # lightning callbacks
    if conf.train.early_stopping_callback is not None:
        early_stopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback,
            filename="{epoch:02d}-{" + conf.train.callbacks_monitor + ":.2f}",
        )
        callbacks_store.append(model_checkpoint)

    # model callbacks
    if conf.callbacks.callbacks is not None:
        for callback in conf.callbacks.callbacks:
            callbacks_store.append(hydra.utils.instantiate(callback, _recursive_=False))

    # trainer
    trainer = hydra.utils.instantiate(
        conf.train.pl_trainer,
        callbacks=callbacks_store,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    fix(conf)
    train(conf)


if __name__ == "__main__":
    main()
