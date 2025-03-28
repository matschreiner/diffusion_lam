from neural_lam.datastore import DATASTORES

from dlam import utils
from dlam.data_module import WeatherDataModule
from dlam.model.ddpm import DDPM
from dlam.model.score_model import NaiveModel
from dlam.trainer import Trainer


def main(config):
    trainer = Trainer(
        optimizer_config=config.training.optimizer,
        scheduler_config=config.training.scheduler,
    )

    datastore = DATASTORES[config.data.datastore.type](config.data.datastore.path)

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=config.data.ar_steps_train,
        ar_steps_eval=config.data.ar_steps_eval,
        standardize=config.data.standardize,
        num_past_forcing_steps=config.data.num_past_forcing_steps,
        num_future_forcing_steps=config.data.num_future_forcing_steps,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    #  data_module.setup()
    #  dataloader = data_module.train_dataloader()

    score_model = lambda x, _: x
    model = DDPM(score_model)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml_as_attrdict(args.train_config)

    main(config)
