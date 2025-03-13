from neural_lam.datastore import DATASTORES
from neural_lam.weather_dataset import WeatherDataModule

from diffusion_lam import utils
from diffusion_lam.trainer import Trainer


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

    __import__("pdb").set_trace()  # TODO delme


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml_as_attrdict(args.train_config)

    main(config)
