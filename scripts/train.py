import os
from pprint import pprint

import matplotlib.pyplot as plt
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils.data import DataLoader

from dlam import utils
from dlam.trainer import Trainer
from dlam.vis import animate


def main(config):
    dataset = utils.get_component(config.dataset)
    dataloader = DataLoader(dataset, **config.dataloader.get("kwargs", {}))

    profiler_dir_path = "."
    profiler_filename = "profiler"
    trainer = Trainer(
        config.trainer.get("scheduler_config", {}),
        config.trainer.get("optimizer_config", {}),
        **config.trainer.get("kwargs", {}),
        profiler=SimpleProfiler(dirpath=profiler_dir_path, filename=profiler_filename),
        gpus=1
    )

    noise_model = utils.get_component(config.noise_model)
    model = utils.get_component(config.score_based_model, noise_model=noise_model)

    trainer.fit(model, dataloader)

    os.makedirs("results", exist_ok=True)
    utils.save(model.cpu(), "results/final_model.pkl")
    utils.get_component(config.evaluation, model=model, dataset=dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.train_config)

    main(config)
