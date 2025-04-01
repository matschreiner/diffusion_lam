import matplotlib.pyplot as plt
import pmodel
import pytorch_lightning as pl
import torch

import dlam
from dlam import utils

score_model = pmodel.ScoreModel()
ddpm = pmodel.DDPM(score_model)

dataset = dlam.toy_data.HalfmoonDataset(n_samples=100000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048)
pl.Trainer().fit(ddpm, dataloader)
