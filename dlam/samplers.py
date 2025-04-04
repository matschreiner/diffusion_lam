import matplotlib.pyplot as plt
import numpy as np
import torch


class SDESampler:
    def __init__(self, noise_model, noise_schedule):
        self.noise_model = noise_model
        self.noise_schedule = noise_schedule

    def sample(self, example, steps=100):
        with torch.no_grad():
            x = torch.randn_like(example)
            xt = [x]
            ts = torch.arange(steps - 1, 0, -1) / steps

            for t, tm1 in zip(ts[:-1], ts[1:]):
                t_diff = torch.full([len(x), 1], t)
                eps_hat = self.noise_model(x, t_diff)
                alphabar = self.noise_schedule.get_alpha(t)
                alpha = alphabar / self.noise_schedule.get_alpha(tm1)
                beta = 1 - alpha

                x = (
                    1
                    / (alpha**0.5)
                    * (x - (1 - alpha) / ((1 - alphabar) ** 0.5) * eps_hat)
                )
                if t > 1 / steps:
                    variance = beta
                    std = variance ** (0.5)
                    x += std * torch.randn_like(example)
                xt += [x]

            return x, xt
