import torch


class NoiseScheduleBase(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.s = 1e-4

    def get_alpha(self, t):
        assert all(t >= 0) and all(t <= 1), "t must be in the range [0, 1]"
        alphas = self._get_alpha(t)
        return self.add_precision(alphas)

    def get_sigma(self, t):
        alpha = self.get_alpha(t)
        sigma = torch.sqrt(1 - alpha)
        return sigma

    def _get_alpha(self, t):
        raise NotImplementedError

    def add_precision(self, alpha):
        precision = 1 - 2 * self.s
        alpha = precision * alpha + self.s

        return alpha

    def forward(self, t):
        return self.get_alpha(t)


class PolynomialSchedule(NoiseScheduleBase):
    def __init__(self, power=2.0):
        super().__init__()
        self.power = power

    def _get_alpha(self, t):
        alpha = 1 - torch.pow(t, self.power)
        return alpha**2


if __name__ == "__main__":
    schedule1 = PolynomialSchedule(power=2)
    schedule2 = PolynomialSchedule(power=2.5)

    t = torch.linspace(0, 1, 100)

    alpha1 = schedule1.get_alpha(t)
    alpha2 = schedule2.get_alpha(t)

    import matplotlib.pyplot as plt

    plt.plot(t, alpha1)
    plt.plot(t, alpha2)
    plt.show()
