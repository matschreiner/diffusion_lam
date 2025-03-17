from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dlam.model import noise_schedule


def test_noise_schedule_base():
    ns = noise_schedule.NoiseScheduleBase()
    assert hasattr(ns, "get_alpha")
    assert hasattr(ns, "get_sigma")


def test_noise_schedule_raises():
    ns = noise_schedule.NoiseScheduleBase()
    t = torch.tensor([0.5])
    with pytest.raises(NotImplementedError):
        ns.get_alpha(t)

    with pytest.raises(NotImplementedError):
        ns.get_sigma(t)


def test_noise_schedule_base_throws_error_when_t_out_of_range():
    t_bad1 = torch.tensor([0, 1.1])
    t_bad2 = torch.tensor([-0.1, 1.0])
    ps = noise_schedule.NoiseScheduleBase()
    with pytest.raises(AssertionError):
        ps.get_alpha(t_bad1)
    with pytest.raises(AssertionError):
        ps.get_alpha(t_bad2)


def test_noise_correct_sigma():
    mock_get_alpha = MagicMock()
    mock_return = torch.tensor([0.2])
    mock_get_alpha.return_value = mock_return

    ns = noise_schedule.NoiseScheduleBase()
    ns.get_alpha = mock_get_alpha
    t = torch.tensor([0.5])

    expected_sigma = torch.sqrt(1 - mock_return)
    assert torch.allclose(ns.get_sigma(t), expected_sigma)


def test_polynomial_scheduler_instantiate():
    power = 2
    noise_schedule.PolynomialSchedule(power=power)


def test_polynomial_scheduler_can_get_alpha():
    power = 2
    ps = noise_schedule.PolynomialSchedule(power=power)
    t = torch.tensor(np.linspace(0, 1, 10))
    assert ps.get_alpha(t).shape == t.shape


def test_polynomial_scheduler_can_get_sigma():
    power = 2.56
    ps = noise_schedule.PolynomialSchedule(power=power)
    t = torch.tensor(np.linspace(0, 1, 10))
    alpha = ps.get_alpha(t)
    expected_sigma = torch.sqrt(1 - alpha)
    assert torch.allclose(ps.get_sigma(t), expected_sigma)


def test_alphas_not_zero():
    power = 2
    ps = noise_schedule.PolynomialSchedule(power=power)
    t = torch.tensor([0, 1])
    alpha = ps.get_alpha(t)
    assert alpha[0] > 0
    assert alpha[1] < 1
