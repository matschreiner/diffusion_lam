import numpy as np

from dlam.scheduler import GraphcastScheduler


def test_warmup_cosine_annealing_produces_expected_schedule(optimizer):
    min_factor = 0.01
    max_factor = 1
    warmup_steps = 10
    annealing_steps = 10
    initial_lr = optimizer.param_groups[0]["lr"]

    scheduler = GraphcastScheduler(
        optimizer,
        min_factor=min_factor,
        max_factor=max_factor,
        annealing_steps=annealing_steps,
        warmup_steps=warmup_steps,
    )

    lrs = []
    for _ in range(25):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    expected_warmup_lr = np.linspace(
        min_factor * initial_lr,
        max_factor * initial_lr,
        warmup_steps,
        endpoint=False,
    )
    warmup_lr = lrs[:warmup_steps]
    assert np.allclose(warmup_lr, expected_warmup_lr)

    annealing_lr = lrs[warmup_steps : warmup_steps + annealing_steps]

    # Formula for the cosine annealing
    expected_annealing_lr = min_factor * initial_lr + 0.5 * (
        max_factor * initial_lr - min_factor * initial_lr
    ) * (1 + np.cos(np.pi * np.arange(annealing_steps) / annealing_steps))
    assert np.allclose(annealing_lr, expected_annealing_lr)

    end_lr = np.array(lrs[warmup_steps + annealing_steps :])
    assert all(end_lr == min_factor * initial_lr)
