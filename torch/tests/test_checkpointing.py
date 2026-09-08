import copy
from pathlib import Path

import numpy as np
import torch

from llmpr_torch.checkpointing import load_checkpoint, save_checkpoint


def _objects(seed=3):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.2), torch.nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.97**step)
    sampler = torch.Generator().manual_seed(seed + 10)
    return model, optimizer, scheduler, sampler


def _step(model, optimizer, scheduler, sampler):
    x = torch.randn(4, 3)
    choice = torch.randint(2, (4,), generator=sampler)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), choice)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_resume_matches_uninterrupted_next_loss_and_parameters(tmp_path: Path):
    config = {"model": "tiny", "steps": 2}
    model, optimizer, scheduler, sampler = _objects()
    _step(model, optimizer, scheduler, sampler)
    save_checkpoint(tmp_path / "checkpoint.pt", model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=None, step=1, micro_step=1,
                    tokens_seen=8, config=config, generators={"sampler": sampler})
    expected_loss = _step(model, optimizer, scheduler, sampler)
    expected = copy.deepcopy(model.state_dict())

    resumed, resumed_optimizer, resumed_scheduler, resumed_sampler = _objects(seed=99)
    counters = load_checkpoint(
        tmp_path / "checkpoint.pt", model=resumed, optimizer=resumed_optimizer,
        scheduler=resumed_scheduler, scaler=None, expected_config=config,
        generators={"sampler": resumed_sampler})
    actual_loss = _step(resumed, resumed_optimizer, resumed_scheduler, resumed_sampler)
    assert counters == {"step": 1, "micro_step": 1, "tokens_seen": 8}
    assert actual_loss == expected_loss
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_resume_rejects_changed_config(tmp_path: Path):
    model, optimizer, scheduler, sampler = _objects()
    save_checkpoint(tmp_path / "checkpoint.pt", model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=None, step=0, micro_step=0,
                    tokens_seen=0, config={"a": 1}, generators={"sampler": sampler})
    with np.testing.assert_raises_regex(ValueError, "config"):
        load_checkpoint(tmp_path / "checkpoint.pt", model=model, optimizer=optimizer,
                        scheduler=scheduler, scaler=None, expected_config={"a": 2},
                        generators={"sampler": sampler})
