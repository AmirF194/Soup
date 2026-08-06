"""`device_map="auto"` is wrong under any distributed launch, and it was everywhere.

Found on 8x H100 (benchmarks/gate-h100-validation.md, FINDING 4). The command
`soup train --gpus 8` prints for the user --

    accelerate launch --num_processes 8 soup train -c config.yaml

-- failed on every rank before training started:

    ValueError: You can't train a model that has been loaded with
    `device_map='auto'` in any distributed mode.

`device_map="auto"` shards one model across every visible GPU; under torchrun /
accelerate / deepspeed every rank would try to do that. Six trainers carried the
same unguarded line. So the multi-GPU path the tool advertises had never worked
on the SFT path, and a one-GPU dev box cannot see it.

These tests run everywhere: the distributed environment is just env vars.
"""

import pytest

from soup_cli.utils.gpu import resolve_device_map

TRAINERS = ["sft", "dpo", "grpo", "kto", "pretrain", "ppo"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Real torchrun vars leak in when the suite itself runs under a launcher."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


class TestSingleProcess:
    def test_cpu_stays_cpu(self):
        """'auto' on CPU produces meta tensors -- the reason the original line
        special-cased CPU at all. That behaviour must survive the fix."""
        assert resolve_device_map("cpu") == "cpu"

    def test_gpu_without_a_launcher_is_auto(self):
        assert resolve_device_map("cuda") == "auto"

    def test_world_size_one_is_auto_even_with_local_rank_set(self, monkeypatch):
        """`accelerate launch --num_processes 1` sets LOCAL_RANK=0 and
        WORLD_SIZE=1. That is a single process, and pinning it to {"" : 0} would
        needlessly disable auto-placement."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        assert resolve_device_map("cuda") == "auto"


class TestDistributed:
    @pytest.mark.parametrize("rank", [0, 1, 7])
    def test_each_rank_pins_its_own_device(self, monkeypatch, rank):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_device_map("cuda") == {"": rank}

    def test_cpu_wins_over_a_distributed_environment(self, monkeypatch):
        """--device cpu under a launcher must not be handed a CUDA index."""
        monkeypatch.setenv("LOCAL_RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_device_map("cpu") == "cpu"

    def test_never_returns_auto_when_distributed(self, monkeypatch):
        """The whole defect in one assertion: transformers raises on 'auto' in
        any distributed mode, so this is the value that must never come back."""
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert resolve_device_map("cuda") != "auto"


class TestMalformedEnvironment:
    @pytest.mark.parametrize("world", ["", "junk", "1.5"])
    def test_unparsable_world_size_falls_back_to_auto(self, monkeypatch, world):
        """A process with no usable distributed environment IS single-process."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", world)
        assert resolve_device_map("cuda") == "auto"

    def test_unparsable_local_rank_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "junk")
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_device_map("cuda") == "auto"


class TestNoTrainerStillHardcodesAuto:
    """Guards the COVERAGE, not the helper: a seventh trainer copying the old
    line back in would silently reopen a defect that is invisible on one GPU."""

    @pytest.mark.parametrize("name", TRAINERS)
    def test_trainer_uses_the_shared_helper(self, name):
        import importlib
        import inspect

        module = importlib.import_module("soup_cli.trainer.%s" % name)
        source = inspect.getsource(module)
        assert 'dev_map = "cpu" if self.device == "cpu" else "auto"' not in source
        assert "resolve_device_map" in source
