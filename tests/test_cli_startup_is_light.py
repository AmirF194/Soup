"""The light-CLI invariant: ``import soup_cli.cli`` must not load PyTorch.

This replaces the family of per-module ``test_no_top_level_torch`` AST guards for
the purpose of protecting CLI startup. Those guards prove a *syntactic* property
(this file has no top-level ``import torch``); the requirement is a *runtime* one
(torch is not in ``sys.modules``). An AST guard cannot see through a transitive
import, nor through a lazily-written factory that is *called* at module scope --
which is exactly how the v0.71.41 regression got in with every guard green.

One runtime assertion covers every module transitively and cannot drift.

Everything here runs in a FRESH subprocess: the pytest process has already
imported torch via other test modules, so an in-process assertion would be
vacuous (it would fail on a clean tree and pass on nothing).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Heavy deps that the documented "light core" (v0.71.0 deps-split) excludes.
# These live behind the [train] extra; the CLI must import without them.
HEAVY = ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes")

_PROBE = (
    "import sys\n"
    "import soup_cli.cli\n"
    "print(','.join(sorted(m for m in {heavy!r} if m in sys.modules)))\n"
)


def _loaded_heavy_deps() -> list[str]:
    """Import the CLI in a clean interpreter; return heavy deps it pulled in."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=HEAVY)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return [m for m in proc.stdout.strip().split(",") if m]


def test_cli_import_does_not_load_torch():
    """The headline invariant. A violation is a ~7x CLI startup regression."""
    leaked = _loaded_heavy_deps()
    assert not leaked, (
        f"`import soup_cli.cli` pulled in {leaked}. The light core must stay "
        "torch-free -- run scripts/find_import_leak.py to get the call site. "
        "A lazy import that is CALLED at module scope is not lazy."
    )


def test_probe_would_catch_a_regression():
    """Control: the probe must actually detect torch when it IS loaded.

    Without this, a probe that silently returned ``[]`` -- wrong module name,
    swallowed error, changed output format -- would make the test above pass
    forever while proving nothing.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch; import soup_cli.cli; "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip("torch not installed (light-core-only environment)")
    assert proc.stdout.strip() == "True", (proc.stdout, proc.stderr)


@pytest.mark.parametrize(
    "module",
    [
        # Modules CLAUDE.md documents as "NO top-level torch". Each is reachable
        # from a light command or is a pure planner/verdict half.
        "soup_cli.utils.reward_stress",
        "soup_cli.utils.reward_synth",
        "soup_cli.utils.ship_verdict",
        "soup_cli.utils.layer_stream",
    ],
)
def test_documented_light_module_stays_light(module: str):
    """Per-module runtime check for the modules that claim to be torch-free.

    Same reasoning as above: the claim is about ``sys.modules``, so the test has
    to be about ``sys.modules``.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, {module}; "
            "print(','.join(sorted(m for m in "
            f"{HEAVY!r} if m in sys.modules)))",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    leaked = [m for m in proc.stdout.strip().split(",") if m]
    assert not leaked, f"{module} pulled in {leaked} at import time"


def test_stress_sentinel_matches_the_controller():
    """``reward_stress`` copies the sentinel rather than importing it.

    The copy exists so the light CLI path does not pay for the controller's
    transformers import. Pin the two values equal, so the duplication cannot
    silently drift into two different sentinels.
    """
    from soup_cli.utils import reward_hack_control, reward_stress

    assert reward_stress.DEFAULT_SENTINEL == reward_hack_control._DEFAULT_SENTINEL
