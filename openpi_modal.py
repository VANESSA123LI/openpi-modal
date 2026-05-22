"""Modal app for LoRA fine-tuning and serving pi0.5 (openpi).

Quickstart
----------
  modal secret create huggingface HF_TOKEN=hf_xxx     # once
  modal run   openpi_modal.py::download_base           # cache pi05_base weights
  modal run   openpi_modal.py::train --exp-name smoke  # LoRA fine-tune on LIBERO
  modal deploy openpi_modal.py                          # bring up the policy server

The `pi05_libero_lora` config is registered by pi05_lora_patch.py, which is
baked into the image. Swap in your own teleop dataset later by adding a config
in that patch file (a template is included there).
"""

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "openpi-pi05"
OPENPI_DIR = "/openpi"

# Default training run identity (override on the CLI for train; set as env for serve).
DEFAULT_CONFIG = "pi05_libero_lora"
DEFAULT_EXP = "smoke"

# --- Persistent storage ----------------------------------------------------
# cache_vol: base checkpoints + computed norm-stats assets (OPENPI_DATA_HOME)
# data_vol:  HuggingFace / LeRobot dataset downloads
# ckpt_vol:  training outputs (./checkpoints inside the repo)
cache_vol = modal.Volume.from_name("openpi-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("openpi-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)

VOLUMES = {
    "/cache": cache_vol,
    "/data": data_vol,
    f"{OPENPI_DIR}/checkpoints": ckpt_vol,
}

# HF token: required for gated/custom datasets, harmless for the public LIBERO set.
# Create with:  modal secret create huggingface HF_TOKEN=hf_xxx
SECRETS = [modal.Secret.from_name("huggingface")]

ENV = {
    "OPENPI_DATA_HOME": "/cache/openpi",   # base ckpts + norm assets land here
    "HF_HOME": "/data/hf",
    "HF_LEROBOT_HOME": "/data/lerobot",    # LeRobot dataset cache
    "WANDB_MODE": "disabled",              # flip to "online" + add a wandb secret to track
}

# --- Container image --------------------------------------------------------
# openpi targets Ubuntu 22.04 + uv. JAX CUDA wheels are self-contained, so the
# host GPU driver Modal provides is enough -- no separate CUDA toolkit needed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "build-essential", "clang", "wget")
    .pip_install("uv")
    .run_commands(
        "git lfs install",
        f"git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git {OPENPI_DIR}",
    )
    .workdir(OPENPI_DIR)
    # Build the uv-managed venv with all (CUDA) deps at image-build time.
    .run_commands(
        "GIT_LFS_SKIP_SMUDGE=1 uv sync",
        "GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .",
    )
    # Register the pi0.5 LoRA configs by appending our patch to openpi's config.py.
    .add_local_file("pi05_lora_patch.py", f"{OPENPI_DIR}/pi05_lora_patch.py", copy=True)
    .run_commands(
        f"cat {OPENPI_DIR}/pi05_lora_patch.py >> {OPENPI_DIR}/src/openpi/training/config.py",
        # Fail the build loudly if the configs didn't register.
        "uv run python -c \"import openpi.training.config as c; c.get_config('pi05_libero_lora'); c.get_config('pi05_egodex_lora'); print('lora configs OK')\"",
    )
    .env(ENV)
)

app = modal.App(APP_NAME, image=image)

HOURS = 60 * 60


def _run(cmd: str) -> None:
    """Run a shell command in the openpi repo, streaming output to Modal logs."""
    print(f"\n$ {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, cwd=OPENPI_DIR, check=True)


# --- Prefetch base weights --------------------------------------------------
@app.function(volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def download_base(model: str = "pi05_base"):
    """Download a pretrained checkpoint into the cache volume (one-time)."""
    _run(
        "uv run python -c "
        f"\"import openpi.shared.download as d; "
        f"p = d.maybe_download('gs://openpi-assets/checkpoints/{model}'); "
        f"print('cached at', p)\""
    )
    cache_vol.commit()


# --- Training ---------------------------------------------------------------
@app.function(
    gpu=os.environ.get("OPENPI_TRAIN_GPU", "L40S"),  # ~48GB, comfortable for pi0.5 LoRA
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * HOURS,
)
def train(
    exp_name: str = DEFAULT_EXP,
    config_name: str = DEFAULT_CONFIG,
    num_train_steps: int = 30_000,
    batch_size: int = 16,
    overwrite: bool = False,
    resume: bool = False,
):
    """Compute norm stats (cached) then LoRA fine-tune pi0.5."""
    stats_marker = Path(f"/cache/openpi/.normstats_done_{config_name}")
    if not stats_marker.exists():
        _run(f"uv run scripts/compute_norm_stats.py --config-name {config_name}")
        stats_marker.touch()
        cache_vol.commit()

    flag = "--overwrite" if overwrite else ("--resume" if resume else "")
    _run(
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "
        f"uv run scripts/train.py {config_name} "
        f"--exp-name={exp_name} "
        f"--num-train-steps={num_train_steps} "
        f"--batch-size={batch_size} {flag}".strip()
    )
    ckpt_vol.commit()
    print(f"\nDone. Checkpoints under /openpi/checkpoints/{config_name}/{exp_name}/")


def _latest_checkpoint(config_name: str, exp_name: str) -> str:
    base = Path(f"{OPENPI_DIR}/checkpoints/{config_name}/{exp_name}")
    steps = sorted(int(p.name) for p in base.iterdir() if p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No checkpoints found under {base}")
    return str(base / str(steps[-1]))


# --- Inference: persistent policy server (websocket on :8000) ---------------
@app.function(
    gpu=os.environ.get("OPENPI_SERVE_GPU", "L4"),  # inference needs >8GB; L4 (24GB) is cheap
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * HOURS,
    min_containers=1,
)
@modal.web_server(8000, startup_timeout=10 * 60)
def serve():
    """Serve the trained policy. Connect with openpi's websocket client.

    Override target with env vars OPENPI_SERVE_CONFIG / OPENPI_SERVE_EXP, or
    point straight at a dir with OPENPI_SERVE_DIR.
    """
    config_name = os.environ.get("OPENPI_SERVE_CONFIG", DEFAULT_CONFIG)
    exp_name = os.environ.get("OPENPI_SERVE_EXP", DEFAULT_EXP)
    ckpt_dir = os.environ.get("OPENPI_SERVE_DIR") or _latest_checkpoint(config_name, exp_name)
    print(f"Serving {config_name} from {ckpt_dir}", flush=True)
    subprocess.Popen(
        "uv run scripts/serve_policy.py policy:checkpoint "
        f"--policy.config={config_name} --policy.dir={ckpt_dir}",
        shell=True,
        cwd=OPENPI_DIR,
    )


# --- Local convenience entrypoint ------------------------------------------
@app.local_entrypoint()
def main(exp_name: str = DEFAULT_EXP):
    """`modal run openpi_modal.py` -> prefetch weights, then train."""
    download_base.remote()
    train.remote(exp_name=exp_name)
