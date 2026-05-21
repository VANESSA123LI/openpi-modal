# ruff: noqa
# ---------------------------------------------------------------------------
# Appended to openpi/src/openpi/training/config.py at image-build time.
# It registers LoRA fine-tuning configs for pi0.5 (upstream ships LoRA only
# for pi0, not pi0.5 -- see github.com/Physical-Intelligence/openpi issues
# #672 and #842). We mirror `pi0_libero_low_mem_finetune` but with pi05=True.
#
# Everything below runs in the config.py module namespace, so TrainConfig,
# pi0_config, LeRobotLiberoDataConfig, DataConfig, weight_loaders, _optimizer,
# _CONFIGS and _CONFIGS_DICT are all already defined above.
# ---------------------------------------------------------------------------

# A pi0.5 model in LoRA mode: base weights frozen, low-rank adapters trained.
_PI05_LORA_MODEL = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)

# 1) Smoke-test config on the public LIBERO dataset. Use this to validate the
#    whole Modal train -> checkpoint -> serve loop before your own data arrives.
_pi05_libero_lora = TrainConfig(
    name="pi05_libero_lora",
    model=_PI05_LORA_MODEL,
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    # LoRA fits a much smaller footprint; keep the batch modest for a 24-48GB GPU.
    batch_size=16,
    num_train_steps=30_000,
    freeze_filter=_PI05_LORA_MODEL.get_freeze_filter(),
    # EMA is disabled for LoRA (only adapters train).
    ema_decay=None,
)

_NEW_CONFIGS = [_pi05_libero_lora]

# ---------------------------------------------------------------------------
# TEMPLATE for your own teleoperation data (fill in when the dataset exists).
# A custom LeRobot dataset needs a data config that maps your robot's
# camera/state/action keys into the model's expected layout. Uncomment and
# adapt once you know your dataset's repo_id and feature schema. Until then we
# leave it out so the module imports cleanly.
#
# _pi05_custom_lora = TrainConfig(
#     name="pi05_custom_lora",
#     model=_PI05_LORA_MODEL,
#     data=LeRobotDataConfig(            # <- pick/define the right factory
#         repo_id="<your-hf-username>/<your-dataset>",
#         base_config=DataConfig(prompt_from_task=True),
#     ),
#     weight_loader=weight_loaders.CheckpointWeightLoader(
#         "gs://openpi-assets/checkpoints/pi05_base/params"
#     ),
#     batch_size=16,
#     num_train_steps=20_000,
#     freeze_filter=_PI05_LORA_MODEL.get_freeze_filter(),
#     ema_decay=None,
# )
# _NEW_CONFIGS.append(_pi05_custom_lora)
# ---------------------------------------------------------------------------

for _c in _NEW_CONFIGS:
    if _c.name not in _CONFIGS_DICT:
        _CONFIGS.append(_c)
        _CONFIGS_DICT[_c.name] = _c
