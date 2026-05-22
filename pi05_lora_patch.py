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

# ===========================================================================
# EgoDex (Apple) -> pi0.5 co-training config.
# Mirrors openpi's LeRobotLiberoDataConfig / LiberoInputs pattern. The dataset
# is built by egodex_modal.py with columns:
#   observation.images.image (egocentric video), observation.state (20),
#   action (20), and a per-frame task string (prompt_from_task=True).
# Single egocentric camera -> base_0_rgb; no wrist cams -> zeros + False mask.
#
# NOTE: written by mirroring openpi source, NOT yet run end-to-end. Validate
# the RepackTransform key strings against how openpi surfaces your LeRobot
# columns, and confirm action/state dims after a first training step.
# ===========================================================================
import numpy as _np  # noqa: E402


def _egodex_parse_image(image):
    image = _np.asarray(image)
    if _np.issubdtype(image.dtype, _np.floating):
        image = (255 * image).astype(_np.uint8)
    if image.shape[0] == 3:  # CHW -> HWC
        image = _np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class EgoDexInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _egodex_parse_image(data["observation/image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": _np.zeros_like(base),
                "right_wrist_0_rgb": _np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": _np.True_,
                "left_wrist_0_rgb": _np.False_,   # EgoDex has no wrist cameras
                "right_wrist_0_rgb": _np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class EgoDexOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": _np.asarray(data["actions"][:, :20])}  # 20-dim bimanual


@dataclasses.dataclass(frozen=True)
class EgoDexDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs, model_config) -> DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "observation/image": "observation.images.image",
                    "observation/state": "observation.state",
                    "actions": "action",
                })
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[EgoDexInputs(model_type=model_config.model_type)],
            outputs=[EgoDexOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


_pi05_egodex_lora = TrainConfig(
    name="pi05_egodex_lora",
    model=_PI05_LORA_MODEL,
    data=EgoDexDataConfig(
        repo_id="you/egodex_test",  # <- set to the LeRobot repo_id from egodex_modal.py
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=16,
    num_train_steps=20_000,
    freeze_filter=_PI05_LORA_MODEL.get_freeze_filter(),
    ema_decay=None,
)
_NEW_CONFIGS.append(_pi05_egodex_lora)

for _c in _NEW_CONFIGS:
    if _c.name not in _CONFIGS_DICT:
        _CONFIGS.append(_c)
        _CONFIGS_DICT[_c.name] = _c
