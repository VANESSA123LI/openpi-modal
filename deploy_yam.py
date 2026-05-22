"""Run a pi0.5 policy on physical I2RT YAM arms.

Architecture
------------
    [YAM arms + cameras]  <--ZMQ/SDK-->  THIS SCRIPT  <--websocket-->  [pi0.5 policy server]
                                          (robot PC)                    (Modal serve / local GPU)

This script owns the real-time loop. Each control tick it pops one action from a
queue and commands the arms; when the queue drains it captures a fresh
observation, asks the policy server for a new action chunk, and refills.

Prereqs on the robot PC
-----------------------
    # openpi client (lightweight, no JAX needed)
    cd <openpi>/packages/openpi-client && pip install -e .
    # your YAM stack for hardware I/O (reads joints+cameras, commands arms)
    #   github.com/uynitsuj/robots_realtime  (or the I2RT YAM SDK)

Start the policy server first:
    # local (recommended for real control):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_custom_lora --policy.dir=<your_checkpoint>
    # or point --host at the Modal deployment URL (higher latency).

Run:
    python deploy_yam.py --host localhost --port 8000 \
        --prompt "pick up the red block and place it in the bowl"
"""

import argparse
import time
from collections import deque

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# ---------------------------------------------------------------------------
# HARDWARE ADAPTER -- fill these in with your YAM SDK / robots_realtime calls.
# These are the SAME reads/writes your teleoperation + data-collection path
# already uses, just driven by the policy instead of a GELLO leader arm.
# CRITICAL: the layout you build here MUST match how the model was trained --
# same camera ordering, same state vector composition, same action space and
# units. Mismatch = the policy silently produces garbage.
# ---------------------------------------------------------------------------


class YamInterface:
    def __init__(self):
        # TODO: connect to the YAM arms + cameras (open ZMQ sockets / SDK handles).
        raise NotImplementedError("Wire up your YAM connection here")

    def read_state(self) -> np.ndarray:
        """Return the proprioceptive state vector, UNNORMALIZED.

        For bimanual YAM this is typically both arms' joint positions plus
        gripper widths concatenated, e.g. [left_q(7), left_grip(1),
        right_q(7), right_grip(1)] -> shape (16,). Must match training order.
        """
        raise NotImplementedError

    def read_cameras(self) -> dict[str, np.ndarray]:
        """Return raw camera frames as HxWx3 uint8 arrays.

        Keys must match what the policy was trained with. A common pi0.5
        layout is one scene cam + per-wrist cams:
            {"base": ..., "left_wrist": ..., "right_wrist": ...}
        """
        raise NotImplementedError

    def command_action(self, action: np.ndarray) -> None:
        """Send ONE action to the arms (e.g. target joint positions + grippers).

        Same dimensionality/units as the training action space. If the policy
        outputs delta actions, integrate against the current state here.
        """
        raise NotImplementedError


def build_observation(yam: YamInterface, prompt: str) -> dict:
    """Pack YAM sensors into the dict the pi0.5 policy server expects.

    Keys here MUST line up with your training data config's image/state keys.
    The example below uses a 3-camera bimanual layout; adjust to yours.
    """
    cams = yam.read_cameras()

    def prep(img):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))

    return {
        "observation/image": prep(cams["base"]),
        "observation/left_wrist_image": prep(cams["left_wrist"]),
        "observation/right_wrist_image": prep(cams["right_wrist"]),
        "observation/state": yam.read_state(),
        "prompt": prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost", help="policy server host")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", required=True, help="task instruction")
    ap.add_argument("--control-hz", type=float, default=30.0,
                    help="rate at which actions are sent to the arms")
    ap.add_argument("--actions-per-chunk", type=int, default=10,
                    help="how many actions from each chunk to execute before "
                         "re-querying (<= model action_horizon)")
    args = ap.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Connected to policy server at {args.host}:{args.port}")

    yam = YamInterface()
    period = 1.0 / args.control_hz
    queue: deque[np.ndarray] = deque()

    try:
        while True:
            tick = time.time()

            if not queue:
                obs = build_observation(yam, args.prompt)
                # actions: (action_horizon, action_dim)
                actions = client.infer(obs)["actions"]
                for a in actions[: args.actions_per_chunk]:
                    queue.append(np.asarray(a))

            yam.command_action(queue.popleft())

            # keep a steady control rate
            sleep = period - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
