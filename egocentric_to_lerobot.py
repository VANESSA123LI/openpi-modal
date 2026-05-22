"""Modal pipeline: raw egocentric video -> LeRobot dataset for pi0.5 co-training.

You cannot train pi0.5 on raw video: its loss is action prediction, so every
frame needs an action target. Following EgoMimic (egomimic.github.io), we derive
a *pseudo-action* from the human hand:

    action_t = [ relative wrist motion (t -> t+1) ,  gripper open/close ]

Relative wrist motion is used (not absolute) so it's invariant to head/camera
movement -- the same trick that lets human and robot data share an action space.
The result is written as a LeRobot dataset tagged as the "human" embodiment, with
images masked to hide the human arm, ready to be MIXED with real YAM teleop data
during a pi0.5 fine-tune (human-only will give a prior, not a deployable policy).

Stages
------
  1. extract_poses : video -> per-frame hand pose (wrist + 21 keypoints + handedness)
  2. build_dataset : poses -> relative-delta actions + gripper -> LeRobot dataset

Quality ceiling is set by the pose stage. Defaults to MediaPipe Hands (light,
runs anywhere) which gives 2.5D landmarks -> good gripper proxy, rough wrist
motion. For metric 3D wrist trajectories swap in HaMeR/WiLoR (3D hand mesh) or,
best of all, record with Project Aria and use its MPS hand-tracking + SLAM pose.

  modal run egocentric_to_lerobot.py::extract_poses --video clip01.mp4
  modal run egocentric_to_lerobot.py::build_dataset --repo-id you/ego_yam_human
"""

import json
from pathlib import Path

import modal

app = modal.App("openpi-ego-prep")

# Reuse the same data volume the training app reads from.
data_vol = modal.Volume.from_name("openpi-data", create_if_missing=True)
VOLUMES = {"/data": data_vol}

RAW_DIR = "/data/ego_raw"          # put your *.mp4 here
POSE_DIR = "/data/ego_poses"       # per-clip pose arrays land here
LEROBOT_HOME = "/data/lerobot"     # LeRobot dataset cache

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "opencv-python-headless",
        "mediapipe==0.10.14",   # swap for hamer/wilor for metric 3D wrist pose
        "numpy",
        "lerobot",              # writes the LeRobot v2 dataset
    )
    .env({"HF_LEROBOT_HOME": LEROBOT_HOME})
)

HOURS = 60 * 60


# ---------------------------------------------------------------------------
# Stage 1: per-frame hand pose
# ---------------------------------------------------------------------------
@app.function(image=image, volumes=VOLUMES, timeout=4 * HOURS)
def extract_poses(video: str, fps: int = 10):
    """Decode one clip and dump per-frame hand pose to POSE_DIR/<clip>.json.

    Output per frame: wrist (x,y,z_rel) in [0,1] image coords + relative depth,
    21 hand landmarks, handedness, and a gripper scalar (thumb-index distance).
    """
    import cv2
    import mediapipe as mp
    import numpy as np

    src = Path(RAW_DIR) / video
    assert src.exists(), f"{src} not found -- upload clips to {RAW_DIR} first"

    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / fps))

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )

    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            rec = {"t": i / src_fps, "found": False}
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm])  # (21,3)
                wrist = pts[0]
                # gripper proxy: normalized thumb-tip(4) <-> index-tip(8) distance
                grip = float(np.linalg.norm(pts[4, :2] - pts[8, :2]))
                rec.update(
                    found=True,
                    wrist=wrist.tolist(),
                    keypoints=pts.tolist(),
                    gripper=grip,
                )
            frames.append(rec)
        i += 1
    cap.release()

    out = Path(POSE_DIR) / (Path(video).stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fps": fps, "frames": frames}))
    data_vol.commit()
    found = sum(f["found"] for f in frames)
    print(f"{video}: {len(frames)} frames, hand found in {found}")


# ---------------------------------------------------------------------------
# Stage 2: poses -> relative-delta actions -> LeRobot dataset
# ---------------------------------------------------------------------------
@app.function(image=image, volumes=VOLUMES, timeout=4 * HOURS)
def build_dataset(repo_id: str, prompt: str = "manipulation demo", gripper_thresh: float = 0.08):
    """Turn extracted poses into a LeRobot dataset of (state, action) pairs.

    action = [d_wrist_x, d_wrist_y, d_wrist_z, d_gripper]   (relative, per step)
    state  = [wrist_x, wrist_y, wrist_z, gripper_open]      (current)

    NOTE: this is a 4-dim placeholder action space. To co-train with YAM you must
    RETARGET these to the robot's action representation (e.g. 6DoF wrist delta +
    gripper per arm) so human and robot share one space -- that needs the YAM
    action def, which doesn't exist yet. See the EgoMimic 'wrist-level action +
    embodiment adapter' design. Also: images here are NOT yet masked to hide the
    human arm (add a hand/arm segmentation mask before serious training).
    """
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "action": {"dtype": "float32", "shape": (4,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=10, features=features, use_videos=False)

    clips = sorted(Path(POSE_DIR).glob("*.json"))
    assert clips, f"No pose files in {POSE_DIR} -- run extract_poses first"

    for clip in clips:
        frames = [f for f in json.loads(clip.read_text())["frames"] if f["found"]]
        if len(frames) < 2:
            continue
        wrist = np.array([f["wrist"] for f in frames], dtype=np.float32)      # (T,3)
        grip = np.array([f["gripper"] for f in frames], dtype=np.float32)      # (T,)
        grip_open = (grip > gripper_thresh).astype(np.float32)

        for t in range(len(frames) - 1):
            state = np.concatenate([wrist[t], grip_open[t:t + 1]])
            d_wrist = wrist[t + 1] - wrist[t]                    # relative motion
            d_grip = grip_open[t + 1:t + 2] - grip_open[t:t + 1]
            action = np.concatenate([d_wrist, d_grip]).astype(np.float32)
            ds.add_frame({"observation.state": state, "action": action}, task=prompt)
        ds.save_episode()

    print(f"Wrote LeRobot dataset '{repo_id}' under {LEROBOT_HOME}")
    print("Next: retarget the 4-dim action to the YAM action space, then add a "
          "pi05 co-training config that mixes this with robot teleop data.")
    data_vol.commit()


@app.local_entrypoint()
def main(repo_id: str = "you/ego_yam_human"):
    """Convenience: assumes clips already uploaded to /data/ego_raw."""
    print("Upload clips to the 'openpi-data' volume under ego_raw/, then run "
          "extract_poses per clip, then build_dataset.")
