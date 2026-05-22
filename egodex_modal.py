"""Modal pipeline: Apple EgoDex -> LeRobot dataset for pi0.5 co-training.

EgoDex (github.com/apple/ml-egodex, CC-BY-NC-ND) ships 1080p@30Hz egocentric
video WITH ground-truth 3D pose (head/arms/wrists + 25 joints per hand) from
Apple Vision Pro SLAM. So unlike raw video, we get clean wrist trajectories for
free -- no MediaPipe needed.

This converts EgoDex episodes into a LeRobot dataset whose action space is a
20-dim bimanual wrist+gripper representation that FITS pi0.5 (<=32 action dims):

    per hand: [ d_wrist_pos(3), wrist_rot6d_next(6), gripper(1) ]  x 2 hands = 20

Poses are re-expressed in the CAMERA frame (like the EgoDex benchmark) so the
action is invariant to head motion. Gripper is a proxy = thumb-tip <-> index-tip
distance. Language annotation (f.attrs['llm_description']) becomes the task prompt.

IMPORTANT this is PRETRAINING/PRIOR data (human embodiment). To drive a YAM you
still need to retarget this action space to the robot and co-train with real
robot demos. Also mind the CC-BY-NC-ND license for any non-research use.

Stages
------
  download_egodex   : fetch + unzip a split (default 'test', 16GB) into a volume
  egodex_to_lerobot : HDF5 + mp4 -> LeRobot dataset

  modal run egodex_modal.py::download_egodex --split test
  modal run egodex_modal.py::egodex_to_lerobot --split test --repo-id you/egodex_test --max-episodes 50
"""

from pathlib import Path

import modal

app = modal.App("openpi-egodex")

data_vol = modal.Volume.from_name("openpi-data", create_if_missing=True)
VOLUMES = {"/data": data_vol}

BASE_URL = "https://ml-site.cdn-apple.com/datasets/egodex"
RAW_ROOT = "/data/egodex"            # extracted: /data/egodex/<split>/<task>/<idx>.{hdf5,mp4}
LEROBOT_HOME = "/data/lerobot"

# Sizes per split (for cost awareness): part1..part5 = 300GB each, extra=200GB, test=16GB.
SPLIT_ZIPS = {
    "test": ["test.zip"],
    "extra": ["extra.zip"],
    "part1": ["part1.zip"], "part2": ["part2.zip"], "part3": ["part3.zip"],
    "part4": ["part4.zip"], "part5": ["part5.zip"],
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "unzip", "curl", "libgl1", "libglib2.0-0")
    .pip_install("h5py", "opencv-python-headless", "numpy", "lerobot")
    .env({"HF_LEROBOT_HOME": LEROBOT_HOME})
)

HOURS = 60 * 60


@app.function(image=image, volumes=VOLUMES, timeout=12 * HOURS)
def download_egodex(split: str = "test"):
    """Stream-download + unzip one EgoDex split into the volume.

    Default 'test' (16GB) is the cheap way to validate the whole pipeline before
    committing to a 300GB training part.
    """
    import subprocess

    assert split in SPLIT_ZIPS, f"split must be one of {list(SPLIT_ZIPS)}"
    dest = Path(RAW_ROOT)
    dest.mkdir(parents=True, exist_ok=True)
    for zipname in SPLIT_ZIPS[split]:
        url = f"{BASE_URL}/{zipname}"
        zpath = dest / zipname
        print(f"Downloading {url} ...", flush=True)
        subprocess.run(["curl", "-fL", url, "-o", str(zpath)], check=True)
        print(f"Unzipping {zipname} ...", flush=True)
        subprocess.run(["unzip", "-q", "-o", str(zpath), "-d", str(dest)], check=True)
        zpath.unlink()  # reclaim space
        data_vol.commit()
    print("Done. Episodes under", dest)


# ---- pose math --------------------------------------------------------------
def _inv_se3(T):
    import numpy as np
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def _rot6d(R):
    # 6D rotation = first two columns of the rotation matrix (Zhou et al.)
    return R[:3, :2].T.reshape(-1)


@app.function(image=image, volumes=VOLUMES, timeout=12 * HOURS)
def egodex_to_lerobot(
    split: str = "test",
    repo_id: str = "you/egodex_test",
    target_fps: int = 10,
    max_episodes: int = 50,
    image_size: int = 224,
    thumb_key: str = "leftThumbTip",
    index_key: str = "leftIndexFingerTip",
):
    """Convert EgoDex HDF5 + mp4 episodes to a LeRobot dataset.

    NOTE on fingertip keys: EgoDex has ~25 joints/hand; the exact key strings for
    fingertips (e.g. 'leftThumbTip' vs 'leftThumbFingerTip') should be verified
    against a real file -- inspect with `h5py` and adjust thumb_key/index_key.
    """
    import cv2
    import h5py
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.image": {
            "dtype": "video", "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (20,), "names": None},
        "action": {"dtype": "float32", "shape": (20,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=target_fps, features=features)

    episodes = sorted(Path(RAW_ROOT, split).glob("*/*.hdf5"))[:max_episodes]
    assert episodes, f"No .hdf5 under {RAW_ROOT}/{split} -- run download_egodex first"
    print(f"Converting {len(episodes)} episodes from split '{split}'")

    SRC_FPS = 30
    stride = max(1, round(SRC_FPS / target_fps))

    def hand_features(f, side, t):
        """Return (pos[3], rot6d[6], grip[1]) for one hand at frame t, in camera frame."""
        T_cam_inv = _inv_se3(f["transforms/camera"][t])
        wrist_cam = T_cam_inv @ f[f"transforms/{side}Hand"][t]
        pos = wrist_cam[:3, 3].astype(np.float32)
        rot6d = _rot6d(wrist_cam[:3, :3]).astype(np.float32)
        tk = thumb_key if side == "left" else thumb_key.replace("left", "right")
        ik = index_key if side == "left" else index_key.replace("left", "right")
        thumb = f[f"transforms/{tk}"][t][:3, 3]
        index = f[f"transforms/{ik}"][t][:3, 3]
        grip = np.float32(np.linalg.norm(thumb - index))
        return pos, rot6d, np.array([grip], dtype=np.float32)

    for ep in episodes:
        mp4 = ep.with_suffix(".mp4")
        if not mp4.exists():
            continue
        with h5py.File(ep, "r") as f:
            N = f["transforms/camera"].shape[0]
            prompt = f.attrs.get("llm_description", "manipulation demo")
            if isinstance(prompt, bytes):
                prompt = prompt.decode()

            ts = list(range(0, N, stride))
            states, actions = [], []
            for t in ts:
                lp, lr, lg = hand_features(f, "left", t)
                rp, rr, rg = hand_features(f, "right", t)
                states.append(np.concatenate([lp, lr, lg, rp, rr, rg]))  # 20
            states = np.stack(states)
            # action_t = relative wrist motion to next sampled frame + next gripper
            for i in range(len(ts) - 1):
                s, s_next = states[i], states[i + 1]
                act = s_next.copy()
                act[0:3] -= s[0:3]      # left  d_pos
                act[10:13] -= s[10:13]  # right d_pos  (left block = 10 dims)
                actions.append(act.astype(np.float32))

        # decode the matching frames
        cap = cv2.VideoCapture(str(mp4))
        frames = {}
        want = set(ts[:-1])
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (image_size, image_size))
                frames[i] = fr
            i += 1
        cap.release()

        for i, t in enumerate(ts[:-1]):
            if t not in frames:
                continue
            ds.add_frame(
                {
                    "observation.images.image": frames[t],
                    "observation.state": states[i],
                    "action": actions[i],
                },
                task=str(prompt),
            )
        ds.save_episode()
        print(f"  + {ep.parent.name}/{ep.stem}: {len(ts)-1} steps")

    data_vol.commit()
    print(f"\nWrote LeRobot dataset '{repo_id}' under {LEROBOT_HOME}")
    print("Next: add a pi05 LoRA config whose data transform maps "
          "observation.images.image/state/action into the model, then train.")
