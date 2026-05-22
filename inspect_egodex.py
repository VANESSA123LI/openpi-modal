"""Dump the structure of one EgoDex HDF5 so we can lock down key names/shapes
before converting or downloading anything heavy.

    modal run inspect_egodex.py::inspect --split test
"""

from pathlib import Path

import modal

app = modal.App("openpi-egodex-inspect")
data_vol = modal.Volume.from_name("openpi-data", create_if_missing=True)
RAW_ROOT = "/data/egodex"

image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py", "numpy")


@app.function(image=image, volumes={"/data": data_vol}, timeout=600)
def inspect(split: str = "test"):
    import h5py

    files = sorted(Path(RAW_ROOT, split).glob("*/*.hdf5"))
    assert files, f"No .hdf5 under {RAW_ROOT}/{split} -- run download_egodex first"
    ep = files[0]
    print(f"Inspecting {ep}  (of {len(files)} episodes)\n")

    with h5py.File(ep, "r") as f:
        print("=== datasets (key: shape dtype) ===")
        keys = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)
                print(f"  {name}: {obj.shape} {obj.dtype}")

        f.visititems(visit)

        print("\n=== file attrs (language / metadata) ===")
        for k, v in f.attrs.items():
            sv = v.decode() if isinstance(v, bytes) else v
            print(f"  {k} = {sv}")

        # Highlight the keys the converter depends on
        print("\n=== converter-relevant keys present? ===")
        for k in ["transforms/camera", "transforms/leftHand", "transforms/rightHand",
                  "camera/intrinsic"]:
            print(f"  {k:35s} {'FOUND' if k in keys else 'MISSING'}")
        print("\n  fingertip-ish keys (set thumb_key/index_key from these):")
        for k in keys:
            kl = k.lower()
            if "tip" in kl or "thumb" in kl or "index" in kl:
                print(f"    {k}")

    mp4 = ep.with_suffix(".mp4")
    print(f"\nPaired video {mp4.name}: {'present' if mp4.exists() else 'MISSING'}")
