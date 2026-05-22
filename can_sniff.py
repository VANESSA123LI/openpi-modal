"""Read-only CAN sniffer for a candleLight/gs_usb CANable on macOS.

Listens for frames on the bus (e.g. YAM motor feedback) and prints them.
It does NOT transmit anything, so the arm will not move.

    python can_sniff.py            # 1 Mbps, listen 5s
    python can_sniff.py --bitrate 500000 --seconds 8
"""

import argparse
import time

import can
import usb.core

VID, PID = 0x1D50, 0x606F  # CANable 2.0 / candleLight gs_usb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if dev is None:
        raise SystemExit("CANable (1d50:606f) not found on USB. Replug it.")

    print(f"Opening gs_usb bus @ {args.bitrate} bps (read-only, {args.seconds}s)...")
    bus = can.Bus(interface="gs_usb", channel=dev.product, index=0, bitrate=args.bitrate)

    n = 0
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            msg = bus.recv(timeout=0.5)
            if msg is not None:
                n += 1
                print(f"  id=0x{msg.arbitration_id:X}  dlc={msg.dlc}  data={msg.data.hex()}")
    finally:
        bus.shutdown()

    print(f"\nReceived {n} frame(s).")
    if n == 0:
        print(
            "No frames seen. Either the bitrate is wrong, or the YAM motors are\n"
            "query-only (they reply to requests rather than broadcasting). A passive\n"
            "listen can't confirm those without sending a request frame."
        )


if __name__ == "__main__":
    main()
