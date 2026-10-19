"""Send a message of an outside sensor (a smoke detector, a door contact, the script of a hub) to the sensor door of web_app.py.

    export HW3_SENSOR_TOKEN="the same secret that web_app.py got"
    python3 sensor_send.py smoke-1 alarm                        # a signed message (X-Signature, X-Timestamp): safe on a network
    python3 sensor_send.py smoke-1 clear
    python3 sensor_send.py smoke-1 alarm --detail "room 2" --url http://192.168.1.20:8001/api/sensor
    python3 sensor_send.py smoke-1 alarm --unsigned             # the plain secret in X-Token (only for a hub that cannot sign; the secret travels in clear text)

The signature is HMAC-SHA256 of "<timestamp>.<body>" with the secret: the secret itself is never sent and a copied message cannot be sent again.
Only the Python standard library is used, so the script can run on a small board (Raspberry Pi) next to the detector.
"""
import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sensor", help="name of the sensor: letters, digits, _ . - (up to 40 characters)")
    ap.add_argument("state", choices=["alarm", "clear"])
    ap.add_argument("--url", default="http://127.0.0.1:8001/api/sensor")
    ap.add_argument("--detail", default="")
    ap.add_argument("--unsigned", action="store_true", help="send the plain secret in X-Token instead of a signature")
    a = ap.parse_args()
    secret = os.environ.get("HW3_SENSOR_TOKEN", "")
    if len(secret) < 12:
        print("set HW3_SENSOR_TOKEN (the same secret as web_app.py has, at least 12 characters)", file=sys.stderr)
        return 2
    body = json.dumps(dict(sensor=a.sensor, alarm=(a.state == "alarm"), detail=a.detail)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if a.unsigned:
        headers["X-Token"] = secret
    else:
        ts = str(int(time.time()))
        headers["X-Timestamp"] = ts
        headers["X-Signature"] = hmac.new(secret.encode("utf-8"), ts.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
    req = urllib.request.Request(a.url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print(r.status, r.read().decode("utf-8", "replace"))
            return 0
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode("utf-8", "replace"), file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach {a.url}: {type(e).__name__}: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
