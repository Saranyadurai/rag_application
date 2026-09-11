"""
List models actually available to your Fireworks account.

Usage:
    $env:FIREWORKS_API_KEY = "fw_your_key_here"
    python list_fireworks_models.py
"""

import os
import requests

API_KEY = os.environ.get("FIREWORKS_API_KEY")
if not API_KEY:
    raise RuntimeError('Set FIREWORKS_API_KEY first: $env:FIREWORKS_API_KEY = "fw_your_key_here"')

resp = requests.get(
    "https://api.fireworks.ai/inference/v1/models",
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()

models = data.get("data", [])
print(f"{len(models)} model(s) available to your account:\n")
for m in models:
    print(f"  {m.get('id')}")
