#!/usr/bin/env bash
set -euo pipefail
LOG="/Users/eunsung/minimax/t2v_progress.log"
STATE="/Users/eunsung/minimax/t2v_1344_progress.json"
INTERVAL=180

snapshot() {
  python3.12 - <<'PY'
import json, subprocess, time, urllib.request
from pathlib import Path

state_path = Path("/Users/eunsung/minimax/t2v_1344_progress.json")
prompt_id = None
start = None
if state_path.exists():
    st = json.loads(state_path.read_text())
    prompt_id = st.get("prompt_id")
    start = st.get("start_epoch")
elapsed = (time.time() - start) if start else None

def get(path):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188" + path, timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}

q = get("/queue")
hist = get(f"/history/{prompt_id}") if prompt_id else {}
page = int(subprocess.check_output(["pagesize"]).decode())
vm = subprocess.check_output(["vm_stat"]).decode().splitlines()
stats = {}
for line in vm[1:]:
    if ":" not in line:
        continue
    k, v = line.split(":", 1)
    num = "".join(ch for ch in v if ch.isdigit())
    if num:
        stats[k.strip()] = int(num) * page
swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode().strip()
ps = subprocess.check_output(["ps", "-axo", "pid,rss,pcpu,pmem,command"], text=True)
comfy = next((ln for ln in ps.splitlines() if "main.py" in ln and "--listen" in ln), "")

running = len(q.get("queue_running", [])) if isinstance(q, dict) else "?"
pending = len(q.get("queue_pending", [])) if isinstance(q, dict) else "?"
status = "none"
outputs = None
if isinstance(hist, dict) and prompt_id in hist:
    rec = hist[prompt_id]
    status = (rec.get("status") or {}).get("status_str", "unknown")
    outputs = rec.get("outputs")

comp = stats.get("Pages occupied by compressor", 0)
free = stats.get("Pages free", 0)
wired = stats.get("Pages wired down", 0)
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print(f"  elapsed {elapsed/60:.1f} min" if elapsed is not None else "  elapsed n/a")
print(f"  job {prompt_id}  queue running={running} pending={pending}  history={status}")
print(f"  RAM compressor {comp/1e9:.2f} GB  free {free/1e9:.2f} GB  wired {wired/1e9:.2f} GB")
print(f"  {swap}")
if comfy:
    parts = comfy.split(None, 4)
    if len(parts) >= 4:
        rss_mb = int(parts[1]) / 1024
        print(f"  ComfyUI RSS {rss_mb:.0f} MB  CPU {parts[2]}%  MEM {parts[3]}%")
if outputs:
    print(f"  outputs {json.dumps(outputs)[:400]}")
print()
PY
}

{
  echo "===== T2V 1344x768 5s progress (every 3 min) ====="
  echo
} > "$LOG"

while true; do
  snapshot | tee -a "$LOG"
  sleep "$INTERVAL"
done
