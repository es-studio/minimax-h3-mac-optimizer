#!/usr/bin/env bash
set -euo pipefail
LOG="/Users/eunsung/minimax/memory_watch.log"
PROMPT_ID="4c03202d-6a8f-4504-be49-c6ae92c69e95"
INTERVAL=180

snapshot() {
  python3.12 - "$PROMPT_ID" <<'PY'
import json, subprocess, time, urllib.request, sys
prompt_id = sys.argv[1]
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
ps = subprocess.check_output(
    ["ps", "-axo", "pid,rss,pcpu,pmem,command"], text=True
).splitlines()
comfy = [ln for ln in ps if "main.py" in ln and "ComfyUI" in ln or ("main.py" in ln and "--listen" in ln)]
try:
    q = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5).read())
    running, pending = len(q.get("queue_running", [])), len(q.get("queue_pending", []))
except Exception as e:
    running, pending = f"err:{e}", ""
try:
    hist = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=5).read())
    status = (hist.get(prompt_id) or {}).get("status", {})
except Exception:
    status = {}
free = stats.get("Pages free", 0)
inactive = stats.get("Pages inactive", 0)
spec = stats.get("Pages speculative", 0)
wired = stats.get("Pages wired down", 0)
active = stats.get("Pages active", 0)
comp = stats.get("Pages occupied by compressor", 0)
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print(f"  RAM 24GB | free {free/1e9:.2f} | active {active/1e9:.2f} | wired {wired/1e9:.2f} | compressor {comp/1e9:.2f} GB")
print(f"  avail(free+inactive+spec) {(free+inactive+spec)/1e9:.2f} GB")
print(f"  {swap}")
print(f"  queue running={running} pending={pending} history={status.get('status_str','none')}")
for ln in comfy:
    print(f"  PROC {ln.strip()}")
print()
PY
}

while true; do
  snapshot | tee -a "$LOG"
  sleep "$INTERVAL"
done
