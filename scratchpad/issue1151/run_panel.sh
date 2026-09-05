#!/bin/bash
set -u
cd /home/user/discopt
echo "=== load gate ==="; uptime
.venv/bin/python -u scratchpad/issue1151/panel.py --time-limit 20 \
    --out scratchpad/issue1151/panel.json
echo "rc=$?"
echo "=== PANEL DONE ==="
