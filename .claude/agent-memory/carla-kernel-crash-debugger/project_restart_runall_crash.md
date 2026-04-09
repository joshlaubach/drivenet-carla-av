---
name: Restart-and-Run-All crash when all towns already collected
description: 0xC0000409 crash triggered by unnecessary CARLA kill/relaunch cycles when no collection work is needed; fixed with _town_complete() guard
type: project
---

**Confirmed 2026-03-30.** All 6 towns fully collected (54/54 conditions, 7 chunks each). "Restart and Run All" triggered 6 unnecessary restart_carla_and_reconnect() cycles. One of the kill/relaunch cycles triggered STATUS_STACK_BUFFER_OVERRUN (0xC0000409) in libcarla's Boost.Asio thread.

**Why:** Even without sensor streaming, the kill/relaunch cycle itself can crash the kernel. _disconnect_client() tears down the Boost.Asio io_context, then proc.kill() terminates CARLA from the OS side. If C++ background threads haven't fully drained before the TCP connection drops, Fast-Fail fires.

**Fix applied:** Added `_town_complete(town)` guard to all 6 runner cells. Checks checkpoint.json conditions_completed against the full condition count for that town. Short-circuits before any CARLA interaction when the town is already complete.

**How to apply:** Any new runner cells or batch-run patterns must check _town_complete() before calling restart_carla_and_reconnect(). The guard is in cell gbm1luvul7n of 01_data_collection.ipynb.
