---
name: Sensor listen() callbacks must be wrapped in try/except
description: Unguarded Python exceptions in CARLA sensor.listen() callbacks propagate into C++ callback thread and can corrupt Boost.Asio io_context
type: project
---

**Confirmed 2026-03-30.** Applied try/except guards to both camera and collision sensor callbacks in spawn_ego_and_sensors() (cell 14ff8a52ff72 of 01_data_collection.ipynb).

**Why:** CARLA's Python bindings dispatch sensor callbacks from a C++ thread. If a Python exception propagates out of the callback (e.g., MemoryError from queue.put() under high memory pressure, or any other runtime error), it can corrupt the internal Boost.Asio io_context state. Over time this leads to STATUS_STACK_BUFFER_OVERRUN. The camera callback is highest risk because it processes 800x600x4 byte image buffers at 20 FPS.

**Pattern:**
```python
def _camera_callback(image):
    try:
        image_queue.put(image)
    except Exception:
        pass  # drop frame rather than destabilize C++ thread
camera.listen(_camera_callback)
```

**How to apply:** Every sensor.listen() callback in this codebase must have a top-level try/except. For collision sensors, catch the exception but still record the event (append "unknown" as fallback).
