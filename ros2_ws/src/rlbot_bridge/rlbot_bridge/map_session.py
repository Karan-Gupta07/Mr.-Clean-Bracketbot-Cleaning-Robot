"""Save occupancy images and the SLAM pose graph into a new directory.

    ros2 run rlbot_bridge save_map out/maps/room_run_1

Run beside SLAM Toolbox on the same machine/filesystem. Stop driving first.
Reload with mapping.launch.py map_file:=/absolute/path/to/room_run_1/map.
"""

import argparse
import math
from pathlib import Path
import re

import rclpy
from slam_toolbox.srv import SaveMap, SerializePoseGraph


def call_service(node, service_type, name, request, timeout):
    client = node.create_client(service_type, name)
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service unavailable: {name}; is SLAM Toolbox active in mapping mode?")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError(f"service timed out: {name}")
        result = future.result()
        if result is None or result.result != 0:
            raise RuntimeError(f"{name} failed: {result}")
    finally:
        node.destroy_client(client)


def save_session(node, directory, timeout=30.0):
    directory = Path(directory).expanduser().resolve()
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(directory)):
        raise ValueError("SLAM Toolbox's map saver requires a path without spaces or special characters")
    directory.mkdir(parents=True, exist_ok=False)
    stem = directory / "map"
    image = SaveMap.Request()
    image.name.data = str(stem)
    call_service(node, SaveMap, "/slam_toolbox/save_map", image, timeout)
    graph = SerializePoseGraph.Request()
    graph.filename = str(stem)
    call_service(node, SerializePoseGraph, "/slam_toolbox/serialize_map", graph, timeout)
    for suffix in (".yaml", ".pgm", ".posegraph", ".data"):
        path = Path(str(stem) + suffix)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"save returned success but file is missing/empty: {path}")
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    rclpy.init(args=[])
    node = rclpy.create_node("save_mapping_session")
    try:
        stem = save_session(node, args.directory, args.timeout)
        print(f"Saved occupancy map and reloadable SLAM state: {stem}")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Save failed: {error}\nAny partial output has been left intact; use a new directory when retrying.\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
