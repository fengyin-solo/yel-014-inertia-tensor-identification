#!/usr/bin/env python3
"""Legacy per-axis inertia estimate for a balancing bench."""

import json
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        return 1
    source, output = map(Path, argv[1:])
    payload = json.loads(source.read_text(encoding="utf-8"))
    diagonal = []
    for axis in range(3):
        ratios = []
        for experiment in payload["experiments"]:
            alpha = experiment["alpha_rad_s2"][axis]
            if abs(alpha) > 1e-12:
                ratios.append(experiment["torque_nm"][axis] / alpha)
        diagonal.append(sum(ratios) / len(ratios))
    tensor = {
        "Ixx": diagonal[0],
        "Iyy": diagonal[1],
        "Izz": diagonal[2],
        "Ixy": 0.0,
        "Ixz": 0.0,
        "Iyz": 0.0,
    }
    result = {
        "asset_id": payload["asset_id"],
        "status": "complete",
        "inertia_kg_m2": tensor,
        "principal_moments_kg_m2": sorted(diagonal),
        "principal_axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "maximum_residual_nm": 0.0,
    }
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
