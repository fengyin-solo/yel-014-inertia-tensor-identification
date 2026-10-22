#!/usr/bin/env python3
"""Recover a symmetric inertia tensor from rigid-body torque measurements."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from fractions import Fraction
from itertools import combinations
from pathlib import Path


INPUT_KEYS = {"asset_id", "experiments", "tolerance", "axis_checks"}
EXPERIMENT_KEYS = {"omega_rad_s", "alpha_rad_s2", "torque_nm"}
AXIS_CHECK_KEYS = {"body", "observed"}
TENSOR_KEYS = ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz")


class Refusal(Exception):
    def __init__(self, kind):
        self.kind = kind


def refuse(kind):
    sys.stderr.write(f"运行拒收: {kind}\n")
    raise SystemExit(1 if kind in {"BAD_INPUT", "IO_FAILURE"} else 2)


def pairs_hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal("BAD_INPUT")
        result[key] = value
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value):
    if not isinstance(value, list) or len(value) != 3 or any(not finite(item) or abs(item) > 1e6 for item in value):
        raise Refusal("BAD_INPUT")
    return [float(item) for item in value]


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != INPUT_KEYS:
        raise Refusal("BAD_INPUT")
    asset_id = payload["asset_id"]
    if (
        not isinstance(asset_id, str)
        or not 2 <= len(asset_id) <= 48
        or asset_id != asset_id.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in asset_id)
    ):
        raise Refusal("BAD_INPUT")
    tolerance = payload["tolerance"]
    if not finite(tolerance) or not 1e-10 <= tolerance <= 1e-3:
        raise Refusal("BAD_INPUT")
    experiments = payload["experiments"]
    if not isinstance(experiments, list) or not 7 <= len(experiments) <= 24:
        raise Refusal("BAD_INPUT")
    clean = []
    for item in experiments:
        if not isinstance(item, dict) or set(item) != EXPERIMENT_KEYS:
            raise Refusal("BAD_INPUT")
        clean.append(
            {
                "omega": vector(item["omega_rad_s"]),
                "alpha": vector(item["alpha_rad_s2"]),
                "torque": vector(item["torque_nm"]),
            }
        )
    axis_checks = payload["axis_checks"]
    if not isinstance(axis_checks, list) or not 4 <= len(axis_checks) <= 6:
        raise Refusal("BAD_INPUT")
    clean_axes = []
    for item in axis_checks:
        if not isinstance(item, dict) or set(item) != AXIS_CHECK_KEYS:
            raise Refusal("BAD_INPUT")
        clean_axes.append(
            {
                "body": vector(item["body"]),
                "observed": vector(item["observed"]),
            }
        )
    return asset_id, clean, float(tolerance), clean_axes


def inverse_matrix(matrix):
    size = len(matrix)
    augmented = [
        [Fraction(value) for value in matrix[row]]
        + [Fraction(1 if row == column else 0) for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            raise Refusal("BAD_MODEL")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size * 2)
            ]
    return [row[size:] for row in augmented]


def matrix_product(left, right):
    return [
        [
            sum(left[row][index] * right[index][column] for index in range(len(right)))
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def transform_vector(matrix, vector):
    return [
        sum(matrix[row][column] * vector[column] for column in range(len(vector)))
        for row in range(len(matrix))
    ]


def recover_axis_mapping(axis_checks):
    supported = {}
    for subset in combinations(axis_checks, 3):
        body_columns = [
            [Fraction(item["body"][row]) for item in subset]
            for row in range(3)
        ]
        observed_columns = [
            [Fraction(item["observed"][row]) for item in subset]
            for row in range(3)
        ]
        try:
            inverse = inverse_matrix(body_columns)
        except Refusal:
            continue
        matrix = matrix_product(observed_columns, inverse)
        if any(value.denominator != 1 for row in matrix for value in row):
            continue
        integer_matrix = [[int(value) for value in row] for row in matrix]
        if abs(round(matrix_determinant(integer_matrix))) != 1:
            continue
        score = sum(
            transform_vector(integer_matrix, item["body"]) == item["observed"]
            for item in axis_checks
        )
        key = tuple(tuple(row) for row in integer_matrix)
        supported[key] = max(score, supported.get(key, -1))
    if not supported:
        raise Refusal("BAD_MODEL")
    best = max(supported.values())
    winners = [matrix for matrix, score in supported.items() if score == best]
    if best < len(axis_checks) - 1 or len(winners) != 1:
        raise Refusal("BAD_MODEL")
    matrix = [[float(value) for value in row] for row in winners[0]]
    inverse = inverse_matrix(matrix)
    return matrix, inverse


def matrix_determinant(matrix):
    work = [[Fraction(value) for value in row] for row in matrix]
    value = Fraction(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            value *= -1
        value *= work[column][column]
        for row in range(column + 1, len(work)):
            factor = work[row][column] / work[column][column]
            work[row] = [
                work[row][index] - factor * work[column][index]
                for index in range(len(work))
            ]
    return value


def to_body(value, inverse):
    transformed = transform_vector(inverse, value)
    return [float(item) for item in transformed]


def equation_row(omega, alpha):
    wx, wy, wz = omega
    ax, ay, az = alpha
    rows = [
        [ax, -wz * wy, wy * wz, ay - wz * wx, az + wy * wx, wy * wy - wz * wz],
        [wz * wx, ay, -wx * wz, ax + wz * wy, wz * wz - wx * wx, az - wx * wy],
        [-wy * wx, wx * wy, az, wx * wx - wy * wy, ax - wy * wz, ay + wx * wz],
    ]
    return rows


def solve_system(matrix, right):
    size = len(matrix)
    augmented = [matrix[row][:] + [right[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            raise Refusal("BAD_MODEL")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[index][-1] for index in range(size)]


def fit_tensor(experiments):
    normal = [[0.0] * 6 for _ in range(6)]
    right = [0.0] * 6
    equation_rows = []
    for experiment in experiments:
        for row, target in zip(equation_row(experiment["omega"], experiment["alpha"]), experiment["torque"]):
            equation_rows.append((row, target))
            for left in range(6):
                right[left] += row[left] * target
                for right_index in range(6):
                    normal[left][right_index] += row[left] * row[right_index]
    coefficients = solve_system(normal, right)
    residual = max(
        abs(sum(left * value for left, value in zip(row, coefficients)) - target)
        for row, target in equation_rows
    )
    if not math.isfinite(residual):
        raise Refusal("BAD_MODEL")
    matrix = [
        [coefficients[0], coefficients[3], coefficients[4]],
        [coefficients[3], coefficients[1], coefficients[5]],
        [coefficients[4], coefficients[5], coefficients[2]],
    ]
    return coefficients, matrix, residual


def identify_tensor(experiments, tolerance):
    try:
        coefficients, matrix, residual = fit_tensor(experiments)
        if residual <= tolerance:
            return coefficients, matrix, residual
    except Refusal:
        pass
    for removed_count in range(1, min(3, len(experiments) - 6) + 1):
        matches = []
        for removed in combinations(range(len(experiments)), removed_count):
            removed_set = set(removed)
            subset = [
                item for index, item in enumerate(experiments)
                if index not in removed_set
            ]
            try:
                candidate = fit_tensor(subset)
            except Refusal:
                continue
            if candidate[2] <= tolerance:
                matches.append(candidate)
        if matches:
            if len(matches) != 1:
                raise Refusal("BAD_MODEL")
            return matches[0]
    raise Refusal("BAD_MODEL")


def principal_axes(matrix):
    values = [row[:] for row in matrix]
    vectors = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    for _ in range(64):
        pairs = [(abs(values[row][column]), row, column) for row in range(3) for column in range(row + 1, 3)]
        off, first, second = max(pairs)
        if off < 1e-14:
            break
        angle = 0.5 * math.atan2(2.0 * values[first][second], values[second][second] - values[first][first])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for index in range(3):
            left = values[index][first]
            right = values[index][second]
            values[index][first] = cosine * left - sine * right
            values[index][second] = sine * left + cosine * right
        for index in range(3):
            left = values[first][index]
            right = values[second][index]
            values[first][index] = cosine * left - sine * right
            values[second][index] = sine * left + cosine * right
        for index in range(3):
            left = vectors[index][first]
            right = vectors[index][second]
            vectors[index][first] = cosine * left - sine * right
            vectors[index][second] = sine * left + cosine * right
    pairs = []
    for column in range(3):
        vector = [vectors[row][column] for row in range(3)]
        norm = math.sqrt(sum(value * value for value in vector))
        vector = [value / norm for value in vector]
        pivot = max(range(3), key=lambda index: abs(vector[index]))
        if vector[pivot] < 0:
            vector = [-value for value in vector]
        pairs.append((values[column][column], vector))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return [item[0] for item in pairs], [item[1] for item in pairs]


def publish(output, payload):
    if not output.parent.is_dir() or output.is_dir() or not os.access(output.parent, os.W_OK):
        raise Refusal("IO_FAILURE")
    descriptor = None
    stage = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".inertia-", dir=output.parent)
        stage = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, output)
        stage = None
    except OSError:
        raise Refusal("IO_FAILURE")
    finally:
        if stage is not None:
            stage.unlink(missing_ok=True)


def main(argv):
    if len(argv) != 3:
        refuse("BAD_INPUT")
    source, output = map(Path, argv[1:])
    try:
        payload = json.loads(source.read_bytes().decode("utf-8"), object_pairs_hook=pairs_hook)
        asset_id, experiments, tolerance, axis_checks = validate(payload)
        _, inverse = recover_axis_mapping(axis_checks)
        experiments = [
            {
                "omega": to_body(item["omega"], inverse),
                "alpha": to_body(item["alpha"], inverse),
                "torque": to_body(item["torque"], inverse),
            }
            for item in experiments
        ]
        coefficients, matrix, residual = identify_tensor(experiments, tolerance)
        moments, axes = principal_axes(matrix)
        if any(not math.isfinite(value) or value <= 0 for value in moments):
            raise Refusal("BAD_MODEL")
        result = {
            "asset_id": asset_id,
            "status": "complete",
            "inertia_kg_m2": {
                "Ixx": coefficients[0],
                "Iyy": coefficients[1],
                "Izz": coefficients[2],
                "Ixy": coefficients[3],
                "Ixz": coefficients[4],
                "Iyz": coefficients[5],
            },
            "principal_moments_kg_m2": moments,
            "principal_axes": axes,
            "maximum_residual_nm": residual,
        }
        publish(output, (json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        return 0
    except Refusal as exc:
        refuse(exc.kind)
    except (OSError, UnicodeError, ValueError):
        refuse("BAD_INPUT")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
