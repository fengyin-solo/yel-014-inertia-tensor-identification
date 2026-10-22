"""Acceptance tests for symmetric inertia-tensor identification."""

import json
import os
import subprocess
import sys

import pytest


ENTRY = os.environ.get("INERTIA_ENTRY", "/app/inertia_tensor_probe.py")
VIEW_TRANSFORM = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]


def to_view(vector):
    return [
        sum(VIEW_TRANSFORM[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def tensor_torque(tensor, omega, alpha):
    inertia_omega = [
        sum(tensor[row][column] * omega[column] for column in range(3))
        for row in range(3)
    ]
    gyroscopic = [
        omega[1] * inertia_omega[2] - omega[2] * inertia_omega[1],
        omega[2] * inertia_omega[0] - omega[0] * inertia_omega[2],
        omega[0] * inertia_omega[1] - omega[1] * inertia_omega[0],
    ]
    return [
        sum(tensor[row][column] * alpha[column] for column in range(3)) + gyroscopic[row]
        for row in range(3)
    ]


def record(tensor=None, corrupt_experiments=0, corrupt_axis_check=False):
    tensor = tensor or [
        [2.2, 0.35, -0.18],
        [0.35, 2.9, 0.24],
        [-0.18, 0.24, 3.4],
    ]
    raw = [
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([1.3, -0.7, 0.4], [0.2, -0.3, 0.5]),
        ([-0.6, 1.1, 0.9], [0.7, 0.4, -0.2]),
        ([0.8, 0.5, -1.2], [-0.4, 0.6, 0.3]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.2, -0.4, 0.8], [0.3, 0.5, -0.1]),
    ]
    experiments = []
    for omega, alpha in raw:
        torque = tensor_torque(tensor, omega, alpha)
        experiments.append(
            {
                "omega_rad_s": to_view(omega),
                "alpha_rad_s2": to_view(alpha),
                "torque_nm": to_view(torque),
            }
        )
    for index in range(corrupt_experiments):
        experiments[index]["torque_nm"][index % 3] += 0.5
    axis_checks = []
    for index, body in enumerate(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.4, -0.7, 0.2]]
    ):
        observed = to_view(body)
        if corrupt_axis_check and index == 3:
            observed[1] += 1.0
        axis_checks.append({"body": body, "observed": observed})
    return {
        "asset_id": "BALANCE-RIG-14",
        "experiments": experiments,
        "tolerance": 1e-8,
        "axis_checks": axis_checks,
    }


def run(tmp_path, payload, raw=None):
    source = tmp_path / "record.json"
    output = tmp_path / "result.json"
    if raw is None:
        source.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    else:
        source.write_text(raw, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, source, output


def test_success_is_quiet(tmp_path):
    """题面：「命令成功时，标准输出不得有任何字符。」"""
    result, _, output = run(tmp_path, record())
    assert result.returncode == 0 and result.stdout == "" and result.stderr == "" and output.is_file()


def test_error_channel_is_stderr_only(tmp_path):
    """题面：「出错时唯一可见内容写入标准错误。」"""
    payload = record()
    payload["experiments"] = []
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_full_tensor_reconstruction(tmp_path):
    """题面：「三次单轴测量不足以描述偏轴质量，计算时不能丢掉 `omega x (I * omega)`。」"""
    result, _, output = run(tmp_path, record())
    tensor = json.loads(output.read_text(encoding="utf-8"))["inertia_kg_m2"]
    assert result.returncode == 0
    assert tensor["Ixx"] == pytest.approx(2.2, abs=1e-9)
    assert tensor["Ixy"] == pytest.approx(0.35, abs=1e-9)
    assert tensor["Iyz"] == pytest.approx(0.24, abs=1e-9)


def test_principal_axes_are_orthonormal(tmp_path):
    """题面：「三轴必须两两正交。」"""
    result, _, output = run(tmp_path, record())
    document = json.loads(output.read_text(encoding="utf-8"))
    axes = document["principal_axes"]
    dots = [
        sum(axes[left][index] * axes[right][index] for index in range(3))
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    norms = [sum(value * value for value in axis) for axis in axes]
    assert result.returncode == 0
    assert all(dot == pytest.approx(0.0, abs=1e-9) for dot in dots)
    assert all(norm == pytest.approx(1.0, abs=1e-9) for norm in norms)


def test_principal_moments_are_ascending(tmp_path):
    """题面：「特征值排序为升序，对应的三个单位特征向量就是主轴。」"""
    result, _, output = run(tmp_path, record())
    moments = json.loads(output.read_text(encoding="utf-8"))["principal_moments_kg_m2"]
    assert result.returncode == 0 and moments == sorted(moments)


def test_seven_experiments_are_accepted(tmp_path):
    """题面：「一次交班至少给 7 组、至多 24 组多轴载荷试验。」"""
    payload = record()
    payload["experiments"] = payload["experiments"][:7]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 0 and output.is_file()


def test_six_experiments_are_refused(tmp_path):
    """题面：「一次交班至少给 7 组、至多 24 组多轴载荷试验。」"""
    payload = record()
    payload["experiments"] = payload["experiments"][:6]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_tolerance_minimum_is_accepted(tmp_path):
    """题面：「拟合容差取 `tolerance ∈ [1e-10, 1e-3]`，闭区间。」"""
    payload = record()
    payload["tolerance"] = 1e-10
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 0 and output.is_file()


def test_tolerance_above_upper_bound_is_refused(tmp_path):
    """题面：「拟合容差取 `tolerance ∈ [1e-10, 1e-3]`，闭区间。」"""
    payload = record()
    payload["tolerance"] = 1.1e-3
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_top_level_extra_key_is_refused(tmp_path):
    """题面：「根层级固定为下面三个字段。」"""
    payload = record()
    payload["axis_order"] = ["x", "y", "z"]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_invalid_utf8_record_is_refused(tmp_path):
    """题面：「台架交来的记录只接受 UTF-8 文本。」"""
    source = tmp_path / "record.json"
    output = tmp_path / "result.json"
    source.write_bytes(b"\xff\xfe\x00")
    result = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_duplicate_key_is_refused(tmp_path):
    """题面：「若根层级、实验字段或数值域不合交班格式，脚本返回 1，终端错误行固定为 `运行拒收: BAD_INPUT`。」"""
    raw = (
        '{"asset_id":"RIG-14","experiments":[], "experiments":[],'
        '"tolerance":0.000001,"axis_checks":[]}'
    )
    result, _, output = run(tmp_path, None, raw=raw)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


@pytest.mark.parametrize("field", ["omega_rad_s", "alpha_rad_s2", "torque_nm"])
def test_vector_must_have_three_finite_numbers(tmp_path, field):
    """题面：「每路恰好三个有限实数，绝对值最高到 1e6；布尔值仍按类型错误处理。」"""
    payload = record()
    payload["experiments"][0][field] = [1.0, 2.0]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_boolean_is_not_a_number(tmp_path):
    """题面：「每路恰好三个有限实数，绝对值最高到 1e6；布尔值仍按类型错误处理。」"""
    payload = record()
    payload["experiments"][0]["torque_nm"][0] = True
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_asset_id_rules_are_enforced(tmp_path):
    """题面：「`asset_id` 至少两字、至多四十八字；首尾空格和不可见字符一律拒收。」"""
    payload = record()
    payload["asset_id"] = " RIG-14 "
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_output_key_order_is_exact(tmp_path):
    """题面：「交付单先写设备标签 `asset_id`，再写状态 `status`。」"""
    result, _, output = run(tmp_path, record())
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert list(document) == [
        "asset_id",
        "status",
        "inertia_kg_m2",
        "principal_moments_kg_m2",
        "principal_axes",
        "maximum_residual_nm",
    ]


def test_output_is_compact_with_one_lf(tmp_path):
    """题面：「交付 JSON 不换行、不在逗号或冒号后加空格，文件尾仅放一个 LF。」"""
    result, _, output = run(tmp_path, record())
    payload = output.read_bytes()
    assert result.returncode == 0
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload
    assert b": " not in payload and b", " not in payload


def test_output_tail_key_order_is_exact(tmp_path):
    """题面：「后续四段按下面顺序排列：」"""
    result, _, output = run(tmp_path, record())
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert list(document)[2:] == [
        "inertia_kg_m2",
        "principal_moments_kg_m2",
        "principal_axes",
        "maximum_residual_nm",
    ]


def test_status_is_complete(tmp_path):
    """题面：「`status` 每次都是 `complete`。」"""
    result, _, output = run(tmp_path, record())
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0 and document["status"] == "complete"


def test_inertia_keys_have_fixed_order(tmp_path):
    """题面：「张量内部按 `Ixx`、`Iyy`、`Izz`、`Ixy`、`Ixz`、`Iyz` 排列。」"""
    result, _, output = run(tmp_path, record())
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert list(document["inertia_kg_m2"]) == ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]


def test_axis_order_matches_moments(tmp_path):
    """题面：「主轴顺序必须跟主惯量同序。」"""
    result, _, output = run(tmp_path, record())
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert len(document["principal_axes"]) == len(document["principal_moments_kg_m2"]) == 3


def test_repeat_is_byte_identical(tmp_path):
    """题面：「相同台架记录再次签名时，交付文件应保持第一次的原始字节。」"""
    result, source, output = run(tmp_path, record())
    before = output.read_bytes()
    second = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and second.returncode == 0 and output.read_bytes() == before


def test_rank_deficient_experiments_are_refused(tmp_path):
    """题面：「六元方程的可解性要求试验矩阵达到满秩。」"""
    payload = record()
    payload["experiments"] = [payload["experiments"][0] for _ in range(8)]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 2 and result.stderr == "运行拒收: BAD_MODEL\n" and not output.exists()


def test_rank_failure_is_reported_as_bad_model(tmp_path):
    """题面：「若六元系统凑不出满秩解，脚本返回 2，终端错误行固定为 `运行拒收: BAD_MODEL`。」"""
    payload = record()
    payload["experiments"] = [payload["experiments"][0] for _ in range(8)]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 2 and result.stderr == "运行拒收: BAD_MODEL\n" and not output.exists()


def test_non_positive_definite_tensor_is_refused(tmp_path):
    """题面：「恢复出的 3×3 对称矩阵三个特征值都应为正，否则它不是可交付的物理惯量。」"""
    payload = record(
        [
            [-1.2, 0.1, 0.0],
            [0.1, 2.0, 0.2],
            [0.0, 0.2, 3.0],
        ]
    )
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 2 and result.stderr == "运行拒收: BAD_MODEL\n" and not output.exists()


def test_experiment_extra_key_is_refused(tmp_path):
    """题面：「三路向量缺一不可，也不能夹带第四个字段。」"""
    payload = record()
    payload["experiments"][0]["axis"] = 1
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_residual_above_tolerance_is_refused(tmp_path):
    """题面：「若残差门限失守，脚本同样返回 2，终端错误行固定为 `运行拒收: BAD_MODEL`。」"""
    payload = record()
    payload["tolerance"] = 1e-10
    for index in range(4):
        payload["experiments"][index]["torque_nm"][index % 3] += 0.01
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 2 and result.stderr == "运行拒收: BAD_MODEL\n" and not output.exists()


def test_axis_mapping_and_one_bad_check_are_recovered(tmp_path):
    """题面：「先用 `axis_checks` 恢复矩阵 `A`，并把每组试验逆变换回机体系」。"""
    result, _, output = run(tmp_path, record(corrupt_axis_check=True))
    tensor = json.loads(output.read_text(encoding="utf-8"))["inertia_kg_m2"]
    assert result.returncode == 0
    assert tensor["Ixx"] == pytest.approx(2.2, abs=1e-9)
    assert tensor["Iyz"] == pytest.approx(0.24, abs=1e-9)


def test_one_corrupted_experiment_is_excluded_uniquely(tmp_path):
    """题面：「该最小删除数不得大于 3，且达到它的删除组合必须唯一。」"""
    result, _, output = run(tmp_path, record(corrupt_experiments=1))
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert document["inertia_kg_m2"]["Ixx"] == pytest.approx(2.2, abs=1e-9)
    assert document["maximum_residual_nm"] <= 1e-8


def test_three_corrupted_experiments_are_excluded_uniquely(tmp_path):
    """题面：「至多三组试验可能因触发记录错误而背离真实惯量模型。」"""
    result, _, output = run(tmp_path, record(corrupt_experiments=3))
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert document["inertia_kg_m2"]["Ixx"] == pytest.approx(2.2, abs=1e-9)
    assert document["maximum_residual_nm"] <= 1e-8


def test_axis_check_shape_is_strict(tmp_path):
    """题面：「`axis_checks` 有 4 到 6 项，每项恰好有 `body` 与 `observed` 两个三分量实数向量。」"""
    payload = record()
    del payload["axis_checks"][0]["observed"]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_tied_axis_mapping_is_bad_model(tmp_path):
    """题面：「必须恰有一个置换获得最高支持数。」"""
    payload = record()
    payload["axis_checks"] = [
        {"body": [1.0, 1.0, 1.0], "observed": [1.0, 1.0, 1.0]}
        for _ in range(4)
    ]
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 2 and result.stderr == "运行拒收: BAD_MODEL\n" and not output.exists()


def test_residual_bound_is_reported(tmp_path):
    """题面：「所有实验行中最大的绝对方程残差记作 `maximum_residual_nm`，其上限正是 `tolerance`。」"""
    result, _, output = run(tmp_path, record())
    residual = json.loads(output.read_text(encoding="utf-8"))["maximum_residual_nm"]
    assert result.returncode == 0 and residual <= 1e-8


def test_bad_input_preserves_existing_result(tmp_path):
    """题面：「任何报错都让旧文件原封不动，目录里也不留草稿。」"""
    result, source, output = run(tmp_path, record())
    before = output.read_bytes()
    source.write_text('{"asset_id":"RIG-14"}\n', encoding="utf-8")
    second = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and second.returncode == 1 and output.read_bytes() == before


def test_absent_parent_is_io_failure(tmp_path):
    """题面：「正式交付所在目录无法写入时，脚本返回 1，终端错误行固定为 `运行拒收: IO_FAILURE`，旧交付保持原字节。」"""
    source = tmp_path / "record.json"
    source.write_text(json.dumps(record(), separators=(",", ":")) + "\n", encoding="utf-8")
    output = tmp_path / "absent" / "result.json"
    result = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1 and result.stderr == "运行拒收: IO_FAILURE\n" and not output.exists()


def test_refusal_has_no_traceback(tmp_path):
    """题面：「退回时终端除该错误行外不应出现其它输出，也不允许 Python traceback 混入。」"""
    payload = record()
    payload["experiments"][0]["extra"] = 1
    result, _, output = run(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()
