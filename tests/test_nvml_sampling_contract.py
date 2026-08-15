"""Compatibility tests for the legacy GreenAI NVML collection scripts."""

from __future__ import annotations

import ast
import csv
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    ROOT / "GPU_Performance/Cloud/AutoDL_Version2/4090/Functions/CNN_info.py",
    ROOT / "GPU_Performance/Cloud/AutoDL_Version2/3080/Functions/CNN_info.py",
    ROOT / "GPU_Performance/Local/4070/Functions/CNN_info.py",
)
EXPECTED_TRACE = (
    "timestamp,power_in_watts,sm_clock\n"
    "123.0, 250.0, 1500\n"
)


def _extract_function(path: Path, name: str, namespace: dict | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(namespace or {})
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class _StopEvent:
    def __init__(self) -> None:
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped


class _FakeTime:
    def __init__(self, stop_event: _StopEvent) -> None:
        self.stop_event = stop_event

    @staticmethod
    def time() -> float:
        return 123.0

    def sleep(self, _interval: float) -> None:
        self.stop_event.stopped = True


class _NvmlError(Exception):
    pass


class _SuccessfulNvml:
    NVMLError = _NvmlError
    NVML_CLOCK_SM = object()

    @staticmethod
    def nvmlDeviceGetPowerUsage(_handle) -> float:
        return 250_000.0

    @staticmethod
    def nvmlDeviceGetClockInfo(_handle, _clock) -> int:
        return 1500


class _FailingNvml(_SuccessfulNvml):
    @staticmethod
    def nvmlDeviceGetPowerUsage(_handle) -> float:
        raise _NvmlError("sampling failed")


class NvmlSamplingContractTest(unittest.TestCase):
    def test_successful_sampling_preserves_legacy_csv_bytes(self) -> None:
        for path in TARGETS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as temp_dir:
                stop_event = _StopEvent()
                errors = []
                sampler = _extract_function(
                    path,
                    "nvml_sampling_thread",
                    {
                        "pynvml": _SuccessfulNvml,
                        "time": _FakeTime(stop_event),
                    },
                )

                sampler(object(), Path(temp_dir), stop_event, 0.1, errors)

                trace = Path(temp_dir, "energy_consumption_file.csv").read_text()
                self.assertEqual(EXPECTED_TRACE, trace)
                self.assertEqual([], errors)

    def test_sampling_failure_is_returned_to_the_training_thread(self) -> None:
        for path in TARGETS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as temp_dir:
                stop_event = _StopEvent()
                errors = []
                sampler = _extract_function(
                    path,
                    "nvml_sampling_thread",
                    {"pynvml": _FailingNvml, "time": _FakeTime(stop_event)},
                )

                sampler(object(), Path(temp_dir), stop_event, 0.1, errors)

                self.assertEqual(1, len(errors))
                self.assertIsInstance(errors[0], _NvmlError)

    def test_current_trace_segment_requires_samples_and_time_coverage(self) -> None:
        for path in TARGETS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as temp_dir:
                validate = _extract_function(
                    path,
                    "_validate_nvml_trace_segment",
                    {"csv": csv, "math": math},
                )
                trace_path = Path(temp_dir, "energy_consumption_file.csv")
                prefix = (
                    "timestamp,power_in_watts,sm_clock\n"
                    "1.0, 100.0, 1000\n"
                )
                trace_path.write_text(
                    prefix
                    + "timestamp,power_in_watts,sm_clock\n"
                    + "99.95, 200.0, 1400\n"
                    + "101.0, 205.0, 1400\n"
                    + "101.98, 210.0, 1400\n",
                    encoding="utf-8",
                )
                validate(trace_path, len(prefix.encode()), 100.0, 102.0, 0.1)

                trace_path.write_text(
                    "timestamp,power_in_watts,sm_clock\n"
                    "100.0, 200.0, 1400\n"
                    "100.8, 205.0, 1400\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "does not cover"):
                    validate(trace_path, 0, 100.0, 102.0, 0.1)

    def test_sampler_cleanup_is_in_finally(self) -> None:
        expected_calls = {
            "stop_event.set",
            "sampler_thread.join",
            "pynvml.nvmlShutdown",
        }
        for path in TARGETS:
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                train = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "train_func"
                )
                finally_calls = {
                    ast.unparse(call.func)
                    for try_node in ast.walk(train)
                    if isinstance(try_node, ast.Try)
                    for statement in try_node.finalbody
                    for call in ast.walk(statement)
                    if isinstance(call, ast.Call)
                }
                self.assertTrue(expected_calls.issubset(finally_calls))

    def test_sampling_starts_after_legacy_training_setup(self) -> None:
        for path in TARGETS:
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                body = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_train_func_body"
                )
                calls = {
                    ast.unparse(call.func): call.lineno
                    for call in ast.walk(body)
                    if isinstance(call, ast.Call)
                }
                epoch_loop = next(
                    node
                    for node in body.body
                    if isinstance(node, ast.For)
                    and ast.unparse(node.target) == "epoch"
                )
                self.assertLess(calls["net.to"], calls["start_sampler"])
                self.assertLess(calls["start_sampler"], epoch_loop.lineno)


if __name__ == "__main__":
    unittest.main()
