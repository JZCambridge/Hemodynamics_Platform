import io
from contextlib import redirect_stdout
from pathlib import Path
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import psutil


GUI_ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS_ROOT = GUI_ROOT.parent / "Functions_JZ"
for import_path in (GUI_ROOT, FUNCTIONS_ROOT):
    import_string = str(import_path)
    if import_string not in sys.path:
        sys.path.insert(0, import_string)

from Modules_YL.CenterlineGeneration.CenterlineGenerationMain import CenterlineEx


class SafeCenterlineRunnerTests(unittest.TestCase):
    def test_stage_commands_match_the_legacy_vendor_sequence(self):
        commands = CenterlineEx._safe_stage_commands(
            r"D:\vendor\Part1\part1.exe",
            r"D:\cases\case with spaces",
            manual_flag=True,
        )
        self.assertEqual(
            [stage_name for stage_name, _ in commands],
            ["part1", "part2alt_manual", "part2altb", "part3", "part4"],
        )
        self.assertEqual(commands[-1][1][-1], "45")
        self.assertEqual(len(commands[0][1]), 2)
        self.assertEqual(
            commands[1][1][0],
            os.path.abspath(r"D:\vendor\Tenoke_ccta_Part2Alt\Tenoke_ccta_Part2Alt.exe"),
        )
        self.assertEqual(commands[0][1][1], os.path.abspath(r"D:\cases\case with spaces"))

    def test_stage_captures_success_and_nonzero_output(self):
        success = CenterlineEx._run_stage_safely(
            "success",
            [sys.executable, "-c", "print('stage ok')"],
            timeout_seconds=5,
            output_limit=100,
        )
        self.assertEqual(success["returncode"], 0)
        self.assertEqual(success["error"], "")
        self.assertIn("stage ok", success["output"])

        failure = CenterlineEx._run_stage_safely(
            "failure",
            [sys.executable, "-c", "print('stage failed'); raise SystemExit(7)"],
            timeout_seconds=5,
            output_limit=100,
        )
        self.assertEqual(failure["returncode"], 7)
        self.assertIn("return code 7", failure["error"])
        self.assertIn("stage failed", failure["output"])

    def test_stage_output_keeps_only_the_requested_tail(self):
        result = CenterlineEx._run_stage_safely(
            "bounded-output",
            [sys.executable, "-c", "print('x' * 500 + 'TAIL')"],
            timeout_seconds=5,
            output_limit=24,
        )
        self.assertLessEqual(len(result["output"].encode()), 24)
        self.assertTrue(result["output"].endswith("TAIL"))

    def test_stage_output_is_visible_before_process_finishes(self):
        terminal_output = io.StringIO()
        result_holder = {}

        def run_stage():
            result_holder["result"] = CenterlineEx._run_stage_safely(
                "live-output",
                [
                    sys.executable,
                    "-c",
                    (
                        "import time; "
                        "print('FIRST', flush=True); "
                        "time.sleep(1.2); "
                        "print('SECOND', flush=True)"
                    ),
                ],
                timeout_seconds=5,
                output_limit=100,
            )

        with redirect_stdout(terminal_output):
            stage_thread = threading.Thread(target=run_stage)
            stage_thread.start()
            deadline = time.time() + 2
            while "FIRST" not in terminal_output.getvalue() and time.time() < deadline:
                time.sleep(0.02)
            first_was_live = "FIRST" in terminal_output.getvalue()
            process_was_running = stage_thread.is_alive()
            stage_thread.join(5)

        self.assertTrue(first_was_live)
        self.assertTrue(process_was_running)
        self.assertFalse(stage_thread.is_alive())
        self.assertIn("SECOND", terminal_output.getvalue())
        self.assertIn("FIRST", result_holder["result"]["output"])
        self.assertIn("SECOND", result_holder["result"]["output"])

    def test_stage_timeout_is_returned_instead_of_raised(self):
        result = CenterlineEx._run_stage_safely(
            "timeout",
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.1,
            output_limit=100,
        )
        self.assertTrue(result["timed_out"])
        self.assertIn("Timed out", result["error"])

    @unittest.skipUnless(sys.platform == "win32", "Windows process-tree behavior")
    def test_stage_timeout_terminates_descendant_process(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            pid_path = Path(temporary_directory) / "child.pid"
            child_code = "import time; time.sleep(30)"
            parent_code = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', {0!r}]); "
                "pathlib.Path({1!r}).write_text(str(child.pid)); "
                "time.sleep(30)"
            ).format(child_code, str(pid_path))
            result = CenterlineEx._run_stage_safely(
                "process-tree-timeout",
                [sys.executable, "-c", parent_code],
                timeout_seconds=0.5,
                output_limit=100,
            )
            self.assertTrue(result["timed_out"])
            child_pid = int(pid_path.read_text())
            for _ in range(20):
                if not psutil.pid_exists(child_pid):
                    break
                time.sleep(0.1)
            self.assertFalse(psutil.pid_exists(child_pid))

    def test_missing_executable_is_returned_instead_of_raised(self):
        result = CenterlineEx._run_stage_safely(
            "missing",
            [r"Z:\definitely_missing\not_an_executable.exe", "case"],
            timeout_seconds=5,
            output_limit=100,
        )
        self.assertIsNone(result["returncode"])
        self.assertIn("Could not start process", result["error"])

    def test_case_failure_stops_only_remaining_stages(self):
        runner = object.__new__(CenterlineEx)
        with tempfile.TemporaryDirectory() as temporary_directory:
            commands = [
                ("one", [sys.executable, "-c", "print('one')"]),
                ("two", [sys.executable, "-c", "raise SystemExit(2)"]),
                ("three", [sys.executable, "-c", "print('must not run')"]),
            ]
            with mock.patch.object(
                CenterlineEx, "_safe_stage_commands", return_value=commands
            ):
                result = runner.run_case_safely(
                    destination=temporary_directory,
                    exe_path="placeholder.exe",
                    timeout_seconds=5,
                )

        self.assertFalse(result["success"])
        self.assertTrue(result["exe_failed"])
        self.assertEqual([stage["name"] for stage in result["stages"]], ["one", "two"])
        self.assertIn("return code 2", result["error_message"])

    def test_case_success_has_empty_error(self):
        runner = object.__new__(CenterlineEx)
        command = [("one", [sys.executable, "-c", "raise SystemExit(0)"])]
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(
                CenterlineEx, "_safe_stage_commands", return_value=command
            ):
                result = runner.run_case_safely(
                    destination=temporary_directory,
                    exe_path="placeholder.exe",
                    timeout_seconds=5,
                )

        self.assertTrue(result["success"])
        self.assertFalse(result["exe_failed"])
        self.assertEqual(result["error_message"], "")


if __name__ == "__main__":
    unittest.main()
