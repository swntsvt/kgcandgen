import logging
import os
import tempfile
import unittest
from pathlib import Path

from src.logging_utils import setup_logging


class SetupLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        project_logger = logging.getLogger("src")
        for handler in project_logger.handlers[:]:
            project_logger.removeHandler(handler)
            handler.close()

    def test_setup_logging_writes_to_console_and_file(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                logger = setup_logging()
                self.assertEqual(logger.name, "src")

                handlers = logger.handlers
                has_file_handler = any(isinstance(h, logging.FileHandler) for h in handlers)
                has_console_handler = any(type(h) is logging.StreamHandler for h in handlers)

                self.assertTrue(has_file_handler)
                self.assertTrue(has_console_handler)

                message = "unit-test-log-message"
                logger.info(message)

                for handler in handlers:
                    handler.flush()

                log_files = list(Path("logs").glob("experiment_*.log"))
                self.assertEqual(len(log_files), 1)
                self.assertIn(message, log_files[0].read_text(encoding="utf-8"))
            finally:
                os.chdir(original_cwd)

    def test_setup_logging_closes_previous_file_handlers(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                first_logger = setup_logging()
                first_file_handlers = [
                    handler
                    for handler in first_logger.handlers
                    if isinstance(handler, logging.FileHandler)
                ]
                self.assertTrue(first_file_handlers)
                first_handler = first_file_handlers[0]

                second_logger = setup_logging()
                self.assertIs(first_logger, second_logger)
                self.assertTrue(first_handler.stream is None or first_handler.stream.closed)
            finally:
                os.chdir(original_cwd)

    def test_setup_logging_does_not_mutate_root_handlers(self) -> None:
        root_logger = logging.getLogger()
        sentinel_handler = logging.StreamHandler()
        root_logger.addHandler(sentinel_handler)
        try:
            setup_logging()
            self.assertIn(sentinel_handler, root_logger.handlers)
        finally:
            root_logger.removeHandler(sentinel_handler)
            sentinel_handler.close()


if __name__ == "__main__":
    unittest.main()
