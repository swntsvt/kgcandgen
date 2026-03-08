import logging
import os
import tempfile
import unittest
from pathlib import Path

from src.logging_utils import setup_logging


class SetupLoggingTests(unittest.TestCase):
    def test_setup_logging_writes_to_console_and_file(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                logger = setup_logging()

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


if __name__ == "__main__":
    unittest.main()
