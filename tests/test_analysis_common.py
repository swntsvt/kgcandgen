import unittest

import numpy as np

from src.analysis.common import coerce_count, coerce_float


class AnalysisCommonCoercionTests(unittest.TestCase):
    def test_coerce_count_accepts_integral_values(self) -> None:
        self.assertEqual(coerce_count(3), 3)
        self.assertEqual(coerce_count(3.0), 3)
        self.assertEqual(coerce_count("3"), 3)
        self.assertEqual(coerce_count(np.int64(3)), 3)
        self.assertEqual(coerce_count(np.float64(3.0)), 3)

    def test_coerce_count_rejects_non_integral_or_bool_values(self) -> None:
        with self.assertRaises(TypeError):
            coerce_count(True)
        with self.assertRaises(TypeError):
            coerce_count(False)
        with self.assertRaises(TypeError):
            coerce_count(1.2)
        with self.assertRaises(TypeError):
            coerce_count("1.2")
        with self.assertRaises(TypeError):
            coerce_count(np.float64(1.2))
        with self.assertRaises(TypeError):
            coerce_count("")
        with self.assertRaises(TypeError):
            coerce_count("3.0")
        with self.assertRaises(TypeError):
            coerce_count("1e3")

    def test_coerce_float_accepts_numeric_values(self) -> None:
        self.assertEqual(coerce_float(3), 3.0)
        self.assertEqual(coerce_float(3.5), 3.5)
        self.assertEqual(coerce_float("3.5"), 3.5)
        self.assertEqual(coerce_float(" 2 "), 2.0)
        self.assertEqual(coerce_float(np.int64(7)), 7.0)
        self.assertEqual(coerce_float(np.float64(7.25)), 7.25)

    def test_coerce_float_rejects_bool_or_non_numeric_values(self) -> None:
        with self.assertRaises(TypeError):
            coerce_float(True)
        with self.assertRaises(TypeError):
            coerce_float(False)
        with self.assertRaises(TypeError):
            coerce_float("")
        with self.assertRaises(TypeError):
            coerce_float("not-a-number")
        with self.assertRaises(TypeError):
            coerce_float(complex(1, 2))
        with self.assertRaises(TypeError):
            coerce_float(float("nan"))
        with self.assertRaises(TypeError):
            coerce_float(float("inf"))
        with self.assertRaises(TypeError):
            coerce_float("nan")
        with self.assertRaises(TypeError):
            coerce_float("inf")


if __name__ == "__main__":
    unittest.main()
