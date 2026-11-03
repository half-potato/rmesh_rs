import unittest

import numpy as np

from compare_linear_renders import compare


class LinearRenderGateTests(unittest.TestCase):
    def test_single_bad_channel_fails_despite_small_average(self):
        reference = np.zeros((3, 20, 30))
        candidate = np.zeros((20, 30, 3))
        candidate[8, 9, 2] = 0.0101
        report = compare(reference, candidate)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_pixels"], 1)
        self.assertEqual(report["worst_y_x_channel"], [8, 9, 2])
        self.assertLess(report["mean_abs_error"], 1e-4)

    def test_hdr_is_not_clipped_and_layout_is_explicit(self):
        candidate = np.full((7, 9, 3), 8.0)
        reference = np.moveaxis(candidate.copy(), -1, 0)
        self.assertTrue(compare(reference, candidate)["passed"])
        candidate[0, 0, 0] += 0.02
        self.assertFalse(compare(reference, candidate)["passed"])

    def test_absolute_threshold_is_inclusive(self):
        self.assertTrue(compare(np.zeros((3, 1, 1)), np.full((1, 1, 3), .01))["passed"])

    def test_mse_averages_all_pixels_and_channels_without_clipping(self):
        reference = np.zeros((3, 1, 2), dtype=np.float32)
        candidate = np.array([[[1., 2., 3.], [0., 0., 0.]]])
        report = compare(reference, candidate)
        self.assertAlmostEqual(report["mse"], 14. / 6.)
        np.testing.assert_allclose(report["mse_rgb"], [0.5, 2., 4.5])
        self.assertAlmostEqual(report["rmse"] ** 2, report["mse"])

    def test_invalid_inputs_cannot_pass(self):
        reference = np.zeros((3, 2, 2))
        for candidate in [np.zeros((2, 2, 3), dtype=np.uint8),
                          np.full((2, 2, 3), np.nan),
                          np.full((2, 2, 3), np.inf),
                          np.zeros((2, 2, 4)), np.zeros((3, 2, 3))]:
            with self.assertRaises(ValueError):
                compare(reference, candidate)
        for epsilon in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                compare(reference, np.zeros((2, 2, 3)), epsilon)


if __name__ == "__main__":
    unittest.main()
