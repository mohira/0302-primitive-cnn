import unittest

import numpy as np
from numpy.testing import assert_array_equal


class Conv2D:
    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel

    def forward(self, img: np.ndarray) -> np.ndarray:
        kernel_width, kernel_height, kernel_ch = self.kernel.shape

        ret = []

        out_width = img.shape[0] - kernel_width + 1
        out_height = img.shape[1] - kernel_height + 1

        for idx in range(out_width):
            for jdx in range(out_height):
                hoge = img[idx:kernel_width + idx, jdx:kernel_height + jdx]

                fuga = hoge * self.kernel
                ret.append(fuga.sum(axis=0).sum(axis=0))

        ret = np.array(ret).reshape((out_width, out_height, kernel_ch))

        return ret


class TestConv2D(unittest.TestCase):
    def test_入力3x3x1とカーネル2x2x1(self):
        img = np.array([[[1], [2], [3]],
                        [[4], [5], [6]],
                        [[7], [8], [9]]])

        kernel = np.array([[[0], [1]],
                           [[1], [2]]])

        expected = np.array([[[16], [20]],
                             [[28], [32]]])

        conv = Conv2D(kernel)

        assert_array_equal(expected, conv.forward(img))

    def test_入力3x3x1とカーネル3x3x1(self):
        img = np.array([[[1], [2], [3]],
                        [[4], [5], [6]],
                        [[7], [8], [9]]])

        kernel = np.array([[[1], [0], [0]],
                           [[0], [1], [0]],
                           [[0], [0], [1]]])

        expected = np.array([[[15]]])

        conv = Conv2D(kernel)

        assert_array_equal(expected, conv.forward(img))

    def test_hoge(self):
        img = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                        [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                        [[7, 7, 7], [8, 8, 8], [9, 9, 9]]])

        kernel = np.array([[[0, 0, 1], [1, 1, 0]],
                           [[1, 1, 0], [2, 0, 1]]])

        expected = np.array([[[16, 6, 6], [20, 8, 8]],
                             [[28, 12, 12], [32, 14, 14]]])

        conv = Conv2D(kernel)

        assert_array_equal(expected, conv.forward(img))


if __name__ == "__main__":
    unittest.main()
