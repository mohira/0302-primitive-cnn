import unittest

import numpy as np
from numpy.testing import assert_array_equal


class Conv1D:
    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel

    def forward(self, img: np.ndarray) -> np.ndarray:
        kernel_width, kernel_height = self.kernel.shape

        ret = []

        final_width = img.shape[0] - kernel_width + 1
        final_height = img.shape[1] - kernel_height + 1

        for idx in range(final_width):
            for jdx in range(final_height):
                hoge = img[idx:kernel_width + idx, jdx:kernel_height + jdx]
                fuga = hoge * self.kernel
                ret.append(fuga.sum())

        ret = np.array(ret).reshape((final_width, final_height))

        return ret


class TestConv1D(unittest.TestCase):
    def test_入力3x3とカーネル2x2(self):
        img = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

        kernel = np.array([[0, 1],
                           [1, 2]])

        expected = np.array([[16, 20],
                             [28, 32]])

        conv = Conv1D(kernel)

        assert_array_equal(expected, conv.forward(img))

    def test_入力3x3とカーネル3x3(self):
        img = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

        kernel = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        expected = np.array([[15]])

        conv = Conv1D(kernel)

        assert_array_equal(expected, conv.forward(img))


if __name__ == "__main__":
    unittest.main()
