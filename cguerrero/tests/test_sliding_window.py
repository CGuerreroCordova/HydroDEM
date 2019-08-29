from unittest import TestCase
import numpy as np
from numpy import testing
from cguerrero.hydrodem.sliding_window import (SlidingWindow, InnerWindow,
                                               SlidingIgnoreBorder,
                                               CircularWindow, NoCenterWindow,
                                               IgnoreBorderInnerSliding)
from cguerrero.hydrodem.exceptions import (NumpyArrayExpectedError,
                                           WindowSizeEvenError,
                                           WindowSizeHighError)
from constants import (sliding_window_3x3, sliding_window_element_1_1,
                       sliding_window_element_1_3, sliding_window_element_3_3,
                       sliding_window_ignore_border_3x3,
                       ignore_border_element_1_1, ignore_border_element_3_3,
                       ignore_border_element_5_4, circular_window_3x4,
                       circular_window_element_1_1, inner_window_6x6,
                       circular_window_element_3_3, inner_window_element_2_2,
                       circular_window_element_2_3, inner_window_element_3_3,
                       no_center_sliding_window_3x4,
                       no_center_sliding_window_1_1,
                       no_center_sliding_window_1_2, first_element_test,
                       second_element_test, third_element_test,
                       fourth_element_test, mixing_sliding_element_4_5,
                       mixing_sliding_element_4_9)


class TestSlidingWindow(TestCase):
    """
    Contains the test methods to tests iterators of SlidingWindow class and
    subclasses of it
    """
    classes = [SlidingWindow, SlidingIgnoreBorder, CircularWindow,
               NoCenterWindow, InnerWindow]

    def test_sliding_window_creation(self):
        """
        Check the correct creation of an instance of SlidingWindow and
        subclasses: SlidingIgnoreBorder, CircularWindow, NoCenterWindow,
        InnerWindow
        """
        window_size_test = 3
        grid = np.arange(25).reshape((5, 5))
        grid_float = grid.astype('float32')
        kwargs = {'window_size': window_size_test, 'grid': grid}
        for constructor in self.classes:
            if constructor == InnerWindow:
                kwargs['inner_size'] = 3
            sliding = constructor(**kwargs)
            self.assertEqual(sliding.window_size, window_size_test)
            if constructor == SlidingIgnoreBorder:
                self.assertFalse(sliding.grid == grid_float)
            else:
                testing.assert_array_equal(sliding.grid, grid_float)
            self.assertEqual(sliding.iter_over_ones, False)
            self.assertEqual(sliding._indices_nan, [])

    def _create_instance_context(self, class_sliding, *args, **kwargs):
        """
        Call an instance creation of SlidingWindow within a raise exception
        context

        Parameters
        ----------
        grid : array_like
            grid to try to create the instance
        window_size_test : int
            windows size to create the instance

        Returns
        -------
        The context of the exception raised

        """
        with self.assertRaises(Exception) as context:
            class_sliding(*args, **kwargs)
        return context

    def test_sliding_no_ndarray(self):
        """
        Check that an NumpyArrayExpectedError exception is raised when a non
        ndarray element is passed as parameter to create the instance of
        SlidingWindow, SlidingIgnoreBorder, CircularWindow and NoCenterWindow
        """
        window_size_test = 3
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        kwargs = {'window_size': window_size_test, 'grid': grid}
        for constructor in self.classes:
            if constructor == InnerWindow:
                kwargs['inner_size'] = 3
            context = self._create_instance_context(constructor, **kwargs)
            self.assertTrue(NumpyArrayExpectedError, type(context.exception))
            self.assertTrue('Expected numpy ndarray type' in
                            str(context.exception))

    def test_sliding_even_window_size(self):
        """
        Check that an WindowSizeEvenError exception is raised when a even
        number is given as window_size when trying to create an instance of
        SlidingWindow, SlidingIgnoreBorder, CircularWindow and NoCenterWindow
        """
        window_size_test = 4
        grid = np.arange(25).reshape((5, 5))
        kwargs = {'window_size': window_size_test, 'grid': grid}
        for constructor in self.classes:
            if constructor == InnerWindow:
                kwargs['inner_size'] = 3
            context = self._create_instance_context(constructor, **kwargs)
            self.assertEqual(WindowSizeEvenError.__name__,
                             type(context.exception).__name__)
            self.assertTrue(f'Window size: {window_size_test} cannot be an '
                            f'even number' in str(context.exception))

    def test_sliding_window_size_high(self):
        """
        Check that an WindowSizeHighError exception is raised when a
        window_size is higher than some dimension of the grid when trying to
        create an instance of SlidingWindow, SlidingIgnoreBorder,
        CircularWindow and NoCenterWindow
        """
        window_size_test = 7
        dimensions = (5, 5)
        grid = np.arange(25).reshape(dimensions)
        kwargs = {'window_size': window_size_test, 'grid': grid}
        for constructor in self.classes:
            if constructor == InnerWindow:
                kwargs['inner_size'] = 3
            context = self._create_instance_context(constructor, **kwargs)
            self.assertIs(WindowSizeHighError.__name__,
                          type(context.exception).__name__)
            self.assertTrue(
                f'Window size: {window_size_test} cannot be higher '
                f'than grid dimensions: {dimensions}' in
                str(context.exception))

    def test_sliding_window_iterator(self):
        """
        Check the iteration in sliding window.
        """
        grid = np.arange(16).reshape((4, 4))
        sliding = SlidingWindow(grid, window_size=3)
        for iter_window, input_slices in zip(sliding, sliding_window_3x3):
            self.assertEqual(sliding._indices_nan, [])
            testing.assert_array_equal(iter_window[0], input_slices[0])
            self.assertEqual(iter_window[1], input_slices[1])

    def test_sliding_window_element(self):
        """
        Check getting window element.
        """
        grid = np.arange(25).reshape((5, 5))
        sliding = SlidingWindow(grid, window_size=3)
        testing.assert_array_equal(sliding[1, 1], sliding_window_element_1_1)
        testing.assert_array_equal(sliding[3, 3], sliding_window_element_3_3)
        testing.assert_array_equal(sliding[1, 3], sliding_window_element_1_3)

    def test_sliding_window_ignore_border_iterator(self):
        """
        Check the iteration in sliding window ignoring borders
        """
        grid = np.arange(9).reshape((3, 3))
        sliding = SlidingIgnoreBorder(grid, window_size=3)
        for iter_window, input_slices in \
                zip(sliding, sliding_window_ignore_border_3x3):
            self.assertEqual(sliding._indices_nan, [])
            testing.assert_array_equal(iter_window[0], input_slices[0])
            self.assertEqual(iter_window[1], input_slices[1])

    def test_sliding_window_ignore_border_element(self):
        """
        Check getting window element from sliding window ignoring borders
        """
        grid = np.arange(25).reshape((5, 5))
        sliding = SlidingIgnoreBorder(grid, window_size=3)
        testing.assert_array_equal(sliding[1, 1], ignore_border_element_1_1)
        testing.assert_array_equal(sliding[3, 3], ignore_border_element_3_3)
        testing.assert_array_equal(sliding[5, 4], ignore_border_element_5_4)

    def test_sliding_circular_window_iterator(self):
        """
        Check the iteration in sliding circular window.
        """
        grid = np.arange(12).reshape((3, 4))
        sliding = CircularWindow(grid, window_size=3)
        for iter_window, input_slices in zip(sliding, circular_window_3x4):
            self.assertNotEqual(sliding._indices_nan, [])
            testing.assert_array_equal(iter_window[0], input_slices[0])
            self.assertEqual(iter_window[1], input_slices[1])

    def test_sliding_circular_window_element(self):
        """
        Check getting window element from a circular sliding window.
        """
        grid = np.arange(25).reshape((5, 5))
        sliding = CircularWindow(grid, window_size=3)
        testing.assert_array_equal(sliding[1, 1], circular_window_element_1_1)
        testing.assert_array_equal(sliding[3, 3], circular_window_element_3_3)
        testing.assert_array_equal(sliding[2, 3], circular_window_element_2_3)
        self.assertTrue(np.isnan(sliding[1, 1][0, 0]))
        self.assertTrue(np.isnan(sliding[3, 3][0, 2]))
        self.assertTrue(np.isnan(sliding[1, 2][2, 0]))

    def test_sliding_inner_window_iterator(self):
        """
        Check the iteration in sliding with inner window.
        """
        grid = np.arange(36).reshape((6, 6))
        sliding = InnerWindow(grid, window_size=5, inner_size=3)
        for iter_window, input_slices in zip(sliding, inner_window_6x6):
            self.assertNotEqual(sliding._indices_nan, [])
            testing.assert_array_equal(iter_window[0], input_slices[0])
            self.assertEqual(iter_window[1], input_slices[1])

    def test_sliding_inner_window_element(self):
        """
        Check getting window element from a sliding with inner window.
        """
        grid = np.arange(36).reshape((6, 6))
        sliding = InnerWindow(grid, window_size=5, inner_size=3)
        testing.assert_array_equal(sliding[2, 2], inner_window_element_2_2)
        testing.assert_array_equal(sliding[3, 3], inner_window_element_3_3)
        # Check some of nan values of inner window
        self.assertTrue(np.isnan(sliding[2, 2][1, 1]))
        self.assertTrue(np.isnan(sliding[2, 2][1, 2]))
        self.assertTrue(np.isnan(sliding[2, 2][1, 3]))
        self.assertTrue(np.isnan(sliding[3, 3][1, 3]))

    def test_sliding_no_center_window_iterator(self):
        """
        Check the iteration in sliding with no center element.
        """
        grid = np.arange(12).reshape((3, 4))
        sliding = NoCenterWindow(grid, window_size=3)
        for iter_window, input_slices in zip(sliding,
                                             no_center_sliding_window_3x4):
            self.assertNotEqual(sliding._indices_nan, [])
            testing.assert_array_equal(iter_window[0], input_slices[0])
            self.assertEqual(iter_window[1], input_slices[1])

    def test_sliding_no_center_window_element(self):
        """
        Check getting window element from a sliding with no center element.
        """
        grid = np.arange(12).reshape((3, 4))
        sliding = NoCenterWindow(grid, window_size=3)
        testing.assert_array_equal(sliding[1, 1], no_center_sliding_window_1_1)
        testing.assert_array_equal(sliding[1, 2], no_center_sliding_window_1_2)
        self.assertTrue(np.isnan(sliding[1, 1][1, 1]))
        self.assertTrue(np.isnan(sliding[1, 2][1, 1]))
        self.assertFalse(np.isnan(sliding[1, 1][0, 1]))
        self.assertFalse(np.isnan(sliding[1, 2][1, 0]))

    def test_sliding_window_composition_creation(self):
        """
        Check the correct creation of an instance of IgnoreBorderInnerSliding
        """
        window_size_test = 3
        grid = np.arange(81).reshape((9, 9))
        grid_float = grid.astype('float32')
        sliding = IgnoreBorderInnerSliding(grid, window_size=window_size_test,
                                           inner_size=3)
        self.assertEqual(sliding.window_size, window_size_test)
        self.assertFalse(sliding.grid == grid_float)
        self.assertEqual(sliding.iter_over_ones, False)
        self.assertEqual(sliding._indices_nan, [])

    def test_sliding_window_combination_iterator(self):
        """
        Check the iteration in sliding with class cooperative inheritance
        """
        window_size_test = 5
        grid = np.arange(81).reshape((9, 9))
        sliding = IgnoreBorderInnerSliding(grid, window_size=window_size_test,
                                           inner_size=3)
        iterator = iter(sliding)
        first_window = next(iterator)
        second_window = next(iterator)
        third_window = next(iterator)
        fourth_window = next(iterator)
        self.assertNotEqual(sliding._indices_nan, [])
        testing.assert_array_equal(first_window[0], first_element_test[0])
        testing.assert_array_equal(second_window[0], second_element_test[0])
        testing.assert_array_equal(third_window[0], third_element_test[0])
        testing.assert_array_equal(fourth_window[0], fourth_element_test[0])

    def test_sliding_no_center_window_element(self):
        """
        Check getting window element from a sliding with class cooperative
        inheritance
        """
        window_size_test = 5
        grid = np.arange(81).reshape((9, 9))
        sliding = IgnoreBorderInnerSliding(grid, window_size=window_size_test,
                                           inner_size=3)
        testing.assert_array_equal(sliding[4, 5], mixing_sliding_element_4_5)
        testing.assert_array_equal(sliding[4, 9], mixing_sliding_element_4_9)
        self.assertTrue(np.isnan(sliding[4, 6][2, 2]))
        self.assertTrue(np.isnan(sliding[4, 5][2, 2]))
        self.assertTrue(np.isnan(sliding[4, 9][2, 2]))
