from unittest import TestCase
import numpy as np
from numpy import testing
from cguerrero.hydrodem.sliding_window import (SlidingWindow, InnerWindow,
                                               SlidingIgnoreBorder,
                                               CircularWindow, NoCenterWindow)
from cguerrero.hydrodem.exceptions import (NumpyArrayExpectedError,
                                           WindowSizeEvenError,
                                           WindowSizeHighError)
from constants import (sliding_window_3x3, sliding_window_element_1_1,
                       sliding_window_element_1_3, sliding_window_element_3_3)


class TestSlidingWindow(TestCase):
    classes = [SlidingWindow, SlidingIgnoreBorder, CircularWindow,
               NoCenterWindow, InnerWindow]

    def test_sliding_window_creation(self):
        """
        Check the correct creation of an instance
        """
        window_size_test = 3
        grid = np.arange(25).reshape((5, 5))
        grid_float = grid.astype('float32')

        sliding = SlidingWindow(grid, window_size=window_size_test)

        self.assertEqual(sliding.window_size, window_size_test)
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
        grid_2 = np.arange(16).reshape((4, 4))
        sliding_2 = SlidingWindow(grid_2, window_size=3)
        for iter_window, input_slices in zip(sliding_2, sliding_window_3x3):
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
