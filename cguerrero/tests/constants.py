import numpy as np

sliding_window_3x3 = [(np.array([[0., 1., 2.],
                                 [4., 5., 6.],
                                 [8., 9., 10.]], dtype='float32'), (1, 1)),
                      (np.array([[1., 2., 3.],
                                 [5., 6., 7.],
                                 [9., 10., 11.]], dtype='float32'), (1, 2)),
                      (np.array([[4., 5., 6.],
                                 [8., 9., 10.],
                                 [12., 13., 14.]], dtype='float32'), (2, 1)),
                      (np.array([[5., 6., 7.],
                                 [9., 10., 11.],
                                 [13., 14., 15.]], dtype='float32'), (2, 2))]

sliding_window_element_1_1 = np.array([[0., 1., 2.],
                                       [5., 6., 7.],
                                       [10., 11., 12.]], dtype='float32')

sliding_window_element_3_3 = np.array([[12., 13., 14.],
                                       [17., 18., 19.],
                                       [22., 23., 24.]], dtype='float32')

sliding_window_element_1_3 = np.array([[2., 3., 4.],
                                       [7., 8., 9.],
                                       [12., 13., 14.]], dtype='float32')
