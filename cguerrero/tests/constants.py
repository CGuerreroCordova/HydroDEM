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

sliding_window_ignore_border_3x3 = [(np.array([[np.nan, np.nan, np.nan],
                                               [np.nan, 0., 1.],
                                               [np.nan, 3., 4.]],
                                              dtype='float32'), (1, 1)),
                                    (np.array([[np.nan, np.nan, np.nan],
                                               [0., 1., 2.],
                                               [3., 4., 5.]],
                                              dtype='float32'), (1, 2)),
                                    (np.array([[np.nan, np.nan, np.nan],
                                               [1., 2., np.nan],
                                               [4., 5., np.nan]],
                                              dtype='float32'), (1, 3)),
                                    (np.array([[np.nan, 0., 1.],
                                               [np.nan, 3., 4.],
                                               [np.nan, 6., 7.]],
                                              dtype='float32'), (2, 1)),
                                    (np.array([[0., 1., 2.],
                                               [3., 4., 5.],
                                               [6., 7., 8.]],
                                              dtype='float32'), (2, 2)),
                                    (np.array([[1., 2., np.nan],
                                               [4., 5., np.nan],
                                               [7., 8., np.nan]],
                                              dtype='float32'), (2, 3)),
                                    (np.array([[np.nan, 3., 4.],
                                               [np.nan, 6., 7.],
                                               [np.nan, np.nan, np.nan]],
                                              dtype='float32'), (3, 1)),
                                    (np.array([[3., 4., 5.],
                                               [6., 7., 8.],
                                               [np.nan, np.nan, np.nan]],
                                              dtype='float32'), (3, 2)),
                                    (np.array([[4., 5., np.nan],
                                               [7., 8., np.nan],
                                               [np.nan, np.nan, np.nan]],
                                              dtype='float32'), (3, 3))]

ignore_border_element_1_1 = np.array([[np.nan, np.nan, np.nan],
                                      [np.nan, 0., 1.],
                                      [np.nan, 5., 6.]], dtype='float32')

ignore_border_element_3_3 = np.array([[6., 7., 8.],
                                      [11., 12., 13.],
                                      [16., 17., 18.]], dtype='float32')
ignore_border_element_5_4 = np.array([[17., 18., 19.],
                                      [22., 23., 24.],
                                      [np.nan, np.nan, np.nan]],
                                     dtype='float32')
circular_window_3x4 = [(np.array([[np.nan, 1., np.nan],
                                  [4., 5., 6.],
                                  [np.nan, 9., np.nan]], dtype='float32'),
                        (1, 1)),
                       (np.array([[np.nan, 2., np.nan],
                                  [5., 6., 7.],
                                  [np.nan, 10., np.nan]], dtype='float32'),
                        (1, 2))]
circular_window_element_1_1 = np.array([[np.nan, 1., np.nan],
                                        [5., 6., 7.],
                                        [np.nan, 11., np.nan]],
                                       dtype='float32')
circular_window_element_3_3 = np.array([[np.nan, 13., np.nan],
                                        [17., 18., 19.],
                                        [np.nan, 23., np.nan]],
                                       dtype='float32')
circular_window_element_2_3 = np.array([[np.nan, 8., np.nan],
                                        [12., 13., 14.],
                                        [np.nan, 18., np.nan]],
                                       dtype='float32')

inner_window_6x6 = [(np.array([[0., 1., 2., 3., 4.],
                               [6., np.nan, np.nan, np.nan, 10.],
                               [12., np.nan, 14., np.nan, 16.],
                               [18., np.nan, np.nan, np.nan, 22.],
                               [24., 25., 26., 27., 28.]], dtype='float32'),
                     (2, 2)),
                    (np.array([[1., 2., 3., 4., 5.],
                               [7., np.nan, np.nan, np.nan, 11.],
                               [13., np.nan, 15., np.nan, 17.],
                               [19., np.nan, np.nan, np.nan, 23.],
                               [25., 26., 27., 28., 29.]], dtype='float32'),
                     (2, 3)),
                    (np.array([[6., 7., 8., 9., 10.],
                               [12., np.nan, np.nan, np.nan, 16.],
                               [18., np.nan, 20., np.nan, 22.],
                               [24., np.nan, np.nan, np.nan, 28.],
                               [30., 31., 32., 33., 34.]], dtype='float32'),
                     (3, 2)),
                    (np.array([[7., 8., 9., 10., 11.],
                               [13., np.nan, np.nan, np.nan, 17.],
                               [19., np.nan, 21., np.nan, 23.],
                               [25., np.nan, np.nan, np.nan, 29.],
                               [31., 32., 33., 34., 35.]], dtype='float32'),
                     (3, 3))]

inner_window_element_2_2 = np.array([[0., 1., 2., 3., 4.],
                                     [6., np.nan, np.nan, np.nan, 10.],
                                     [12., np.nan, 14., np.nan, 16.],
                                     [18., np.nan, np.nan, np.nan, 22.],
                                     [24., 25., 26., 27., 28.]],
                                    dtype='float32')
inner_window_element_3_3 = np.array([[7., 8., 9., 10., 11.],
                                     [13., np.nan, np.nan, np.nan, 17.],
                                     [19., np.nan, 21., np.nan, 23.],
                                     [25., np.nan, np.nan, np.nan, 29.],
                                     [31., 32., 33., 34., 35.]],
                                    dtype='float32')
no_center_sliding_window_3x4 = [(np.array([[0., 1., 2.],
                                           [4., np.nan, 6.],
                                           [8., 9., 10.]], dtype='float32'),
                                 (1, 1)),
                                (np.array([[1., 2., 3.],
                                           [5., np.nan, 7.],
                                           [9., 10., 11.]], dtype='float32'),
                                 (1, 2))]

no_center_sliding_window_1_1 = np.array([[0., 1., 2.],
                                         [4., np.nan, 6.],
                                         [8., 9., 10.]], dtype='float32')

no_center_sliding_window_1_2 = np.array([[1., 2., 3.],
                                         [5., np.nan, 7.],
                                         [9., 10., 11.]], dtype='float32')
first_element_test = (np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, 2.],
                                [np.nan, np.nan, np.nan, np.nan, 11.],
                                [np.nan, np.nan, 18., 19., 20.]],
                               dtype='float32'), (2, 2))
second_element_test = (np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan, 3.],
                                 [np.nan, np.nan, np.nan, np.nan, 12.],
                                 [np.nan, 18., 19., 20., 21.]],
                                dtype='float32'), (2, 3))
third_element_test = (np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan],
                                [0., np.nan, np.nan, np.nan, 4.],
                                [9., np.nan, np.nan, np.nan, 13.],
                                [18., 19., 20., 21., 22.]], dtype='float32'),
                      (2, 4))
fourth_element_test = (np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan, np.nan],
                                 [1., np.nan, np.nan, np.nan, 5.],
                                 [10., np.nan, np.nan, np.nan, 14.],
                                 [19., 20., 21., 22., 23.]], dtype='float32'),
                       (2, 5))
mixing_sliding_element_4_9 = np.array([[5., 6., 7., 8., np.nan],
                                       [14., np.nan, np.nan, np.nan, np.nan],
                                       [23., np.nan, np.nan, np.nan, np.nan],
                                       [32., np.nan, np.nan, np.nan, np.nan],
                                       [41., 42., 43., 44., np.nan]],
                                      dtype='float32')

mixing_sliding_element_4_5 = np.array([[1., 2., 3., 4., 5.],
                                       [10., np.nan, np.nan, np.nan, 14.],
                                       [19., np.nan, np.nan, np.nan, 23.],
                                       [28., np.nan, np.nan, np.nan, 32.],
                                       [37., 38., 39., 40., 41.]],
                                      dtype='float32')
