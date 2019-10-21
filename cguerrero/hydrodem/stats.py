import math
import gdal
import numpy as np
from config_loader import Config

class Stats:

    def __init__(self, file_key, range_files, sufix):
        self.sufix = '_' + sufix
        self.range_files = range_files
        self.filename = Config.simulation(file_key)
        self.ndwi = gdal.Open(Config.simulation("NDWI_IMAGE")).ReadAsArray()
        self.total_positives, self.total_negatives = self._totals()
        self.total_values = self.total_positives + self.total_negatives
        self.values_file = dict()
        self.stats_functions = [self.accuracy, self.sensitivity, self.BACC,
                                self.f1_score, self.MCC, self.precision,
                                self.specificity, self.fall_out]

    def _totals(self):
        total_positives = np.count_nonzero(self.ndwi)
        ndwi_complement = 1 - self.ndwi
        total_negatives = np.count_nonzero(ndwi_complement)
        return total_positives, total_negatives

    def accuracy(self):
        return (self.values_file['TN'] + self.values_file['TP']) /\
               self.total_values

    def sensitivity(self):
        return self.values_file['TP'] / self.values_file['P']

    def precision(self):
        return self.values_file['TP'] / (self.values_file['TP'] +
                                         self.values_file['FP'])

    def specificity(self):
        return self.values_file['TN'] / self.values_file['N']

    def fall_out(self):
        return self.values_file['FP'] / (self.values_file['FP'] +
                                         self.values_file['TN'])

    def BACC(self):
        return (self.sensitivity() + self.specificity()) / 2.

    def f1_score(self):
        return (2. * self.values_file['TP']) / (2. * self.values_file['TP'] +
                                                self.values_file['FP'] +
                                                self.values_file['FP'])

    def MCC(self):
        numerator_mcc = (self.values_file['TP'] * self.values_file['TN']) - \
                        (self.values_file['FP'] * self.values_file['FN'])
        denominator_mcc = (self.values_file['TP'] + self.values_file['FP']) * \
                          (self.values_file['TP'] + self.values_file['FN']) * \
                          (self.values_file['TN'] + self.values_file['FP']) * \
                          (self.values_file['TN'] + self.values_file['FN'])

        return numerator_mcc / math.sqrt(denominator_mcc)

    def _set_values(self, file_number):

        ndwi_complement = 1 - self.ndwi
        file = gdal.Open(self.filename.format(file_number)).ReadAsArray()
        mask = file > 0.0001
        true_positives = mask * self.ndwi
        self.values_file['TP'] = np.count_nonzero(true_positives)

        false_negatives = ((self.ndwi - mask) > 0) * 1
        self.values_file['FN'] = np.count_nonzero(false_negatives)

        self.values_file['P'] = self.values_file['TP'] + \
                                self.values_file['FN']

        false_positives = ((mask - self.ndwi) > 0) * 1
        self.values_file['FP'] = np.count_nonzero(false_positives)

        mask_complement = 1 - mask
        true_negatives = mask_complement * ndwi_complement
        self.values_file['TN'] = np.count_nonzero(true_negatives)

        self.values_file['N'] = self.values_file['FP'] + \
                                self.values_file['TN']
    def get_stats(self):
        list_stats = []
        for i in range(1, self.range_files):
            stats_file = {'day': i}
            self._set_values(i)
            for f in self.stats_functions:
                stats_file[f.__name__ + self.sufix] = f()
            list_stats.append(stats_file)
        return list_stats










