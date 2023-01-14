import numpy as np
import matplotlib.pyplot as plt
from math import factorial

from scipy.linalg import circulant
from scipy.special import binom

from data_processing import DataProcessing


class BaseCfar:
    """
    This is the base class for the cfar algorithms
    """

    def __init__(self, n, n_guard, pfa):
        self.N = n  # reference window size
        self.N_2 = self.N // 2
        self.n_guard = n_guard
        self.n2_guard = self.n_guard // 2
        self.pfa = pfa

        if (self.N % 2) != 0:
            raise ImportError("Window size is not even number!")

        if (self.n_guard % 2) != 0:
            raise ImportError("Number of guard cells is not even number!")

        self.pad = (self.N + self.n_guard) // 2
        self.data, self.bkg, self.labels = self.load_data()

    def load_data(self):

        processed_data = DataProcessing()
        normalized_data, normalized_background, labels = processed_data.data_processing()
        # data = self.padd_profiles(normalized_data)

        return normalized_data, normalized_background, labels

    def padd_profiles(self, data_):
        pad_width = ((0, 0), (self.pad, self.pad))  # data = 2D array, pad_width is defined for 2D padding
        padded_data = np.pad(data_, pad_width=pad_width, mode="constant", constant_values=(0, 0))

        print(f"{padded_data.shape = }")

        return padded_data

    def obtain_reference_window_(self, profile, cut_idx):
        # the window includes reference window (of size N), the guard cells (of size n_guard) and the CUT
        reference_cells_leading = profile[cut_idx - self.pad: cut_idx - 1]  # the list doesn't include guard cell
        reference_cells_lagging = profile[cut_idx + 2: cut_idx + self.pad + 1]
        ref_cells = np.concatenate((reference_cells_leading, reference_cells_lagging))

        return ref_cells

    def obtain_reference_window(self, range_cells):
        '''
        calculates leading and lagging noise cells
        with unsymmetrical windows at the left and right border
        using vectorized operations

        example: 10 range cells, noise cells N = 4, guard cells N_guard = 2

                 C:cell under test, G:guard, N:leading noise cells, n:lagging cells

        cell idx 0 1 2 3 4 5 6 7 8 9
                 C G N N n n G          unsymmetrical window (left)
                 G C G N N n n
                 n G C G N N n

                 n n G C G N N          symmetrical window
                   n n G C G N N
                     n n G C G N N
                       n n G C G N N

                       N n n G C G N     unsymmetrical window (right)
                       N N n n G C G
                       G N N n n G C
                 0 1 2 3 4 5 6 7 8 9

            returns matrix with: row idx = idx of cell under test,
                                 each row contains cells for noise estimation for that cell under test
        '''
        # case 1: unsymmetrical window at left bound (terms leading and lagging loose their meaning for unsymmetry)
        num_cuts_with_unsym_win = self.n2_guard + self.N_2  # cut = cell under test
        cells_left = range_cells[:1 + self.n_guard + self.N]
        cells_left_rolled = circulant(cells_left).T
        cells_left_rolled[1:] = np.flip(cells_left_rolled[1:], axis=0)
        cells_left_rolled = cells_left_rolled[:num_cuts_with_unsym_win]

        leading_noise_cells_left = cells_left_rolled[:, 1 + self.n2_guard: 1 + self.n2_guard + self.N_2]
        lagging_noise_cells_left = cells_left_rolled[:, 1 + self.n2_guard + self.N_2: 1 + self.n2_guard + self.N]

        # case 2: symmetrical window middle part
        num_cells_with_sym_win = range_cells.size - 2 * num_cuts_with_unsym_win
        cells_rolled = circulant(range_cells).T
        cells_rolled[1:] = np.flip(cells_rolled[1:], axis=0)
        cells_rolled = cells_rolled[:num_cells_with_sym_win]

        lagging_noise_cells_middle = cells_rolled[:num_cells_with_sym_win, :self.N_2]
        leading_noise_cells_middle = cells_rolled[:num_cells_with_sym_win,
                                     1 + self.n_guard + self.N_2: 1 + self.n_guard + self.N]

        # case 3: unsymmetrical window right bound
        cells_right = range_cells[-(1 + self.n_guard + self.N):]
        cells_right_rolled = circulant(cells_right).T
        cells_right_rolled[1:] = np.flip(cells_right_rolled[1:], axis=0)
        cells_right_rolled = cells_right_rolled[1:num_cuts_with_unsym_win + 1]

        lagging_noise_cells_right = cells_right_rolled[:, :self.N_2]
        leading_noise_cells_right = cells_right_rolled[:, -self.N_2:]

        lagging_noise_cells = np.concatenate((lagging_noise_cells_left,
                                              lagging_noise_cells_middle,
                                              lagging_noise_cells_right), axis=0)

        leading_noise_cells = np.concatenate((leading_noise_cells_left,
                                              leading_noise_cells_middle,
                                              leading_noise_cells_right), axis=0)

        return leading_noise_cells, lagging_noise_cells

    def square_detector(self, ref_range_cells):
        squared_range_cells = np.square(np.abs(ref_range_cells))

        return squared_range_cells


class CACFAR(BaseCfar):
    """
    Cell Averaging CFAR Algorithm
    """

    def __init__(self, pfa=1e-3, n=20, n_guard=2):
        super().__init__(n=n, n_guard=n_guard, pfa=pfa)
        self.alpha_const = self.calculate_cfar_constant()

    def detect(self, range_cells):
        if self.N > range_cells.size:
            raise ImportError("range cell vector too short for given value of N")

        range_cells_squared = self.square_detector(range_cells)
        noise_statistics = self._estimate_noise(range_cells_squared)
        threshold_vector = noise_statistics * self.alpha_const
        thresholded_range_cells = range_cells_squared > threshold_vector

        # plt.figure()
        plt.plot(range_cells_squared, label="profile range")
        plt.plot(threshold_vector, c="orange", label="thresholds")
        #for id in thresholded_range_cells:
        #    plt.scatter(id, range_cells_squared[id], c="red")
        plt.legend(loc="best")
        plt.xlabel('range cells')
        plt.ylabel('power |y|^2')
        plt.title(f"CA CFAR, P_FA =  {self.pfa}")
        plt.grid()
        # plt.show()

        return thresholded_range_cells

    def _estimate_noise(self, range_cells):
        # estimate the mean value of the reference window (noise)
        leading_noise_ref_cells, lagging_noise_ref_cells = self.obtain_reference_window(range_cells)
        leading_noise = np.mean(leading_noise_ref_cells, axis=1)
        lagging_noise = np.mean(lagging_noise_ref_cells, axis=1)
        noise_statistics = np.mean(np.vstack((leading_noise, lagging_noise)), axis=0)

        return noise_statistics

    def calculate_cfar_constant(self):
        """
        Formula for constant calculation:
            alpha = N * [(pfa ^ (-1/N)) - 1]

        :return: cfar (alpha) constant
        """
        # calculate alpha
        alpha_const = self.N * (self.pfa ** (-1 / self.N) - 1)
        print(f"CFAR constant = {alpha_const}")

        return alpha_const


class OSCFAR(BaseCfar):
    def __init__(self, n=20, n_guard=0, k=20, pfa=1e-4):
        super().__init__(n=n, n_guard=n_guard, pfa=pfa)

        if k > n:
            raise ImportError("k is too large for given N")

        self.k = k
        self._cfar_constant = self.calc_cfar_constant(pfa)

    def detect(self, range_cells):
        if self.N > range_cells.size:
            raise ImportError("range cell vector too short for given value of N")

        range_cells_squared = self.square_detector(range_cells)
        noise_statistics = self._estimate_noise(range_cells_squared)
        threshold_vector = noise_statistics * self._cfar_constant
        thresholded_range_cells = range_cells_squared > threshold_vector

        # plt.figure()
        plt.plot(range_cells_squared)
        plt.plot(threshold_vector)
        plt.xlabel('range cells')
        plt.ylabel('power |y|^2')
        plt.title(f"OS CFAR, CFAR constant = {self._cfar_constant}")
        plt.grid()
        # plt.show()

        return thresholded_range_cells

    def _estimate_noise(self, range_cells):
        leading_noise_cells, lagging_noise_cells = self.obtain_reference_window(range_cells)
        noise_cells = np.hstack((leading_noise_cells, lagging_noise_cells))
        k = self.k - 1 # k is here required to start from 0
        noise_estimation = np.partition(noise_cells, k, axis=1)[:, k] # selects k-th ordered statistics
        return noise_estimation

    def calc_cfar_constant(self, p_false_alarm):
        """"
        approximates CFAR constant from P_FA
        using formula from POMR1, p. 614
        tests different alpha values in range [0, 10000] and checks if desired P_FA is met
        finally interpolates linear between the two solution found
        """
        current_p_fa = 1  # set inital value

        for alpha in np.arange(100000):
            last_p_fa = current_p_fa
            last_alpha = alpha
            current_p_fa = self.k * binom(self.N, self.k) * \
                           (factorial(self.k - 1) * factorial(alpha + self.N - self.k)) / factorial(
                alpha + self.N)

            if current_p_fa <= p_false_alarm:
                # interpolate linearly to determine fractional alpha
                frac_offset = (last_p_fa - p_false_alarm) / (last_p_fa - current_p_fa)
                alpha = (alpha - 1) + frac_offset  # correct integer alpha by fractional part to get a more accurate P_FA

                print(f"CFAR constant for OS CFAR found, that achieves P_FA: alpha = {alpha} , P_FA is approximated "
                    f"between  {last_p_fa} and {current_p_fa}")

                return alpha
        return

if __name__ == "__main__":
    """
     Args:
        N:       the number of cells in the reference window, both leading and lagging windows
                 the leading window = 10 cells, the lagging window = 10 cells
                 the overall cfar window = 23 cells = leading + guard + CUT + guard + lagging

        n_guard: number of guard cells in the cfar window
        pfa:     probability of false alarm 

    """

    Cfar = CACFAR(n=40, n_guard=2, pfa=1e-4)
    # Cfar = OSCFAR(n=40, n_guard=0, k=20, pfa=1e-4)
    data, bkf, labels = Cfar.load_data()

    num_detections = 0
    for idx, range_cell in enumerate(data):
        detections = Cfar.detect(range_cell)
        # compare the targets detected from cfar with true labels
        true_lbls = np.where(labels[idx] == 1)
        for lbl in true_lbls[0]:
            if detections[lbl]:
                num_detections += 1

    # num_detections = np.sum(detections)
    all_targets = np.sum(labels)
    # print(detections)
    print(f"P of detections = {num_detections}/ {all_targets} = {num_detections / all_targets}")
