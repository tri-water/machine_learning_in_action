import numpy as np
import random
import matplotlib.pyplot as plt
import os


def load_dataset(file_name):
    cwd = os.getcwd()
    with open(cwd + r'/data/' + file_name) as file:
        line_list = file.readlines()
        line_list = [line.strip().split('\t') for line in line_list]
        line_list = np.mat(line_list, dtype=float)
        data_mat = line_list[:, :-1]
        label_mat = line_list[:, -1]
    return data_mat, label_mat


class Optimiser:
    def __init__(self, data_mat, label_mat, C, tol, max_iter, k_tup=('lin', 0)):
        self.X = data_mat
        self.label_mat = label_mat
        self.C = C
        self.tol = tol
        self.m = label_mat.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # The first column is a flag bit stating whether the error_cache
        # is valid
        self.error_cache = np.mat(np.zeros((self.m, 2)))
        self.max_iter = max_iter
        self.k_tup = k_tup
        self.w = np.zeros((self.X.shape[1], 1))


    def _calc_error_cache(self, k):
        f_Xk = float(np.multiply(self.alphas, self.label_mat).T \
                * (self.X * self.X[k, :].T)) + self.b
        error_k = f_Xk - float(self.label_mat[k])
        return error_k


    def _select_j_rand(self, i):
        """Select a random value, which is not equal to input i

        Params:
        =======
        i: the index of our first alpha
        """
        
        j = i
        while (j == i):
            j = int(random.uniform(0, self.m))
        return j

    def _clip_alpha(self, j, h, l):
        """Clip alpha values that are greater than h or less than l"""
        a_j = self.alphas[j]
        if a_j > h:
            a_j = h
        if a_j < l:
            a_j = l
        return a_j


    def _select_j(self, i, error_i):
        max_k = -1
        max_delta_error = 0
        error_j = 0
        self.error_cache[i]  = [1, error_i]
        # Valid means it has already been calculated
        # The list contains the index of the valid errors
        valid_error_cache_list = np.nonzero(self.error_cache[:, 0].A)[0]

        if (len(valid_error_cache_list)) > 1:
            for k in valid_error_cache_list:
                if k == i:
                    continue
                error_k = self._calc_error_cache(k)
                delta_error = abs(error_i - error_k)
                if (delta_error > max_delta_error):
                    max_delta_error = delta_error
                    max_k = k
                    error_j = error_k
            return max_k, error_j
        else:
            j = self._select_j_rand(i)
            error_j = self._calc_error_cache(j)
        return j, error_j


    def _update_error_k(self, k):
        error_k = self._calc_error_cache(k)
        self.error_cache[k] = [1, error_k]


    def _inner_l(self, i):
        error_i = self._calc_error_cache(i)

        if ((self.label_mat[i]*error_i < -self.tol) and (self.alphas[i] < self.C))\
            or ((self.label_mat[i]*error_i > self.tol) and (self.alphas[i] > 0)):
            j, error_j = self._select_j(i, error_i)
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()

            if (self.label_mat[i] != self.label_mat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                print("L == H")
                return 0

            eta = float(2 * self.X[i, :] * self.X[j, :].T \
                        - self.X[i, :] * self.X[i, :].T\
                        - self.X[j, :] * self.X[j, :].T)
            if eta >= 0:
                print("eta >= 0")
                return 0
            self.alphas[j] -= self.label_mat[j] * (error_i - error_j) / eta
            self.alphas[j] = self._clip_alpha(j, H, L)
            self._update_error_k(j)

            if (abs(self.alphas[j] - alpha_j_old) < 0.00001):
                print("j not moving enough")
                return 0
            self.alphas[i] += self.label_mat[j] * self.label_mat[i]\
                              * (alpha_j_old - self.alphas[j])
            self._update_error_k(i)

            b1 = self.b - error_i \
                 - self.label_mat[i] * (self.alphas[i] - alpha_i_old)\
                 * self.X[i, :] * self.X[i, :].T \
                 - self.label_mat[j] * (self.alphas[j] - alpha_j_old)\
                 * self.X[i, :] * self.X[j, :].T
            b2 = self.b - error_j \
                 - self.label_mat[i] * (self.alphas[i] - alpha_i_old)\
                 * self.X[i, :] * self.X[j, :].T \
                 - self.label_mat[j] * (self.alphas[j] - alpha_j_old)\
                 * self.X[j, :] * self.X[j, :].T
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = float(b1)
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = float(b2)
            else:
                self.b = float((b1 + b2) / 2)
            return 1
        else:
            return 0


    def optimise(self):
        iter = 0
        entire_set = True
        alpha_paris_changed = 0
        
        while (iter < self.max_iter) \
              and ((alpha_paris_changed > 0) or entire_set):

            alpha_paris_changed = 0
            if entire_set:
                for i in range(self.m):
                    alpha_paris_changed += self._inner_l(i)
                print('full_set, iter: {0}, i: {1}, pairs changed {2}'.format(
                    iter, i, alpha_paris_changed
                ))
                iter += 1
            else:
                non_bound_idx = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in non_bound_idx:
                    alpha_paris_changed += self._inner_l(i)
                    print('non-bound, iter {0}, i: {1}, pairs changed {2}'.format(
                        iter, i, alpha_paris_changed
                    ))
                
                iter += 1
            if entire_set:
                entire_set = False
            elif alpha_paris_changed == 0:
                entire_set = True
            print('iteration number: {0}'.format(iter))
        return self.b, self.alphas


    def calc_weights(self):
        m, n = self.X.shape
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(self.alphas[i]*self.label_mat[i], self.X[i, :].T)
        self.w = w
        return w
        

if __name__ == '__main__':
    data_mat, label_mat = load_dataset('testSetCh06.txt')

    optimiser = Optimiser(data_mat, label_mat, 0.6, 0.001, 40)

    b, alphas = optimiser.optimise()

    ws = optimiser.calc_weights()

    print(ws, b)
    plt.scatter(data_mat[label_mat.A1>0, 0].A1, data_mat[label_mat.A1>0, 1].A1,
                c='r', marker='s')
    plt.scatter(data_mat[label_mat.A1<0, 0].A1, data_mat[label_mat.A1<0, 1].A1,
                c='g', marker='o')

    for i, selected in enumerate(alphas>0):
        if selected:
            plt.scatter(data_mat[i,0], data_mat[i,1], edgecolors='b', facecolors='None', s=300,
                        marker='o')
    plt.show()