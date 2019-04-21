import numpy as np
import IPython
from utils import *
import time as Time
# from cython_utils import *
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti

class ADM4(object):
    def __init__(self, para):
        self.para = para
        self.Z1 = np.zeros_like((para.K, para.K))
        self.U1 = np.zeros_like((para.K, para.K))
        self.mu = np.ones(para.K)
        self.A = np.random.uniform(0.5, 0.9, (para.K, para.K))

    def fit(self, mark_seq, time_seq):
        para = self.para
        max_iter = para.max_iter
        em_max_iter = para.em_max_iter
        decay = para.decay
        K = para.K
        sparse = para.sparse
        rho = para.rho
        threshold = para.threshold
        alpha = para.alpha

        T_start = time_seq[0]
        T_stop = time_seq[-1] + 1

        dT = T_stop - time_seq
        GK = kernel_integration(dT, decay)

        Aest = self.A.copy()
        muest = self.mu.copy()

        gij = [np.array([0] * K)]
        BmatA_raw = np.zeros_like(Aest)
        for i in range(len(time_seq)):
            time = time_seq[i]
            mark = mark_seq[i]
            BmatA_raw[:, mark] = BmatA_raw[:, mark] + [GK[i]] * K
            if i > 0:
                tj = time_seq[i-1]
                uj = mark_seq[i-1]
                gij_last = gij[-1].copy()

                gij_now = (gij_last / decay) * decay * np.exp(-decay * (time - tj))
                gij_now[uj] += decay * np.exp(-decay*(time - tj))

                gij.append(gij_now)
        gij_raw = np.array(gij)

        if sparse:
            US = np.zeros_like(Aest)
            ZS = Aest.copy()

        for o in range(max_iter):
            rho = rho * (1.1 ** o)
            for n in range(em_max_iter):
                Amu = np.zeros_like(muest)
                Bmu = Amu.copy()

                CmatA = np.zeros_like(Aest)
                BmatA = BmatA_raw.copy()

                if sparse:
                    BmatA = BmatA + rho * (US - ZS)

                Amu = Amu + T_stop - T_start
                gij = Aest[mark_seq] * gij_raw
                gij = np.insert(gij, 0, values=muest[mark_seq], axis=1)
                pij = gij / np.sum(gij, axis=1)[:, np.newaxis]
                np.add.at(Bmu, mark_seq, pij[:, 0])
                np.add.at(CmatA, mark_seq, -pij[:, 1:])

                mu = Bmu / Amu
                if sparse:
                    A = (-BmatA + np.sqrt(np.square(BmatA) - 8 * rho * CmatA))/(4 * rho)
                else:
                    A = -CmatA / BmatA

                Err = np.sum(abs(A-Aest)) / np.sum(abs(Aest))
                Aest = A.copy()
                muest = mu.copy()
                self.A = Aest.copy()
                self.mu = muest.copy()

                if Err < threshold:
                    break

            if sparse:
                threshold = alpha / rho
                ZS = soft_thres_S(Aest+US, threshold)
                US = US + Aest - ZS


if __name__ == '__main__':
    end_time = 50000
    n_realizations = 1
    decay = 3
    baseline =np.ones(6) * .03
    adjacency = np.zeros((6, 6))
    adjacency[2:, 2:] = np.ones((4, 4)) * 0.1
    adjacency[:3, :3] = np.ones((3, 3)) * 0.15

    hawkes_exp_kernels = SimuHawkesExpKernels(adjacency=adjacency, decays=decay,
                                              baseline=baseline, end_time=end_time,
                                              verbose=False, seed=1039)

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)
    multi.end_time = [(i + 1) / n_realizations * end_time for i in range(n_realizations)]
    multi.simulate()

    type_seq = []
    time_seq = []
    for i, d in enumerate(multi.timestamps[0]):
        for j in d:
            type_seq.append(i)
            time_seq.append(j)
    type_seq = np.array(type_seq)
    time_seq = np.array(time_seq)

    type_seq = type_seq[np.argsort(time_seq)]
    time_seq = time_seq[np.argsort(time_seq)]

    para = Parameters(decay=3, K=6)
    IPython.embed()