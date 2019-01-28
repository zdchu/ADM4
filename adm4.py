import numpy as np
from utils import *
import IPython
from collections import defaultdict
from tick.hawkes import HawkesADM4, SimuHawkesExpKernels, SimuHawkesMulti


class ADM4(object):
    def __init__(self, para):
        self.para = para
        self.Z1 = np.zeros_like((para.K, para.K))
        self.U1 = np.zeros_like((para.K, para.K))
        self.mu = np.ones(para.K)
        self.A = np.random.uniform(0.5, 0.9, (para.K, para.K))

    def fit(self, mark_seq, time_seq):
        para = self.para
        T_start = time_seq[0]
        T_stop = time_seq[-1]

        dT = T_stop - time_seq
        GK = kernel_integration(dT, para.decay)

        Aest = self.A.copy()
        muest = self.mu.copy()

        if para.low_rank:
            UL = np.zeros_like(Aest)
            ZL = Aest.copy()

        if para.sparse:
            US = np.zeros_like(Aest)
            ZS = Aest.copy()

        for o in range(para.max_iter):
            rho = para.rho * (1.1 ** o)
            for n in range(para.em_max_iter):
                NLL = 0 # negative log-likelihood

                Amu = np.zeros_like(muest)
                Bmu = Amu.copy()

                CmatA = np.zeros_like(Aest)
                AmatA = CmatA.copy()
                BmatA = CmatA.copy()

                if para.low_rank:
                    BmatA = BmatA + rho * (UL -ZL)
                    AmatA = AmatA + rho

                if para.sparse:
                    BmatA = BmatA + rho * (US - ZS)
                    AmatA = AmatA + rho

                Amu = Amu + T_stop - T_start

                for i in range(len(mark_seq)):
                    time = time_seq[i]
                    mark = mark_seq[i]

                    BmatA[:, mark] = BmatA[:, mark] + [GK[i]] * para.K

                    lambdai = muest[mark]
                    pii = muest[mark]
                    pij = []
                    if i > 1:
                        tj = time_seq[:i]
                        uj = mark_seq[:i]

                        dt = time - tj
                        gij = exp_kernel(dt, para.decay)

                        auiuj = Aest[mark, uj]
                        pij = auiuj * gij
                        lambdai = lambdai + np.sum(pij)
                    NLL = NLL - np.log(lambdai)
                    pii = pii / lambdai

                    if i > 1:
                        pij = pij / lambdai
                        if np.sum(pij) > 0:
                            for j in range(len(uj)):
                                uuj = uj[j]
                                CmatA[mark, uuj] = CmatA[mark, uuj] - pij[j]
                                # CmatA[uuj, mark] = CmatA[uuj, mark] - pij[j]
                    Bmu[mark] = Bmu[mark] + pii
                mu = Bmu / Amu
                if para.sparse or para.low_rank:
                    A = (-BmatA + np.sqrt(np.square(BmatA) - 8 * rho * CmatA))/(4 * rho)
                else:
                    A = -CmatA / BmatA
                Err = np.sum(abs(A-Aest)) / np.sum(abs(Aest))
                Aest = A.copy()
                muest = mu.copy()
                self.A = Aest.copy()
                self.mu = muest.copy()

                if Err < para.threshold:
                    break

            if para.sparse:
                threshold = para.alpha / rho
                ZS = soft_thres_S(Aest+US, threshold)
                US = US + Aest - ZS


if __name__ == '__main__':
    # test


    end_time = 10000
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

    para = Parameters()

    learner = ADM4(para)
    learner.fit(mark_seq=type_seq, time_seq=time_seq)
    IPython.embed()

