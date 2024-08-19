import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class FactorialIV:
    def __init__(self, outcome, treatment, instrument):
        self.outcome = outcome
        self.treatment = treatment
        self.instrument = instrument
        self.K = treatment.shape[1]
        self.J = 2 ** self.K
        self.fit()

    def _construct_grids(self):
        dz_vals = [np.array([1, 0])] * (2 * self.K)
        ps_grid = np.array(np.meshgrid(*[['a', 'n', 'c']] * self.K)).T.reshape(-1, self.K)
        dz_d_grid = np.array(np.meshgrid(*dz_vals)).T.reshape(-1, 2 * self.K)[:, :self.K]
        dz_z_grid = np.array(np.meshgrid(*dz_vals)).T.reshape(-1, 2 * self.K)[:, self.K:]
        return dz_d_grid, dz_z_grid, ps_grid

    def _build_matrices(self, dz_d_grid, dz_z_grid, ps_grid):
        ps_type = 2 + dz_z_grid - dz_d_grid + 2 * dz_d_grid * dz_z_grid
        ps_dict = ['a', ['n', 'c'], 'n', ['a', 'c']]

        ps_type = np.clip(ps_type, 0, 3)

        A = np.ones((len(dz_d_grid), len(ps_grid)))
        for k in range(self.K):
            for i in range(len(dz_d_grid)):
                k_mat = np.array([1 if p in ps_dict[int(ps_type[i, k])] else 0 for p in ps_grid[:, k]])
                A[i, :] *= k_mat

        B = np.zeros((len(dz_d_grid), len(dz_d_grid)))
        for i, row in enumerate(dz_d_grid):
            hold = [['n', 'c'], ['a', 'c']]
            this_strata = np.array(np.meshgrid(*[hold[int(x)] for x in row])).T.reshape(-1, self.K)
            s_names = [''.join(s) for s in this_strata.astype(str)]
            grab_rows = np.all(dz_d_grid == row, axis=1)
            B[grab_rows, grab_rows] = A[
                grab_rows, [j for j, s in enumerate([''.join(p) for p in ps_grid]) if s in s_names]]

        return A, B

    def _estimate_parameters(self, Y, D, Z):
        dz_d_grid, dz_z_grid, ps_grid = self._construct_grids()
        A, B = self._build_matrices(dz_d_grid, dz_z_grid, ps_grid)

        len_rho = A.shape[1]
        len_psi = B.shape[1]

        def gmm_objective(params):
            rho, psi = params[:len_rho], params[len_rho:len_rho + len_psi]
            residuals = Y - np.dot(D, rho) - np.dot(Z, psi)
            return np.dot(residuals.T, residuals)

        initial_guess = np.zeros(len_rho + len_psi)
        result = minimize(gmm_objective, initial_guess, method='BFGS')

        return result.x[:len_rho], result.x[len_rho:len_rho + len_psi], result.hess_inv

    def fit(self):
        Y, D, Z = self.outcome, self.treatment, self.instrument
        rho, psi, vcov = self._estimate_parameters(Y, D, Z)

        self.rho = rho
        self.psi = psi
        self.vcov = vcov
        self.eff_estimates = self._calculate_effects(rho, psi, vcov)

    def _calculate_effects(self, rho, psi, vcov):
        dz_d_grid, dz_z_grid, ps_grid = self._construct_grids()
        A, B = self._build_matrices(dz_d_grid, dz_z_grid, ps_grid)

        mcafe = self._calculate_marginalized_effects(A, rho, psi, vcov)
        pcafe = self._calculate_perfect_complier_effects(A, rho, psi, vcov)

        return {'MCAFE': mcafe, 'PCAFE': pcafe}

    def _calculate_marginalized_effects(self, A, rho, psi, vcov):
        mcafe_est = np.dot(A.T, psi) / np.dot(A.T, rho)
        mcafe_se = np.sqrt(np.diag(vcov)[:len(mcafe_est)]) / np.abs(np.dot(A.T, rho))
        return mcafe_est, mcafe_se

    def _calculate_perfect_complier_effects(self, A, rho, psi, vcov):
        pcafe_est = np.dot(A.T, psi) / np.dot(A.T, rho)
        pcafe_se = np.sqrt(np.diag(vcov)[len(pcafe_est):]) / np.abs(np.dot(A.T, rho))
        return pcafe_est, pcafe_se

    def summary(self):
        print(" Main effects among perfect compliers:\n")
        print("                                     tval  pval")
        for est, se in zip(self.eff_estimates['PCAFE'][0], self.eff_estimates['PCAFE'][1]):
            tval = est / se
            pval = 2 * norm.sf(np.abs(tval))
            print(f"{est: .5f} {se: .5f} {tval: .5f} {pval: .3f}")

        c_prob = self.rho[-1]
        c_prob_se = np.sqrt(self.vcov[-1, -1])
        print(f"\nEstimated prob. of perfect compliers:  {c_prob:.5f} \tSE =  {c_prob_se:.5f}")

    def tidy(self, conf_int=False, conf_level=0.95):
        alpha = (1 - conf_level) / 2
        z = norm.ppf(1 - alpha)

        rows = []
        for est, se in zip(self.eff_estimates['MCAFE'][0], self.eff_estimates['MCAFE'][1]):
            row = ["MCAFE", est, se]
            if conf_int:
                row += [est - z * se, est + z * se]
            rows.append(row)

        for est, se in zip(self.eff_estimates['PCAFE'][0], self.eff_estimates['PCAFE'][1]):
            row = ["PCAFE", est, se]
            if conf_int:
                row += [est - z * se, est + z * se]
            rows.append(row)

        columns = ["estimand", "estimate", "std.error"]
        if conf_int:
            columns += ["conf.low", "conf.high"]
        return pd.DataFrame(rows, columns=columns)
