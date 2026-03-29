"""
Nonlinear MPC Controller for 2D Kuramoto-Sivashinsky Equation using CasADi + IPOPT

The controller uses exact dense spatial matrices derived from the 
spectral (Crank-Nicolson / Forward Euler) scheme to model KS dynamics natively in CasADi.
"""

import casadi as ca
import numpy as np
import time

class KSMPC2D:
    """
    Nonlinear Model Predictive Controller for the 2D KS Equation.
    """
    
    def __init__(self, N, L, dt, centers, sigma, horizon,
                 Q=1.0, R=0.01, u_min=-50, u_max=50, terminal_weight=10.0):
        self.N = N
        self.N2 = N * N
        self.L = L
        self.dt = dt
        self.dx = L / N
        self.centers = np.array(centers)
        self.sigma = sigma
        self.n_controls = len(centers)
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.terminal_weight = terminal_weight
        self.u_min = u_min
        self.u_max = u_max
        
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        print("Building forcing matrix G...")
        self.G = np.zeros((self.N2, self.n_controls))
        for j, c in enumerate(self.centers):
            dx_dist = np.abs(self.X - c[0])
            dx_dist = np.minimum(dx_dist, L - dx_dist)
            dy_dist = np.abs(self.Y - c[1])
            dy_dist = np.minimum(dy_dist, L - dy_dist)
            dist_sq = dx_dist**2 + dy_dist**2
            self.G[:, j] = np.exp(-0.5 * dist_sq / (sigma**2)).flatten()
            
        self.G_ca = ca.DM(self.G)
        
        print(f"Building dynamics matrices for N={N} (Warning: N^2 x N^2 matrices can be LARGE!)...")
        self._build_dynamics_matrix()
        
        print("Building CasADi Opti MPC NLP...")
        self._build_mpc()
        print("MPC setup complete.")
    
    def _build_dynamics_matrix(self):
        N = self.N
        dx = self.dx
        dt = self.dt
        L = self.L
        N2 = self.N2
        
        kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        L_linear = (KX**2 + KY**2) - (KX**2 + KY**2)**2

        denom = 1.0 - (dt / 2.0) * L_linear
        num_A = 1.0 + (dt / 2.0) * L_linear
        
        diag_A_hat = num_A / denom
        diag_B_hat = dt / denom
        diag_Dx_hat = 1j * KX
        diag_Dy_hat = 1j * KY

        self.A_dyn = np.zeros((N2, N2))
        self.B_dyn = np.zeros((N2, N2))
        self.D_x = np.zeros((N2, N2))
        self.D_y = np.zeros((N2, N2))

        t0 = time.time()
        for i in range(N):
            for j in range(N):
                e_ij = np.zeros((N, N))
                e_ij[i, j] = 1.0
                
                e_ij_hat = np.fft.fft2(e_ij)
                
                col = i * N + j
                self.A_dyn[:, col] = np.fft.ifft2(diag_A_hat * e_ij_hat).real.flatten()
                self.B_dyn[:, col] = np.fft.ifft2(diag_B_hat * e_ij_hat).real.flatten()
                self.D_x[:, col]   = np.fft.ifft2(diag_Dx_hat * e_ij_hat).real.flatten()
                self.D_y[:, col]   = np.fft.ifft2(diag_Dy_hat * e_ij_hat).real.flatten()
                
            if (i+1) % max(1, N//10) == 0:
                print(f"  Matrix build progress: {i+1}/{N} row blocks ({(time.time()-t0):.2f}s)")

        self.A_ca = ca.DM(self.A_dyn)
        self.B_ca = ca.DM(self.B_dyn)
        self.Dx_ca = ca.DM(self.D_x)
        self.Dy_ca = ca.DM(self.D_y)
        
    def _dynamics_step(self, u, ctrl):
        u_x = ca.mtimes(self.Dx_ca, u)
        u_y = ca.mtimes(self.Dy_ca, u)
        
        # In 2D KS, the nonlinear term is -0.5 * (|grad u|^2) 
        nonlinear_part = -0.5 * (u_x * u_x + u_y * u_y)
        forcing_part = ca.mtimes(self.G_ca, ctrl)
        
        return ca.mtimes(self.A_ca, u) + ca.mtimes(self.B_ca, nonlinear_part + forcing_part)
        
    def _build_mpc(self):
        self.opti = ca.Opti()
        
        N2 = self.N2
        H = self.horizon
        n_ctrl = self.n_controls
        
        self.U = self.opti.variable(N2, H + 1)
        self.A = self.opti.variable(n_ctrl, H)
        
        self.u0_param = self.opti.parameter(N2)
        self.u_ref_param = self.opti.parameter(N2)
        
        cost = 0
        for k in range(H):
            state_err = self.U[:, k+1] - self.u_ref_param
            cost += self.Q * ca.sumsqr(state_err)
            cost += self.R * ca.sumsqr(self.A[:, k])
        
        terminal_err = self.U[:, H] - self.u_ref_param
        cost += self.terminal_weight * self.Q * ca.sumsqr(terminal_err)
        
        self.opti.minimize(cost)
        
        self.opti.subject_to(self.U[:, 0] == self.u0_param)
        
        for k in range(H):
            u_k = self.U[:, k]
            a_k = self.A[:, k]
            u_kp1_pred = self._dynamics_step(u_k, a_k)
            self.opti.subject_to(self.U[:, k+1] == u_kp1_pred)
        
        self.opti.subject_to(self.opti.bounded(self.u_min, self.A, self.u_max))
        
        from config import ipopt_options
        self.opti.solver('ipopt', ipopt_options)
        
        self.U_init = None
        self.A_init = None
        
    def solve(self, u0_2d, u_ref_2d, warm_start=True):
        u0 = u0_2d.flatten()
        u_ref = u_ref_2d.flatten()
        
        self.opti.set_value(self.u0_param, u0)
        self.opti.set_value(self.u_ref_param, u_ref)
        
        if warm_start and self.U_init is not None:
            self.opti.set_initial(self.U, self.U_init)
            self.opti.set_initial(self.A, self.A_init)
        else:
            U_guess = np.zeros((self.N2, self.horizon + 1))
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                U_guess[:, k] = (1 - alpha) * u0 + alpha * u_ref
            
            A_guess = np.zeros((self.n_controls, self.horizon))
            self.opti.set_initial(self.U, U_guess)
            self.opti.set_initial(self.A, A_guess)
        
        try:
            sol = self.opti.solve()
            
            U_opt = sol.value(self.U)
            A_opt = sol.value(self.A)
            
            self.U_init = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
            temp_A = A_opt[:, 1:] if self.horizon > 1 else np.zeros((self.n_controls, 0))
            self.A_init = np.hstack([temp_A, A_opt[:, -1:]]) if self.horizon > 0 else A_opt
            
            # Reshape next state to 2D
            u_next_opt_2d = U_opt[:, 1].reshape((self.N, self.N))
            
            return A_opt[:, 0], u_next_opt_2d, A_opt
            
        except Exception as e:
            print(f"MPC solve failed: {e}. Opti status: {self.opti.debug.show_infeasibilities()}")
            try:
                a_opt = self.opti.debug.value(self.A)[:, 0]
                a_opt = np.clip(a_opt, self.u_min, self.u_max)
                return a_opt, None, None
            except:
                print("Falling back to zero control.")
                return np.zeros(self.n_controls), None, None
