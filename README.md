# Solving Wave PDE Equation with MLP

One of the simplest examples of solving PDEs with neural networks is considered in this notebook. A problem is set up in such a way that it admits an exact solution, which can be compared to a PINN approximation. The heat transfer example is based on the **one-dimensional wave equation**:

$$
\frac{\partial^2 u}{\partial t^2} - {c^2} \frac{\partial^2 u}{\partial x^2} = 0, \quad
t \in [0, T], \quad x \in [0, L].
$$

Dirichlet boundary conditions \( u(t, 0) = u(t, L) = 0 \) for \( t \in [0, T] \) and initial conditions \( u(0, x) = \sin \left( \frac{n \pi x}{L} \right) \) and \( u_t(0,x) = \frac{n \pi c}{L}\sin \left( \frac{n \pi x}{L} \right) \) for \( x \in [0, L] \) and a certain \( n \in \{1, 2, \ldots\} \) are imposed. Through **separation of variables** one can obtain the factorized solution as:

$$
u(t, x) = \sin \left( \frac{n \pi}{L} x \right) \left( \cos \left( \frac{n \pi}{L} ct \right) + \sin \left( \frac{n \pi}{L} ct \right) \right)
$$

Equations with other differential operators or boundary conditions can be addressed analogously though. The goal is then to construct a NN model \( u_{\boldsymbol{\theta}}(t, \boldsymbol{x}) \) that approximately solves the governing equations. It will be convenient in the following to define the **residual** of the NN approximation as:

$$
r_{\boldsymbol{\theta}}(t, \boldsymbol{x}) = \frac{\partial^2 u}{\partial t^2} - {c^2} \frac{\partial^2 u}{\partial x^2}
$$

One can now construct a **physics-based loss function**:

$$
L_{\mathrm{physics}} = \lambda_r L_{\mathrm{residual}} + \lambda_i L_{\mathrm{initial}} + \lambda_b L_{\mathrm{boundary}}
$$

that is tailored for solving the PDE. It contains three components that penalize nonzero residuals and deviations from the initial and boundary conditions, respectively. The relative importance of those terms can be adjusted with scalar weights \( \lambda_r, \lambda_i, \lambda_b > 0 \). The different loss terms are explicitly given as:

$$
\begin{align*}
L_{\mathrm{residual}} &= \frac{1}{N_r} \sum_{j=1}^{N_r}
\left( r_{\boldsymbol{\theta}}(t_j^{(r)}, \boldsymbol{x}_j^{(r)}) \right)^2,
\quad t_j^{(r)} \in [0, T], \quad \boldsymbol{x}_j^{(r)} \in \Omega, \\
L_{\mathrm{initial}} &= \frac{1}{N_i} \sum_{j=1}^{N_i}
\left( u_0(\boldsymbol{x}_j^{(i)}) -
u_{\boldsymbol{\theta}}(0, \boldsymbol{x}_j^{(i)}) \right)^2,
\quad \boldsymbol{x}_j^{(i)} \in \Omega, \\
L_{\mathrm{boundary}} &= \frac{1}{N_b} \sum_{j=1}^{N_b}
\left( u_{\mathrm{b}}(t_j^{(b)}, \boldsymbol{x}_j^{(b)}) -
u_{\boldsymbol{\theta}}(t_j^{(b)}, \boldsymbol{x}_j^{(b)}) \right)^2,
\quad t_j^{(b)} \in [0, T], \quad \boldsymbol{x}_j^{(b)} \in \partial \Omega.
\end{align*}
$$

Here, the **collocation points** \( \{(t_j^{(r)}, \boldsymbol{x}_j^{(r)})\}_{j=1}^{N_r} \) test the residual within the domain. Similarly, points at the space-time boundary \( \{\boldsymbol{x}_j^{(i)}\}_{j=1}^{N_i} \) and \( \{(t_j^{(b)}, \boldsymbol{x}_j^{(b)})\}_{j=1}^{N_b} \) test the boundary conditions. An approximate solution \( u_{\hat{\boldsymbol{\theta}}}(t, \boldsymbol{x}) \) can eventually be computed by finding the NN weights \( \hat{\boldsymbol{\theta}} = \operatorname{argmin}_{\boldsymbol{\theta}} L_{\mathrm{physics}} \) that minimize the physics loss.

## PINN Loss

The procedure presented so far actually makes for a generic PDE solver. In a wider context though, it may be important to incorporate some actual experimental data into the scientific modeling process. This could compensate for the inevitable uncertainties and inadequacies to some degree. PINNs offer an elegant mechanism to combine physical knowledge with real data. Given a set of data \( \{(t_i, \boldsymbol{x}_i, u_{\mathrm{meas}}(t_i, \boldsymbol{x}_i))\}_{i=1}^N \) one can simply consider an additional **regression loss**:

$$
L_{\mathrm{data}} = \frac{1}{N} \sum_{i=1}^N \left( u_{\mathrm{meas}}(t_i, \boldsymbol{x}_i) - u_{\boldsymbol{\theta}}(t_i, \boldsymbol{x}_i) \right)^2
$$

It is remarked here that, in a surrogate modeling context, such input-output data could in principle also come from a high-fidelity simulator. A PINN can be trained by minimizing the physics and regression losses as a function of the NN weights. For the sake of completeness, the complete **PINN loss** is written as:

$$
L = L_{\mathrm{data}} + \lambda_r L_{\mathrm{residual}} + \lambda_i L_{\mathrm{initial}} + \lambda_b L_{\mathrm{boundary}}
$$
