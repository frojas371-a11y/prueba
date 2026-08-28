"""
Algoritmo variacional de Uhlmann para el sistema COMPUESTO de dos qubits
(Villavicencio et al., arXiv:2301.04766), optimizando de verdad el ansatz
ambiental de SU(4) -- los 15 angulos discutidos antes -- en vez de usar el
atajo cerrado (SVD) de dos_qubits_uhlmann.py.

Ansatz de Cartan/KAK para el link de cada paso (V_E: SU(4), 15 parametros):

    V_E(theta) = (u1 (x) u2) . exp[i(a XX + b YY + c ZZ)] . (u3 (x) u4)

  - u1,u2,u3,u4 in SU(2), parametrizados Rz(*)Ry(*)Rz(*)  -> 4*3 = 12 angulos
  - (a,b,c): nucleo entrelazante                          -> 3 angulos
  - total: 15 = dim SU(4)

El costo es identico en forma al del qubit unico:
    L_k(theta) = -Re(z_k) + eta*Im(z_k)^2
con z_k = <psi_tilde_k|(I (x) V_E(theta))|psi_{k+1}>, calculado en el espacio
de 16 dimensiones (2 qubits sistema + 2 qubits ambiente). Se optimiza con
BFGS, con warm start desde el theta optimo del paso anterior.
"""
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from uhlmann_variational import canonical_purification, hadamard_test_complex
from dos_qubits_uhlmann import H_composite, rho_composite  # reutiliza el modelo fisico

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
XX, YY, ZZ = np.kron(sx, sx), np.kron(sy, sy), np.kron(sz, sz)


def Rz(a):
    return expm(-0.5j * a * sz)


def Ry(a):
    return expm(-0.5j * a * sy)


def su2(theta3):
    a, b, c = theta3
    return Rz(a) @ Ry(b) @ Rz(c)


def V_E15(theta):
    """Ansatz universal de SU(4) (descomposicion de Cartan/KAK), 15 params."""
    u1, u2 = su2(theta[0:3]), su2(theta[3:6])
    a, b, c = theta[6:9]
    core = expm(1j * (a * XX + b * YY + c * ZZ))
    u3, u4 = su2(theta[9:12]), su2(theta[12:15])
    return np.kron(u1, u2) @ core @ np.kron(u3, u4)


def apply_gauge4(psi, W):
    """Accion derecha del gauge para el sistema compuesto (dim ambiente=4)."""
    return np.kron(W.T, I4) @ psi


def optimize_link_15(psi_k_tilde, psi_next, Wk, theta0=None, eta=20.0,
                      n_restarts=4, rng=None):
    """Igual que optimize_link de uhlmann_variational.py, pero con el
    ansatz de 15 parametros V_E15 en vez de Rz-Ry-Rz.

    El paisaje de optimizacion en 15 dimensiones tiene minimos locales
    reales: arrancando siempre en theta=0 (o cerca), BFGS puede quedarse
    atascado en la solucion "sin entrelazar" (nucleo a=b=c=0) aunque el
    optimo verdadero si necesite un poco de entrelazamiento. Se mitiga con
    varios arranques (el warm start del paso anterior + perturbaciones
    aleatorias) y se toma el mejor.
    """
    if theta0 is None:
        theta0 = np.zeros(15)
    if rng is None:
        rng = np.random.default_rng(0)

    def loss(theta):
        L = V_E15(theta)
        Wnext = Wk @ L
        trial = apply_gauge4(psi_next, Wnext)
        x, y = hadamard_test_complex(psi_k_tilde, trial)
        return -x + eta * y * y

    starts = [theta0]
    for _ in range(n_restarts - 1):
        starts.append(theta0 + rng.normal(scale=0.5, size=15))

    best = None
    for s0 in starts:
        res = minimize(loss, s0, method="BFGS", options={"gtol": 2e-8, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res

    L = V_E15(best.x)
    Wnext = Wk @ L
    x, y = hadamard_test_complex(psi_k_tilde, apply_gauge4(psi_next, Wnext))
    return L, Wnext, x + 1j * y, best.x


def uhlmann_loop_15(rho_fn, N=24, eta=20.0, n_restarts=4, seed=0):
    """Lazo cerrado completo del sistema compuesto, optimizando los 15
    parametros del ansatz de SU(4) en cada uno de los N pasos."""
    rng = np.random.default_rng(seed)
    phis = np.linspace(0, 2 * np.pi, N + 1)
    psi = [canonical_purification(rho_fn(p)) for p in phis]
    W = np.eye(4, dtype=complex)
    psi_tilde = apply_gauge4(psi[0], W)
    theta_guess = np.zeros(15)
    entangling_norms = []
    for k in range(N):
        L, W, _, theta_guess = optimize_link_15(
            psi_tilde, psi[k + 1], W, theta_guess, eta, n_restarts=n_restarts, rng=rng)
        entangling_norms.append(np.linalg.norm(theta_guess[6:9]))
        psi_tilde = apply_gauge4(psi[k + 1], W)
    X, Y = hadamard_test_complex(psi[0], apply_gauge4(psi[-1], W))
    return X + 1j * Y, np.array(entangling_norms)


def make_fig5_circuito(outpath, g_max=1.3, T_max=1.0, n_g=11, n_T=11,
                        N=12, n_restarts=2, log_every=True):
    """Reproduce la Fig. 5 (Phi^AB en g-T, theta=pi/2) barriendo una malla
    y optimizando de verdad, en cada punto, los 15 parametros del ansatz
    de SU(4) -- no el atajo cerrado de dos_qubits_uhlmann.py. Malla
    deliberadamente mas gruesa que la version SVD (90x90): cada punto
    cuesta ~20-30s porque BFGS en 15 dimensiones con varios reinicios es
    mucho mas caro que la solucion cerrada."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_vals = np.linspace(1e-3, g_max, n_g)
    T_vals = np.linspace(1e-3, T_max, n_T)
    Phi = np.zeros((n_T, n_g))
    Ent = np.zeros((n_T, n_g))

    t_start = time.time()
    total = n_g * n_T
    done = 0
    for iT, T in enumerate(T_vals):
        beta = 1.0 / T
        for ig, g in enumerate(g_vals):
            def rfn(phi, beta=beta, g=g):
                r, _ = rho_composite(beta, phi, g)
                return r
            z, ent = uhlmann_loop_15(rfn, N=N, n_restarts=n_restarts, seed=hash((iT, ig)) % (2**31))
            Phi[iT, ig] = np.abs(np.arctan2(z.imag, z.real)) / np.pi
            Ent[iT, ig] = ent.mean()
            done += 1
            if log_every:
                elapsed = time.time() - t_start
                eta_s = elapsed / done * (total - done)
                print(f"[{done}/{total}] g={g:.3f} T={T:.3f} Phi/pi={Phi[iT,ig]:.2f} "
                      f"|nucleo|={Ent[iT,ig]:.4f}  transcurrido={elapsed:.0f}s  "
                      f"restante~{eta_s:.0f}s", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = axes[0].pcolormesh(g_vals, T_vals, Phi, cmap="Blues", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im0, ax=axes[0], label=r"$\Phi^{AB}/\pi$")
    axes[0].set_xlabel("g"); axes[0].set_ylabel("T")
    axes[0].set_title("Circuito variacional (15 parámetros SU(4))")

    im1 = axes[1].pcolormesh(g_vals, T_vals, Ent, cmap="magma", shading="auto")
    fig.colorbar(im1, ax=axes[1], label=r"$|(a,b,c)|$ (núcleo entrelazante)")
    axes[1].set_xlabel("g"); axes[1].set_ylabel("T")
    axes[1].set_title("Cuánto entrelazamiento de ambiente usa el óptimo")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print("Figura guardada en", outpath, " tiempo total=", time.time() - t_start, "s")
    return g_vals, T_vals, Phi, Ent


if __name__ == "__main__":
    # --- 1) verificacion puntual contra el metodo cerrado (SVD) ---
    from dos_qubits_uhlmann import uhlmann_loop_exact

    beta, g = 1 / 0.5, 0.6

    def rfn(phi, beta=beta, g=g):
        r, _ = rho_composite(beta, phi, g)
        return r

    t0 = time.time()
    z15, ent = uhlmann_loop_15(rfn, N=24, n_restarts=4)
    t_15 = time.time() - t0

    t0 = time.time()
    z_exact = uhlmann_loop_exact(rfn, N=24)
    t_exact = time.time() - t0

    print(f"beta={beta:.3f}  g={g}")
    print(f"  circuito 15 params : Z_U = {z15:.6f}   ({t_15:.2f} s)")
    print(f"  cerrado (SVD)      : Z_U = {z_exact:.6f}   ({t_exact:.4f} s)")
    print(f"  |diferencia| = {abs(z15 - z_exact):.2e}")
    print(f"  norma del nucleo entrelazante (a,b,c) por paso: "
          f"min={ent.min():.4f} max={ent.max():.4f} media={ent.mean():.4f}")

    # --- 2) reproduccion de la Fig. 5 con el circuito real (malla mas gruesa) ---
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "figuras", "fig5_circuito_15params.png")
    make_fig5_circuito(outpath, g_max=1.3, T_max=1.0, n_g=11, n_T=11, N=12, n_restarts=2)
