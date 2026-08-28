"""
Fase de Uhlmann de los subsistemas de dos fermiones acoplados (arXiv:2301.04766),
en equilibrio termico, sobre el ecuador (theta=pi/2).

Modelo (Ec. 1 del paper): H(phi) = n(pi/2,phi).sigma (x) 1 + (g/2)(sx(x)sx - sy(x)sy),
n(pi/2,phi) = (cos phi, sin phi, 0), g = 2J/B0. Solo el qubit 1 esta "manejado"
por el campo; el 2 solo siente el acoplamiento anisotropico tipo pairing.

rho(beta,phi) = exp[-beta H(phi)]/Z  (sistema compuesto, 4x4).

Resultado analitico derivado (theta=pi/2): los DOS estados reducidos
rho1(phi)=Tr_2 rho, rho2(phi)=Tr_1 rho son, cada uno, un qubit termico
"tipo ecuador" que rota rigidamente con phi -- exactamente la misma forma
que uhlmann_variational.py ya resuelve -- pero con una magnitud de Bloch
efectiva distinta para cada fermion:

    Omega = sqrt(g^2+4)
    t1(beta,g) = (2/Omega) * tanh(beta*Omega/2)                    [manejado]
    t2(beta,g) = (2/Omega) * tanh(beta*Omega/2) * tanh(beta*g/2)   [no manejado]

    G_U^(i) = cos{ pi * [1 - sqrt(1-t_i^2)] },  i=1,2

t1 es monotona en beta y en g -> a lo sumo un cruce por cero (una transicion).
t2 tiene el factor extra tanh(beta g/2), que sube desde 0 y hace que t2(g)
suba y baje (no monotona) a beta fijo -> puede cruzar el umbral t=sqrt(3)/2
DOS veces: la "doble transicion topologica" del fermion no manejado.

Este script verifica esa prediccion analitica corriendo el ALGORITMO
VARIACIONAL real (reutilizando optimize_link de uhlmann_variational.py)
sobre las purificaciones canonicas de rho1(phi) y rho2(phi), sin usar en
ningun momento las formulas cerradas de arriba.
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uhlmann_variational import (
    canonical_purification, apply_gauge, optimize_link, hadamard_test_complex,
)

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)


def H_composite(phi, g, theta=np.pi / 2):
    """Ecuacion (1) del paper, con B0=2 (unidades ya reescaladas: H=H0/(B0/2))."""
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    field = n[0] * sx + n[1] * sy + n[2] * sz
    H = np.kron(field, I2) + (g / 2.0) * (np.kron(sx, sx) - np.kron(sy, sy))
    return H


def rho_composite(beta, phi, g, theta=np.pi / 2):
    """Construye rho via diagonalizacion exacta (estable para beta grande,
    a diferencia de expm(-beta H) que desborda numericamente a T->0)."""
    H = H_composite(phi, g, theta)
    evals, evecs = np.linalg.eigh(H)
    shifted = np.exp(-beta * (evals - evals.min()))
    w = shifted / shifted.sum()
    rho = (evecs * w) @ evecs.conj().T
    return rho.astype(complex), shifted.sum()


def partial_trace(rho4, keep):
    """Traza parcial de una matriz 4x4 (2 qubits), keep=1 o 2."""
    rho4 = rho4.reshape(2, 2, 2, 2)  # (q1,q2, q1',q2')
    if keep == 1:
        return np.einsum('ikjk->ij', rho4)
    return np.einsum('kikj->ij', rho4)


def optimal_link_exact(rho_k, rho_next):
    """Link optimo de Uhlmann de forma cerrada (sin optimizador), via la
    descomposicion polar de C=sqrt(rho_k) sqrt(rho_next): si C=U~ P (U~
    unitaria, P>=0), el V que maximiza Re<<w_k|(I⊗V)|w_{k+1}>> es V=U~*.
    Generaliza sin cambios a cualquier dimension (2 para 1 qubit, 4 para
    el sistema compuesto de Fig. 5)."""
    from scipy.linalg import sqrtm
    C = sqrtm(rho_k) @ sqrtm(rho_next)
    Us, _, Wh = np.linalg.svd(C)
    return (Us @ Wh).conj()


def uhlmann_loop_exact(rho_fn, N=24):
    """Igual que uhlmann_loop_generic pero resolviendo cada link de forma
    cerrada (SVD) en vez de optimizar -- exacto y mucho mas rapido, util
    para barridos 2D como la Fig. 5 del paper."""
    phis = np.linspace(0, 2 * np.pi, N + 1)
    rhos = [rho_fn(p) for p in phis]
    dim = rhos[0].shape[0]
    Idim = np.eye(dim, dtype=complex)
    psi = [canonical_purification(r) for r in rhos]
    W = np.eye(dim, dtype=complex)
    for k in range(N):
        L = optimal_link_exact(rhos[k], rhos[k + 1])
        W = W @ L
    psi_final = np.kron(W.T, Idim) @ psi[-1]
    X, Y = hadamard_test_complex(psi[0], psi_final)
    return X + 1j * Y


def uhlmann_loop_generic(rho_fn, beta, N=24, eta=20.0):
    """Igual que uhlmann_loop de uhlmann_variational.py, pero para CUALQUIER
    familia rho_fn(phi) de qubits (no solo el termico n.sigma)."""
    phis = np.linspace(0, 2 * np.pi, N + 1)
    psi = [canonical_purification(rho_fn(p)) for p in phis]
    W = np.eye(2, dtype=complex)
    psi_tilde = apply_gauge(psi[0], W)
    theta_guess = np.zeros(3)
    for k in range(N):
        _, W, _, theta_guess = optimize_link(psi_tilde, psi[k + 1], W, theta_guess, eta)
        psi_tilde = apply_gauge(psi[k + 1], W)
    X, Y = hadamard_test_complex(psi[0], apply_gauge(psi[-1], W))
    return X + 1j * Y


def theory_ti(beta, g, which):
    Omega = np.sqrt(g ** 2 + 4)
    t1 = (2.0 / Omega) * np.tanh(beta * Omega / 2.0)
    if which == 1:
        return t1
    return t1 * np.tanh(beta * g / 2.0)


def theory_GU(beta, g, which):
    t = theory_ti(beta, g, which)
    return np.cos(np.pi * (1 - np.sqrt(max(0.0, 1 - t ** 2))))


def run_sweep(beta, g_values, N=24):
    Gvar1, Gvar2, Gth1, Gth2 = [], [], [], []
    for g in g_values:
        def rho1_fn(phi, g=g):
            r4, _ = rho_composite(beta, phi, g)
            return partial_trace(r4, keep=1)

        def rho2_fn(phi, g=g):
            r4, _ = rho_composite(beta, phi, g)
            return partial_trace(r4, keep=2)

        Gvar1.append(uhlmann_loop_generic(rho1_fn, beta, N=N).real)
        Gvar2.append(uhlmann_loop_generic(rho2_fn, beta, N=N).real)
        Gth1.append(theory_GU(beta, g, 1))
        Gth2.append(theory_GU(beta, g, 2))
    return map(np.asarray, (Gvar1, Gvar2, Gth1, Gth2))


def make_figure(outpath, beta=3.0, g_max=6.0, n_points=40, N=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_values = np.linspace(1e-3, g_max, n_points)
    Gvar1, Gvar2, Gth1, Gth2 = run_sweep(beta, g_values, N=N)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(g_values, Gth1, "-", color="tab:blue", label=r"Teoría, qubit manejado (1)")
    ax.plot(g_values, Gvar1, "o", color="tab:blue", ms=4, label=r"Circuito, qubit 1")
    ax.plot(g_values, Gth2, "-", color="tab:red", label=r"Teoría, qubit no manejado (2)")
    ax.plot(g_values, Gvar2, "s", color="tab:red", ms=4, label=r"Circuito, qubit 2")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("g = 2J/B0  (acoplamiento)")
    ax.set_ylabel(r"$G_U$ del subsistema")
    ax.set_title(f"Fase de Uhlmann de los subsistemas vs. acoplamiento  (β={beta}, θ=π/2)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

    n1 = np.where(np.diff(np.sign(Gvar1)))[0]
    n2 = np.where(np.diff(np.sign(Gvar2)))[0]
    print(f"beta={beta}: cruces por cero qubit 1 (manejado)   = {len(n1)}  en g~{g_values[n1]}")
    print(f"beta={beta}: cruces por cero qubit 2 (no manejado) = {len(n2)}  en g~{g_values[n2]}")
    print("max |Gvar1-Gth1| =", np.max(np.abs(Gvar1 - Gth1)))
    print("max |Gvar2-Gth2| =", np.max(np.abs(Gvar2 - Gth2)))


def make_fig5_paper(outpath, g_max=1.6, T_max=1.0, n_g=90, n_T=90, N=24):
    """Reproduce la Fig. 5 del paper: mapa de densidad de Phi^AB(g,T) en
    theta=pi/2, usando el link optimo exacto (SVD), sin optimizador."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_vals = np.linspace(1e-3, g_max, n_g)
    T_vals = np.linspace(1e-3, T_max, n_T)
    Phi = np.zeros((n_T, n_g))

    for iT, T in enumerate(T_vals):
        beta = 1.0 / T
        for ig, g in enumerate(g_vals):
            def rfn(phi, beta=beta, g=g):
                r, _ = rho_composite(beta, phi, g)
                return r
            z = uhlmann_loop_exact(rfn, N=N)
            # theta=pi/2 => Phi es exactamente 0 o pi (mod 2pi): +pi y -pi
            # son la MISMA fase, así que se pliegan con abs() para evitar
            # el moteado numérico de atan2 cerca de Im(z)~0.
            Phi[iT, ig] = np.abs(np.arctan2(z.imag, z.real)) / np.pi

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.pcolormesh(g_vals, T_vals, Phi, cmap="Blues", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\Phi^{AB}/\pi$")
    ax.set_xlabel("g")
    ax.set_ylabel("T")
    ax.set_title(r"Fig. 5 reproducida: $\Phi^{AB}(g,T)$ en $\theta=\pi/2$")
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

    Tc_g0 = 1.0 / np.arccosh(2.0)
    print("T_c teorico en g=0 (=1/ln(2+sqrt3)) =", Tc_g0)
    return g_vals, T_vals, Phi


if __name__ == "__main__":
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")

    make_figure(os.path.join(figdir, "fig11_subsistemaB_doble_transicion.png"),
                beta=6.0, g_max=3.0, n_points=48, N=24)

    make_fig5_paper(os.path.join(figdir, "fig5_PhiAB_g_T.png"),
                     g_max=1.6, T_max=1.0, n_g=90, n_T=90, N=24)
    print("Figuras guardadas en", figdir)
