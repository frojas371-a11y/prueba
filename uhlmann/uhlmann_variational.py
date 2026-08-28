"""
Algoritmo variacional de transporte de Uhlmann para un qubit termico.

Modelo
------
rho(beta,theta,phi) = exp[-beta n(theta,phi).sigma]/Z
                     = 1/2 [I - tanh(beta) n.sigma].

El lazo cerrado recorre phi en [0,2*pi] con theta fijo.

Algoritmo
---------
En cada link k -> k+1 se optimiza una unitaria ambiental de un qubit

    L_k = Rz(a) Ry(b) Rz(c)

para que el solapamiento entre purificaciones vecinas transportadas sea
real, positivo y maximo. En un procesador cuantico la parte real e
imaginaria se obtienen con dos pruebas de Hadamard (SWAP/overlap test).

El gauge acumulado W se actualiza por la derecha:

    W_{k+1} = W_k L_k.

El W final es la holonomia discreta de Uhlmann. No se usa descomposicion
polar para determinar el transporte: todo sale de maximizar el
solapamiento medido.

Este modulo reconstruye el codigo del PDF original, lo completa para que
tambien reproduzca las Figuras 3, 4 y 5 (que en el documento fuente no
llegaban a aparecer en el bloque principal), y separa el calculo de la
generacion de graficas para poder reutilizarlo desde otros scripts
(por ejemplo, el que arma el PDF de deduccion).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)


def Rz(a):
    """Rotacion de un qubit exp(-i a sigma_z/2)."""
    return expm(-0.5j * a * sz)


def Ry(a):
    """Rotacion de un qubit exp(-i a sigma_y/2)."""
    return expm(-0.5j * a * sy)


def V_env(theta):
    """Ansatz ambiental universal SU(2): Rz(a) Ry(b) Rz(c)."""
    a, b, c = theta
    return Rz(a) @ Ry(b) @ Rz(c)


def rho(beta, theta, phi):
    """Matriz densidad termica exacta 2x2 para H = n(theta,phi).sigma."""
    n = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    H = n[0] * sx + n[1] * sy + n[2] * sz
    return 0.5 * (I2 - np.tanh(beta) * H)


def canonical_purification(r):
    """Construye |sqrt(rho)>> por vectorizacion columna (vec, order='F').

    Se usa solo para simular la purificacion fisica sistema-ambiente.
    En hardware esta operacion se reemplaza por la dilatacion fisica
    U_SE(lambda)|psi0,0E>.
    """
    vals, vecs = np.linalg.eigh(r)
    sr = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None))) @ vecs.conj().T
    return sr.reshape(-1, order="F")


def apply_gauge(psi, W):
    """Aplica la accion derecha de Uhlmann w -> w W.

    vec(w W) = (W^T tensor I) vec(w).
    """
    return np.kron(W.T, I2) @ psi


def hadamard_test_complex(psi_ref, psi_trial):
    """Emulacion ideal sin ruido de las dos pruebas de Hadamard.

    Devuelve (x, y) = (Re<psi_ref|psi_trial>, Im<psi_ref|psi_trial>).
    En hardware, x e y son valores de expectativa de la ancilla.
    """
    z = np.vdot(psi_ref, psi_trial)
    return z.real, z.imag


def optimize_link(psi_k_tilde, psi_next, Wk, theta0=None, eta=20.0):
    """Optimiza un link discreto de Uhlmann.

    El costo L = -Re(z) + eta*Im(z)^2 maximiza el solapamiento real
    positivo y penaliza la fase residual. Devuelve el link incremental
    L_k, el gauge actualizado W_{k+1}, el solapamiento optimo z_k y los
    angulos de Euler optimizados (para usarlos como warm start).
    """
    if theta0 is None:
        theta0 = np.zeros(3)

    def loss(theta):
        L = V_env(theta)
        Wnext = Wk @ L
        trial = apply_gauge(psi_next, Wnext)
        x, y = hadamard_test_complex(psi_k_tilde, trial)
        return -x + eta * y * y

    res = minimize(loss, theta0, method="BFGS",
                    options={"gtol": 2e-8, "maxiter": 100})
    L = V_env(res.x)
    Wnext = Wk @ L
    x, y = hadamard_test_complex(psi_k_tilde, apply_gauge(psi_next, Wnext))
    return L, Wnext, x + 1j * y, res.x


def uhlmann_loop(beta, theta=np.pi / 2, N=24, eta=20.0):
    """Ejecuta el lazo variacional cerrado completo de Uhlmann.

    Devuelve distancias de Bures locales, links discretos, conexiones
    aproximadas por link, la holonomia, la amplitud final de Uhlmann y
    su fase.
    """
    phis = np.linspace(0, 2 * np.pi, N + 1)
    psi = [canonical_purification(rho(beta, theta, p)) for p in phis]

    W = np.eye(2, dtype=complex)
    psi_tilde = apply_gauge(psi[0], W)
    theta_guess = np.zeros(3)

    links, bures, connection, local_z = [], [], [], []
    dphi = 2 * np.pi / N
    for k in range(N):
        L, W, z, theta_guess = optimize_link(
            psi_tilde, psi[k + 1], W, theta_guess, eta)
        links.append(L)
        local_z.append(z)
        bures.append(np.sqrt(max(0.0, 2 * (1 - abs(z)))))
        connection.append(logm(L) / dphi)
        psi_tilde = apply_gauge(psi[k + 1], W)

    U_hol = W
    X, Y = hadamard_test_complex(psi[0], apply_gauge(psi[-1], U_hol))
    Z_U = X + 1j * Y
    Phi_U = np.arctan2(Y, X)
    return dict(
        Bures=np.array(bures), Links=links, Connection=connection,
        U_hol=U_hol, Z_U=Z_U, Phi_U=Phi_U, local_z=np.array(local_z),
        phis=phis,
    )


def theory_equator(beta):
    """Amplitud y fase analiticas de Uhlmann para theta=pi/2."""
    G = np.cos(np.pi * (1 - 1 / np.cosh(beta)))
    Phi = 0.0 if G >= 0 else np.pi
    return G, Phi


def find_variational_Tc(T, Gvar):
    """Interpola linealmente el primer cruce por cero de Gvar(T)."""
    idx = np.where(Gvar[:-1] * Gvar[1:] < 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return T[i] - Gvar[i] * (T[i + 1] - T[i]) / (Gvar[i + 1] - Gvar[i])


def run_experiment(T_min=0.30, T_max=1.25, n_points=24, N=24, eta=20.0):
    """Corre el barrido en temperatura usado en las Figuras 2-4.

    Devuelve un diccionario con T, valores variacionales/teoricos de la
    amplitud y la fase de Uhlmann, el error absoluto y la temperatura
    critica variacional estimada por interpolacion del cruce por cero.
    """
    Tc_theory = 1 / np.arccosh(2)
    T = np.linspace(T_min, T_max, n_points)
    Gvar, Gth, PhiVar, PhiTh = [], [], [], []
    for temp in T:
        out = uhlmann_loop(1 / temp, N=N, eta=eta)
        Gvar.append(out["Z_U"].real)
        PhiVar.append(out["Phi_U"] / np.pi)
        g_th, phi_th = theory_equator(1 / temp)
        Gth.append(g_th)
        PhiTh.append(phi_th / np.pi)

    Gvar, Gth = np.asarray(Gvar), np.asarray(Gth)
    PhiVar, PhiTh = np.asarray(PhiVar), np.asarray(PhiTh)
    Tc_var = find_variational_Tc(T, Gvar)
    error = np.abs(Gvar - Gth)
    return dict(
        T=T, Gvar=Gvar, Gth=Gth, PhiVar=PhiVar, PhiTh=PhiTh,
        error=error, Tc_theory=Tc_theory, Tc_var=Tc_var,
    )


def run_bures_profile(T_values=(0.550, 0.759, 1.000), N=24, eta=20.0):
    """Distancia de Bures local D_B(rho_k,rho_{k+1}) a lo largo del lazo.

    Se evalua para varios valores de T (Figura 5). En theta=pi/2 la
    geometria es uniforme en phi, por lo que D_B es aproximadamente
    constante a lo largo del lazo para cada T.
    """
    profiles = {}
    for T in T_values:
        out = uhlmann_loop(1 / T, N=N, eta=eta)
        phi_mid = 0.5 * (out["phis"][:-1] + out["phis"][1:])
        profiles[T] = (phi_mid, out["Bures"])
    return profiles


def make_figures(outdir, exp=None, profiles=None):
    """Genera y guarda en outdir las Figuras 2, 3, 4 y 5 del PDF original."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    if exp is None:
        exp = run_experiment()
    if profiles is None:
        profiles = run_bures_profile()

    T, Gvar, Gth = exp["T"], exp["Gvar"], exp["Gth"]
    PhiVar, PhiTh = exp["PhiVar"], exp["PhiTh"]
    Tc_th, Tc_var = exp["Tc_theory"], exp["Tc_var"]

    # Figura 2: amplitud de Uhlmann Re(G_U) vs T
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(T, Gvar, "o", label="Circuit simulation: Re $G_U$")
    ax.plot(T, Gth, "-", color="tab:orange",
            label=r"Theory: $\cos[\pi(1-\mathrm{sech}\,\beta)]$")
    ax.axhline(0, color="tab:blue", lw=1)
    ax.axvline(Tc_th, ls="--", color="tab:blue", label=r"$T_c^{th}$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"Re $G_U$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figura2_amplitud_uhlmann.png"), dpi=160)
    plt.close(fig)

    # Figura 3: salto de fase Phi_U/pi vs T
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(T, PhiVar, "o", label="Circuit simulation (variational + Hadamard)")
    ax.plot(T, PhiTh, "-", color="tab:orange", label="Theory")
    ax.axvline(Tc_th, ls="--", color="tab:blue",
               label=r"$T_c^{th}=%.6f$" % Tc_th)
    if Tc_var is not None:
        ax.axvline(Tc_var, ls=":", color="tab:blue",
                   label=r"$T_c^{var}\approx%.6f$" % Tc_var)
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\Phi_U/\pi$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figura3_fase_uhlmann.png"), dpi=160)
    plt.close(fig)

    # Figura 4: error absoluto de la amplitud
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(T, exp["error"], "o-")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$|G_U^{var}-G_U^{th}|$")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figura4_error_absoluto.png"), dpi=160)
    plt.close(fig)

    # Figura 5: distancia de Bures local a lo largo del lazo
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for Tval, (phi_mid, bures) in profiles.items():
        ax.plot(phi_mid, bures, "-", label=f"T = {Tval:.3f}")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$D_B(\rho_k,\rho_{k+1})$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figura5_bures_local.png"), dpi=160)
    plt.close(fig)

    return exp, profiles


def main():
    Tc_theory = 1 / np.arccosh(2)
    print("Analytical Tc =", Tc_theory)
    exp = run_experiment()
    if exp["Tc_var"] is not None:
        print("Variational Tc =", exp["Tc_var"])
    print("Max |G_var - G_th| =", exp["error"].max())

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
    make_figures(outdir, exp=exp)
    print("Figuras guardadas en", outdir)


if __name__ == "__main__":
    main()
