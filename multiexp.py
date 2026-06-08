import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, envelope

SINAIS = {
    "real":     dict(caminho="result_real.npy",        passo_us=0.008, corte=300, n_exp=2),
    "simulado": dict(caminho="result_47_complete.npy", passo_us=0.004, corte=750, n_exp=1),
}

def faz_modelo(n):
    def modelo(t, *p):
        return sum(p[2*i]*np.exp(-p[2*i+1]*t) for i in range(n))
    return modelo

def analisa(caminho, passo_us, corte, n_exp, nome):
    fs = 1.0/(passo_us*1e-6)
    sinal = np.ravel(np.load(caminho).astype(float))[corte:]

    env = np.abs(np.atleast_2d(envelope(sinal))[0])
    sinal_env = env / np.max(env)
    t_us = (np.arange(len(sinal_env)) + corte) * passo_us

    picos, _ = find_peaks(sinal_env, height=0.01, distance=int(fs*5e-6))
    t_picos = t_us[picos]
    amp = sinal_env[picos]
    t_rel = t_picos - t_picos[0]

    modelo = faz_modelo(n_exp)
    escalas = np.array([1.0, 3.0, 0.3, 0.1, 0.03])[:n_exp]
    estimativa_inicial = np.ravel([[1.0/n_exp, 0.1*e] for e in escalas])
    lo = [0]*(2*n_exp)
    hi = [2.0, 5.0]*n_exp
    popt, _ = curve_fit(modelo, t_rel, amp, p0=estimativa_inicial, bounds=(lo, hi), maxfev=200000)

    A, alfas = popt[0::2], popt[1::2]
    T = np.mean(np.diff(t_picos))
    f_rep = 1/(T*1e-6)

    termos = " + ".join(f"{A[i]:.3f}*exp(-{alfas[i]:.4f}*t)" for i in range(n_exp))
    print(f"\n[{nome}]")
    print(f"  y_{nome}(t) = {termos}")
    for i in range(n_exp):
        print(f"  alpha_{i+1} = {alfas[i]:.4f} Np/us   tau_{i+1} = {1/alfas[i]:.2f} us")

    return dict(nome=nome, t_us=t_us, env=sinal_env, t_picos=t_picos, amp=amp,
                t0=t_picos[0], popt=popt, modelo=modelo, alfas=alfas, f_rep=f_rep)

def main():
    R = {nome: analisa(**cfg, nome=nome) for nome, cfg in SINAIS.items()}
    cores = {"real": "tab:blue", "simulado": "tab:green"}

    plt.figure(figsize=(12, 6))
    for nome, D in R.items():
        c = cores[nome]
        t = np.linspace(0, D["t_us"][-1] - D["t0"], 600)
        plt.plot(D["t_us"], D["env"], color=c, alpha=0.4)
        plt.scatter(D["t_picos"], D["amp"], color=c, zorder=5)
        plt.plot(t + D["t0"], D["modelo"](t, *D["popt"]), "--", color=c, lw=2, label=nome)
    plt.xlabel(r"Tempo ($\mu$s)")
    plt.ylabel("Amplitude normalizada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(0, color="k", lw=0.8)
    for nome, D in R.items():
        c = cores[nome]
        plt.scatter(-D["alfas"]*1e6, np.zeros(len(D["alfas"])), color=c, marker="x", s=140, label=nome)
        plt.axhline(D["f_rep"]/1e3, color=c, ls=":", alpha=0.6)
    plt.xlabel(r"$s = -\alpha$ (1/s)")
    plt.ylabel("Frequência (kHz)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()