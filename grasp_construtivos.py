"""
Problema da Mochila Limitada - Comparacao dos 8 Mecanismos Gulosos Aleatorios
==============================================================================
Implementa e compara os 8 esquemas construtivos descritos em:
  Lopes et al. (Eds.), Meta-Heuristicas em Pesquisa Operacional (2013)

  1. Semi-guloso          — RCL por faixa de qualidade (alpha-threshold)
  2. Amostragem Gulosa    — amostra p candidatos, escolhe o melhor
  3. Aleatorio Guloso     — p primeiros itens aleatorios, resto guloso
  4. Guloso Proporcional  — selecao estocastica proporcional a densidade
  5. GRASP Reativo        — alpha adaptativo via historico de desempenho
  6. Memoria Longo Prazo  — conjunto elite orienta a construcao
  7. Amostragem Viciada   — selecao por rank com funcao de vicio
  8. Perturbacao de Custos — ruido na densidade, construcao deterministica

Cada mecanismo esta implementado em comparacao_mecanismos/mecanismo_N_*.py.
A infraestrutura compartilhada (instancia, solver, busca local) esta em
comparacao_mecanismos/base.py.

Ambiente: conda env 'mochila'
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from comparacao_mecanismos import (
    generate_instance, solve_pulp, run_grasp,
    construct_semi_greedy,
    construct_greedy_sampling,
    construct_random_greedy,
    construct_proportional_greedy,
    run_grasp_reactive,
    run_grasp_memory,
    construct_biased_sampling,
    run_grasp_cost_perturbation,
)


# ==============================================================================
# Configuracao dos experimentos
# ==============================================================================

SIZES  = [10000]
N_RUNS = 5
N_ITER = 300

# Parametros ajustados para reduzir tempo nos mecanismos 2-8 (exceto semi-guloso):
#   - Amos. Gulosa:    p=3  (era 5) — grupos maiores, menos sortings por grupo
#   - Alea. Guloso:    p_pct=0.05 (era 0.10) — fase aleatoria mais curta
#   - GRASP Reativo:   alpha_set com 5 valores (era 9) — menos bookkeeping
#   - Memoria L. Prazo: elite_size=5 (era 10) — _calcular_I e _atualizar_elite
#                        fazem O(elite_size x n) por iteracao
CONFIGS = {
    "Semi-guloso\n(a=0.30)": {
        "fn":     construct_semi_greedy,
        "kwargs": {"alpha": 0.30},
        "color":  "#1f77b4",
    },
    "Amos. Gulosa\n(p=3)": {
        "fn":     construct_greedy_sampling,
        "kwargs": {"p": 3},
        "color":  "#ff7f0e",
    },
    "Alea. Guloso\n(p=5%)": {
        "fn":     construct_random_greedy,
        "kwargs": {},
        "color":  "#2ca02c",
        "p_pct":  0.05,
    },
    "Guloso\nProporcional": {
        "fn":     construct_proportional_greedy,
        "kwargs": {},
        "color":  "#d62728",
    },
    "Amos. Viciada\n(linear)": {
        "fn":     construct_biased_sampling,
        "kwargs": {"bias": "linear"},
        "color":  "#9467bd",
    },
    "GRASP\nReativo": {
        "runner":        run_grasp_reactive,
        "runner_kwargs": {"alpha_set": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "color":         "#8c564b",
    },
    "Memoria\nLongo Prazo": {
        "runner":        run_grasp_memory,
        "runner_kwargs": {"elite_size": 5},
        "color":         "#e377c2",
    },
    "Perturbacao\nde Custos": {
        "runner":        run_grasp_cost_perturbation,
        "runner_kwargs": {},
        "color":         "#7f7f7f",
    },
}

PULP_LABEL = "PuLP\n(otimo)"
PULP_COLOR = "#17becf"


# ==============================================================================
# Execucao dos experimentos
# ==============================================================================

def run_experiments():
    print("=" * 70)
    print("  COMPARACAO DOS 8 MECANISMOS CONSTRUTIVOS GRASP")
    print("  Mochila Limitada  |  Busca local ativada")
    print("=" * 70)

    resultados   = {sz: {nome: [] for nome in CONFIGS} for sz in SIZES}
    tempos       = {sz: {nome: [] for nome in CONFIGS} for sz in SIZES}
    tempos_pulp  = {sz: [] for sz in SIZES}
    convergencia = {sz: {nome: [] for nome in CONFIGS} for sz in SIZES}

    for sz in SIZES:
        print(f"\n[n = {sz}  |  runs = {N_RUNS}]")

        for run in range(N_RUNS):
            inst = generate_instance(sz, seed=run * 100 + sz)
            val_opt, t_opt, optimal = solve_pulp(inst)
            tag = "" if optimal else " [TL]"
            print(f"  run={run:02d} | otimo={val_opt:.0f}{tag} ({t_opt:.2f}s)", end="")
            tempos_pulp[sz].append(t_opt)

            for nome, cfg in CONFIGS.items():
                if "runner" in cfg:
                    runner_kwargs = cfg.get("runner_kwargs", {})
                    val_g, t_g, conv = cfg["runner"](
                        inst, n_iter=N_ITER, seed=run, **runner_kwargs
                    )
                else:
                    fn     = cfg["fn"]
                    kwargs = cfg["kwargs"].copy()
                    if "p_pct" in cfg:
                        kwargs["p"] = max(1, int(cfg["p_pct"] * sz))
                    val_g, t_g, conv = run_grasp(inst, fn, kwargs,
                                                 n_iter=N_ITER, seed=run)

                gap = (val_opt - val_g) / (val_opt + 1e-9) * 100
                resultados[sz][nome].append(gap)
                tempos[sz][nome].append(t_g)
                convergencia[sz][nome].append(conv)
                lbl = nome.split("\n")[0]
                print(f" | {lbl}={val_g:.0f}(gap={gap:.1f}%,t={t_g:.2f}s)", end="")
            print()

    return resultados, tempos, tempos_pulp, convergencia


# ==============================================================================
# Graficos
# ==============================================================================

def plot_results(resultados, tempos, tempos_pulp):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Comparacao dos 8 Mecanismos Construtivos GRASP — Mochila Limitada\n"
        f"(busca local ativada, {N_ITER} iteracoes por run)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, len(SIZES), figure=fig, hspace=0.45, wspace=0.35)

    nomes  = list(CONFIGS.keys())
    cores  = [cfg["color"] for cfg in CONFIGS.values()]
    labels = [n.split("\n")[0] for n in nomes]

    for col, sz in enumerate(SIZES):
        # --- Linha 1: gap em relacao ao otimo ---
        ax   = fig.add_subplot(gs[0, col])
        dados = [resultados[sz][nome] for nome in nomes]
        bp   = ax.boxplot(dados, patch_artist=True, widths=0.5,
                          medianprops={"color": "black", "lw": 2})
        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        ax.set_title(f"Distancia ao otimo (gap%) — n = {sz}", fontsize=11)
        ax.set_ylabel("Gap em relacao ao otimo (%)")
        ax.set_xticks(range(1, len(nomes) + 1))
        ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="gray", lw=0.8, ls="--")

        # --- Linha 2: tempo de execucao (inclui PuLP como ultima caixa) ---
        ax      = fig.add_subplot(gs[1, col])
        dados_t = [tempos[sz][nome] for nome in nomes] + [tempos_pulp[sz]]
        all_labels = labels + [PULP_LABEL.split("\n")[0]]
        all_cores  = cores + [PULP_COLOR]

        bp = ax.boxplot(dados_t, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "lw": 2})
        for patch, cor in zip(bp["boxes"], all_cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        ax.set_title(f"Tempo de execucao (s) — n = {sz}", fontsize=11)
        ax.set_ylabel("Tempo total (s)")
        ax.set_xticks(range(1, len(all_labels) + 1))
        ax.set_xticklabels(all_labels, fontsize=7, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

        # Destaca a caixa do PuLP com borda mais grossa
        bp["boxes"][-1].set_linewidth(2)
        bp["boxes"][-1].set_edgecolor("#0a6575")

    out = ("C:/Users/breno/Documentos/Faculdade/OTM em Redes/"
           "Trabalho 1 - Mochila/resultados_construtivos.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGrafico salvo em: {out}")
    plt.show()


# ==============================================================================
# Resumo tabular
# ==============================================================================

def print_summary(resultados, tempos, tempos_pulp):
    W = 80
    print("\n" + "=" * W)
    print(f"  Resumo — gap medio (%) e tempo medio (s) em relacao ao otimo PuLP/CBC")
    print(f"  {'Metodo':<28}", end="")
    for sz in SIZES:
        print(f"  {'n='+str(sz):>14}", end="")
    print()
    print("-" * W)
    for nome in CONFIGS:
        lbl = nome.replace("\n", " ")
        print(f"  {lbl:<28}", end="")
        for sz in SIZES:
            gap_medio = np.mean(resultados[sz][nome])
            t_medio   = np.mean(tempos[sz][nome])
            print(f"  {gap_medio:>5.3f}%/{t_medio:>5.2f}s", end="")
        print()
    print("-" * W)
    lbl = "PuLP (otimo)"
    print(f"  {lbl:<28}", end="")
    for sz in SIZES:
        t_medio = np.mean(tempos_pulp[sz])
        print(f"  {'---':>5}/{t_medio:>5.2f}s", end="")
    print()
    print("=" * W)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    resultados, tempos, tempos_pulp, convergencia = run_experiments()
    print_summary(resultados, tempos, tempos_pulp)
    plot_results(resultados, tempos, tempos_pulp)
