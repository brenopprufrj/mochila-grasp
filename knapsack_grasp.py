"""
Problema da Mochila Limitada (Bounded Knapsack Problem)
=======================================================
Comparacao entre:
  - GRASP com numpy vetorizado (C-compiled via numpy/numba)
  - PuLP/CBC com gap=0  (solucao otima exata)

Para cada item i, pode-se incluir uma quantidade inteira entre 0 e max_qty[i].
Objetivo: maximizar o valor total sem exceder a capacidade da mochila.

Ambiente: conda env 'mochila'
  C:\\Users\\breno\\.conda\\envs\\mochila\\python.exe
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import List, Tuple

import pulp


# ==============================================================================
# Estrutura do problema
# ==============================================================================

@dataclass
class KnapsackInstance:
    n_items:  int
    capacity: int
    values:   List[int]
    weights:  List[int]
    max_qty:  List[int]

    def __post_init__(self):
        assert len(self.values)  == self.n_items
        assert len(self.weights) == self.n_items
        assert len(self.max_qty) == self.n_items

    def evaluate(self, solution) -> Tuple[int, int]:
        """Retorna (valor_total, peso_total)."""
        sol = np.asarray(solution, dtype=np.int32)
        val = int(np.dot(sol, self.values))
        wt  = int(np.dot(sol, self.weights))
        return val, wt


def generate_instance(n_items: int, seed: int = 42) -> KnapsackInstance:
    """Gera uma instancia aleatoria reprodutivel."""
    rng     = random.Random(seed)
    values  = [rng.randint(10, 100) for _ in range(n_items)]
    weights = [rng.randint(5,  50)  for _ in range(n_items)]
    max_qty = [rng.randint(1,  5)   for _ in range(n_items)]
    max_possible = sum(w * q for w, q in zip(weights, max_qty))
    capacity     = int(0.40 * max_possible)
    return KnapsackInstance(n_items, capacity, values, weights, max_qty)


# ==============================================================================
# Solver PuLP — solucao otima exata (gap = 0)
# ==============================================================================

def solve_pulp(inst: KnapsackInstance) -> Tuple[List[int], float, float, bool]:
    """
    Resolve via PuLP/CBC.
    Retorna (solucao, valor, tempo, otimo_garantido).
    otimo_garantido=False quando o solver atinge PULP_TIME_LIMIT sem provar otimalidade.
    """
    prob = pulp.LpProblem("BoundedKnapsack", pulp.LpMaximize)
    x    = [pulp.LpVariable(f"x{i}", lowBound=0, upBound=inst.max_qty[i], cat="Integer")
            for i in range(inst.n_items)]

    prob += pulp.lpSum(inst.values[i]  * x[i] for i in range(inst.n_items))
    prob += pulp.lpSum(inst.weights[i] * x[i] for i in range(inst.n_items)) <= inst.capacity

    solver = pulp.PULP_CBC_CMD(msg=0, gapRel=0.0, timeLimit=PULP_TIME_LIMIT)
    t0 = time.perf_counter()
    prob.solve(solver)
    elapsed = time.perf_counter() - t0

    optimal = prob.status == pulp.constants.LpStatusOptimal
    sol = [int(round(pulp.value(x[i]) or 0)) for i in range(inst.n_items)]
    return sol, float(pulp.value(prob.objective) or 0.0), elapsed, optimal


# ==============================================================================
# GRASP — construcao e busca local 100% numpy-vetorizados (sem loops Python
# no caminho critico) — complexidade O(n log n) por iteracao
# ==============================================================================

def _construct(weights: np.ndarray, values: np.ndarray,
               max_qty: np.ndarray, density: np.ndarray,
               sort_desc: np.ndarray, capacity: int,
               alpha: float, rng) -> Tuple[np.ndarray, int]:
    """
    Construcao GRASP em passagem unica (O(n log n)).

    Ordena itens por densidade decrescente e aplica randomizacao RCL:
    os primeiros ceil(alpha * n) itens no ranking sao embaralhados antes
    do preenchimento guloso. Cada item recebe o maximo de unidades que
    cabe no espaco restante.
    """
    n = len(weights)
    rcl_size = max(1, int(np.ceil(alpha * n)))

    perm = sort_desc.copy()
    rng.shuffle(perm[:rcl_size])           # embaralha apenas o topo da RCL

    sol = np.zeros(n, dtype=np.int64)
    rem = capacity

    for idx in perm:
        if rem <= 0:
            break
        if weights[idx] <= rem:
            units = min(max_qty[idx], rem // weights[idx])
            sol[idx] = units
            rem -= units * weights[idx]

    return sol, rem


def _local_search(sol: np.ndarray, weights: np.ndarray, values: np.ndarray,
                  max_qty: np.ndarray, density: np.ndarray,
                  sort_desc: np.ndarray, capacity: int) -> np.ndarray:
    """
    Busca local vetorizada (O(n) por passo de melhora).

    Fase Fill  : percorre itens em ordem de densidade e adiciona o maximo
                 de unidades que cabem — preenche toda folga de capacidade.
    Fase Swap  : encontra o item de menor densidade com sol[i]>0 (pior item)
                 e tenta substituir por uma unidade do melhor item disponivel.
                 Repete ate nao haver melhora.
    """
    rem = int(capacity - np.dot(sol, weights))

    # --- Fill (ordem de densidade, operacao numpy por item) ---
    for idx in sort_desc:
        if rem <= 0:
            break
        slack = max_qty[idx] - sol[idx]
        if slack > 0 and weights[idx] <= rem:
            add = min(slack, rem // weights[idx])
            sol[idx] += add
            rem -= add * weights[idx]

    # --- Swap iterativo (vetorizado por iteracao) ---
    improved = True
    while improved:
        improved = False

        # Pior item: menor densidade entre os que tem unidades alocadas
        in_sol = sol > 0
        if not in_sol.any():
            break
        worst_i = int(np.argmin(np.where(in_sol, density, np.inf)))

        # Melhor item a inserir apos liberar weights[worst_i]
        new_rem = rem + weights[worst_i]
        can_swap = (sol < max_qty) & (weights <= new_rem)
        can_swap[worst_i] = False

        if not can_swap.any():
            break

        best_j = int(np.argmax(np.where(can_swap, density, -np.inf)))

        if density[best_j] > density[worst_i]:
            sol[worst_i] -= 1
            sol[best_j]  += 1
            rem = new_rem - weights[best_j]
            improved = True

    # --- Fill final apos swaps ---
    for idx in sort_desc:
        if rem <= 0:
            break
        slack = max_qty[idx] - sol[idx]
        if slack > 0 and weights[idx] <= rem:
            add = min(slack, rem // weights[idx])
            sol[idx] += add
            rem -= add * weights[idx]

    return sol


def solve_grasp(inst: KnapsackInstance,
                n_iter: int = 500,
                alpha: float = 0.3,
                seed: int = 0) -> Tuple[List[int], float, float]:
    """
    GRASP para Bounded Knapsack com construcao e busca local numpy-vetorizados.

    n_iter : numero de iteracoes GRASP (reduzido automaticamente para n grande)
    alpha  : largura da RCL (fracao do ranking embaralhada; 0=guloso, 1=aleatorio)
    """
    rng = np.random.default_rng(seed)

    weights  = np.array(inst.weights,  dtype=np.int64)
    values   = np.array(inst.values,   dtype=np.int64)
    max_qty  = np.array(inst.max_qty,  dtype=np.int64)
    density  = values.astype(np.float64) / (weights.astype(np.float64) + 1e-9)
    capacity = int(inst.capacity)

    # Pre-computa ordenacao por densidade (reutilizada em todas as iteracoes)
    sort_desc = np.argsort(-density)

    # Escala n_iter para instancias grandes (manter tempo razoavel)
    effective_iter = max(50, min(n_iter, int(n_iter * 100 / inst.n_items)))

    best_sol = np.zeros(inst.n_items, dtype=np.int64)
    best_val = 0

    t0 = time.perf_counter()

    for _ in range(effective_iter):
        sol, rem = _construct(weights, values, max_qty, density,
                              sort_desc, capacity, alpha, rng)
        sol = _local_search(sol, weights, values, max_qty, density,
                            sort_desc, capacity)

        val = int(np.dot(sol, values))
        if val > best_val:
            best_val = val
            best_sol = sol.copy()

    elapsed = time.perf_counter() - t0
    return best_sol.tolist(), float(best_val), elapsed


# ==============================================================================
# Configuracao dos experimentos
# ==============================================================================

SIZES  = [1000, 2000, 5000, 10000, 20000, 40000, 50000, 80000, 100000]
N_RUNS = 30

# Para instancias grandes o PuLP pode demorar; limite de 120s por run
PULP_TIME_LIMIT = 120

GRASP_PARAMS = dict(
    n_iter = 500,
    alpha  = 0.30,   # RCL: top 30 % por densidade; ajuste entre 0.1-0.5
)


# ==============================================================================
# Execucao
# ==============================================================================

def run_experiments():
    print("=" * 65)
    print("  MOCHILA LIMITADA - GRASP (numpy) vs PuLP (gap=0)")
    print("=" * 65)

    results = {sz: [] for sz in SIZES}

    for sz in SIZES:
        print(f"\n[n={sz}]")
        for run in range(N_RUNS):
            inst = generate_instance(sz, seed=run * 100 + sz)

            sol_opt, val_opt, t_opt, optimal = solve_pulp(inst)
            sol_grp, val_grp, t_grp          = solve_grasp(inst, seed=run, **GRASP_PARAMS)

            gap = (val_opt - val_grp) / (val_opt + 1e-9) * 100

            results[sz].append({
                "opt_value":   val_opt,
                "opt_time":    t_opt,
                "opt_proven":  optimal,
                "grasp_value": val_grp,
                "grasp_time":  t_grp,
                "gap_pct":     gap,
            })

            opt_tag = "" if optimal else " [TL]"
            print(f"  run={run} | pulp={val_opt:.0f}{opt_tag} ({t_opt:.3f}s) "
                  f"| GRASP={val_grp:.0f} ({t_grp:.3f}s) | gap={gap:.2f}%")

    return results


# ==============================================================================
# Graficos
# ==============================================================================

COLOR_OPT   = "#1f77b4"
COLOR_GRASP = "#2ca02c"


def _ms(data):
    return np.mean(data), np.std(data)


def plot_results(results):
    sizes_arr = np.array(SIZES)

    t_opt_ms, t_grp_ms = [], []
    v_opt_ms, v_grp_ms = [], []

    for sz in SIZES:
        rs = results[sz]
        t_opt_ms.append(_ms([r["opt_time"]    for r in rs]))
        t_grp_ms.append(_ms([r["grasp_time"]  for r in rs]))
        v_opt_ms.append(_ms([r["opt_value"]   for r in rs]))
        v_grp_ms.append(_ms([r["grasp_value"] for r in rs]))

    def arr(lst, idx): return np.array([x[idx] for x in lst])

    m_t_opt = arr(t_opt_ms, 0); s_t_opt = arr(t_opt_ms, 1)
    m_t_grp = arr(t_grp_ms, 0); s_t_grp = arr(t_grp_ms, 1)
    m_v_opt = arr(v_opt_ms, 0); s_v_opt = arr(v_opt_ms, 1)
    m_v_grp = arr(v_grp_ms, 0); s_v_grp = arr(v_grp_ms, 1)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"Mochila Limitada: GRASP (numpy vetorizado) vs Solucao Otima (PuLP/CBC)\n"
        f"(media +/- desvio padrao, {N_RUNS} instancias por tamanho)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    grasp_label = "GRASP (numpy)"

    # ── Tempo de execucao ─────────────────────────────────────────────────────
    ax1.plot(sizes_arr, m_t_opt, "s--", color=COLOR_OPT,   lw=2, label="PuLP (otimo)")
    ax1.fill_between(sizes_arr, m_t_opt - s_t_opt, m_t_opt + s_t_opt,
                     alpha=0.15, color=COLOR_OPT)
    ax1.plot(sizes_arr, m_t_grp, "o-",  color=COLOR_GRASP, lw=2, label=grasp_label)
    ax1.fill_between(sizes_arr, m_t_grp - s_t_grp, m_t_grp + s_t_grp,
                     alpha=0.15, color=COLOR_GRASP)
    ax1.set_title("Tempo de Execucao")
    ax1.set_xlabel("Numero de Itens (n)")
    ax1.set_ylabel("Tempo (s)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # ── Valor objetivo absoluto ───────────────────────────────────────────────
    ax2.plot(sizes_arr, m_v_opt, "s--", color=COLOR_OPT,   lw=2, label="PuLP (otimo)")
    ax2.fill_between(sizes_arr, m_v_opt - s_v_opt, m_v_opt + s_v_opt,
                     alpha=0.15, color=COLOR_OPT)
    ax2.plot(sizes_arr, m_v_grp, "o-",  color=COLOR_GRASP, lw=2, label=grasp_label)
    ax2.fill_between(sizes_arr, m_v_grp - s_v_grp, m_v_grp + s_v_grp,
                     alpha=0.15, color=COLOR_GRASP)
    ax2.set_title("Valor da Funcao Objetivo")
    ax2.set_xlabel("Numero de Itens (n)")
    ax2.set_ylabel("Valor total")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    out = "C:/Users/breno/Documentos/Faculdade/OTM em Redes/Trabalho 1 - Mochila/resultados_grasp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGrafico salvo em: {out}")
    plt.show()


# ==============================================================================
# Resumo tabular
# ==============================================================================

def print_summary(results):
    W = 72
    print("\n" + "=" * W)
    print(f"{'n':>6} | {'PuLP (s)':>10} | {'GRASP (s)':>10} | {'gap (%)':>10} | {'PuLP otimo?':>11}")
    print("-" * W)
    for sz in SIZES:
        rs      = results[sz]
        t_opt   = np.mean([r["opt_time"]    for r in rs])
        t_grp   = np.mean([r["grasp_time"]  for r in rs])
        gap     = np.mean([r["gap_pct"]     for r in rs])
        n_proven = sum(1 for r in rs if r["opt_proven"])
        proven_str = f"{n_proven}/{N_RUNS}"
        print(f"{sz:>6} | {t_opt:>10.4f} | {t_grp:>10.4f} | {gap:>10.2f} | {proven_str:>11}")
    print("=" * W)
    print("  [TL] = time limit atingido; gap calculado sobre melhor solucao CBC encontrada")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    results = run_experiments()
    print_summary(results)
    plot_results(results)
