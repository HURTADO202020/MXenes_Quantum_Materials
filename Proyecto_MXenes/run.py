"""
================================================================
 run.py — Punto de entrada de la simulación Monte Carlo
 Importa parámetros y funciones, y corre el flujo completo.
================================================================
"""

import numpy as np
from parametros import *
from funciones import *


def main():

    np.random.seed(seed)
    sembrar_numba(seed)

    # --- Paso 0: estimación de campo medio ---
    sc_chico     = construir_supercelda(L_list[0], cell, ti_positions)
    idx_ch, J_ch = construir_libreta_vecinos(sc_chico, d_anillos, J_meV, tol)
    z_estimado   = verificar_coordinacion(idx_ch, J_ch, J_meV)
    Tc_mf        = estimar_Tc_campo_medio(z_estimado, J_meV, k_B)
    print(f"Estimacion de campo medio: T_c ~ {Tc_mf:.0f} K (sobreestimada)")
    print(f"Sugerencia: barra T entre ~{0.2*Tc_mf:.0f} K y ~{1.0*Tc_mf:.0f} K")
    print(f"Coordinaciones z_k: {z_estimado}")
    print()

    # --- Paso 1: correr simulación para cada L ---
    todos = {}
    for L in L_list:
        print(f"Corriendo L={L} ({6*L*L} espines)...")
        resultados, z_coord = correr_simulacion(
            L, cell, ti_positions, d_anillos, J_meV, tol,
            epsilon_base, T_min, T_max, N_temps,
            N_term, N_skip, N_meas, k_B)
        todos[L] = resultados
        print(f"  L={L} listo. Coordinaciones: {z_coord}")

    # --- Paso 2: calcular cumulante de Binder U_L(T) ---
    cumulantes = {}
    for L in L_list:
        cumulantes[L] = {
            T: calcular_cumulante(d["M2"], d["M4"])
            for T, d in todos[L].items()
        }

    # --- Paso 3: localizar cruces entre pares de L ---
    cruces  = []
    pares_L = []
    for L1, L2 in zip(L_list[:-1], L_list[1:]):
        print(f"Buscando cruce entre L={L1} y L={L2}...")
        T_cruce = encontrar_cruce(cumulantes[L1], cumulantes[L2])
        cruces.append(T_cruce)
        pares_L.append(L1)
        print(f"  Cruce encontrado en T = {T_cruce:.1f} K")

    # --- Paso 4: extrapolar al sistema infinito ---
    T_c_kelvin = extrapolar_a_infinito(cruces, pares_L)
    print()
    print(f"T_c (sistema infinito) = {T_c_kelvin:.1f} K")



    from ase.neighborlist import neighbor_list
    sc = construir_supercelda(6, cell, ti_positions)
    i_idx, j_idx, dist = neighbor_list('ijd', sc, cutoff=7.0)

    import matplotlib.pyplot as plt
    plt.hist(dist, bins=100)
    plt.xlabel('Distancia (Å)')
    plt.ylabel('Conteo')
    plt.title('Histograma de distancias')
    plt.savefig('histograma_distancias.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
