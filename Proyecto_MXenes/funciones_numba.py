import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import gridspec
import ase 
from ase import atoms
from ase.neighborlist import neighbor_list
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from numba import njit


#========== REPRODUCIBILIDAD DE NUMBA
@njit
def sembrar_numba(seed):

    np.random.seed(seed)




#============ ESTIMACIÓN DEL RANGO  DE TEMPERATURA=========================================
def estimar_Tc_campo_medio_numba(z_coordinaciones, J_meV, k_B):
    """
    Estima T_c (en Kelvin) por campo medio:
        T_c^MF = ( sum_k  z_k * |J_k| ) / k_B
    z_coordinaciones: array [z_1, z_2, z_3] (vecinos por anillo).
    Devuelve una temperatura en Kelvin (sobreestimada, solo orientativa:
    el T_c real suele estar entre 0.2 y 0.7 veces este valor en 2D).
    """
    suma_meV = np.sum(z_coordinaciones * np.abs(J_meV))   # energía en meV
    return suma_meV / k_B
#==========================================================================================



#============ RED Y LIBRETA DE VECINOS ====================================================def construir_supercelda(L, cell, ti_positions):
def construir_supercelda_numba(L, cell, ti_positions):
    """
    Replica la celda unidad L x L veces en el plano, con condiciones de
    contorno periódicas (PBC) en a y b, y no-periódica en c. Devuelve un
    objeto que conoce las posiciones de los 6*L*L átomos y la geometría.
    Implementación real: ase.Atoms(..., pbc=[True, True, False]).repeat((L, L, 1)).
    """
    #Construir la celda unidad con los 6 átomos de Ti
    simbolos = ['Ti'] * len(ti_positions) # ASE ya tiene elementos en su librería, aqui le paso 6 atomos de Ti

    #crea la celda base que repetiremos infinitamente
    celda_base = ase.Atoms(
            symbols   = simbolos, #elmento en la red
            positions = ti_positions, #posiciones de los átomos
            cell      = cell, #celda base que creé antes
            pbc       = [True, True, False]  # periódica en a y b, no en c
            )

    #Replica L x L veces en el plano con c intacta
    supercell = celda_base.repeat((L, L, 1))
    return supercell



def construir_libreta_vecinos_numba(supercell, d_anillos, J_meV, tol):
    """
    Recorre todos los pares de espines, calcula la distancia con MÍNIMA
    IMAGEN (respetando PBC; válido para celdas oblicuas porque trabaja en
    coordenadas fraccionarias), y conecta los pares cuya distancia coincida
    con d_1, d_2 o d_3 (dentro de 'tol'), asignándoles J_1, J_2 o J_3.

    Devuelve DOS matrices estáticas de tamaño (N x z_total):
      - neighbor_idx[i] : índices de los vecinos del sitio i
      - neighbor_J[i]   : el J (en meV) de cada uno de esos vecinos
    Implementación real: ase.neighborlist.neighbor_list('ijd', ...).
    """
    i_idx, j_idx, distancias = neighbor_list('ijd', supercell, cutoff=d_anillos[-1] + tol)
    # i es el indice del átomo, j es el índice del atomo vecino, d es la distancia entre ellos
    # supercell es la celda que construí anteriormente
    # cutoff: el corte en el que ASE va a dejar de buscar vecinos más cercanos, tiene cierta tolerancia
    N = len(supercell)

    #Para cada par, determinar a qué anillo pertenece y asignarle J
    pares_por_sitio = [[] for _ in range(N)]
    # loop en el cual conecto los pares para los cuales su distancia concuerde con la distancia d_i con su respectiva J
    for pos in range(len(i_idx)):
        i, j, d = i_idx[pos], j_idx[pos], distancias[pos] # desempaquetamos 
        for (d_k, J_k) in zip(d_anillos, J_meV): # empareja la distancia de los anillos con los J
            if abs(d - d_k) < tol: # condición para que no haya errores numéricos
                pares_por_sitio[i].append((j, J_k))
                break
    #Convertir a matrices numpy estáticas
    z_total = len(pares_por_sitio[0])
    neighbor_idx = np.zeros((N, z_total), dtype=int)
    neighbor_J   = np.zeros((N, z_total))

    #rellena las matrices
    for i, pares in enumerate(pares_por_sitio):
        for col, (j, J_k) in enumerate(pares):
            neighbor_idx[i, col] = j
            neighbor_J[i, col]   = J_k


    return neighbor_idx, neighbor_J



def verificar_coordinacion_numba(neighbor_idx, neighbor_J, J_meV):
    """
    Cuenta cuántos vecinos tiene cada sitio a cada distancia (z_1, z_2, z_3
    promedio). DEBEN coincidir con los que asumió el DFT al ajustar los J.
    Si no coinciden, hay un bug en la tolerancia o en la mínima imagen.
    Es la verificación de seguridad MÁS BARATA del flujo entero.
    Devuelve un array [z_1, z_2, z_3].
    """
    # ... cuenta vecinos agrupando por el valor de J ...

    N = neighbor_J.shape[0] # extraemos la dimensión de la matriz
    z_coordinaciones = np.zeros(len(J_meV)) #creamos la lista para guardar las coordinaciones

    for k, J_k in enumerate(J_meV):
        coincidencias = np.sum(neighbor_J == J_k) # Suma sobre una matriz de booleanos, cada True vale
                                                  # 1

        z_coordinaciones[k] = coincidencias / N



    return z_coordinaciones



def construir_patron_epsilon_numba(L, epsilon_base):
    """
    Replica los 6 signos del estado fundamental (epsilon_base) sobre toda la
    supercelda, generando un array de tamaño N = 6*L*L donde cada espín sabe
    cuál es su orientación "correcta" según el DFT.
    """
    return np.tile(epsilon_base, L * L)

#==========================================================================================




#================ALGORITMO DE METRÓPOLIS===================================================
@njit
def calcular_campo_local_numba(spins, neighbor_idx_i, neighbor_J_i):
    """
    Campo local sobre el sitio i:  h_i = sum_j  J_ij * S_j  (en meV).
    Es la "presión neta" que ejercen los vecinos sobre el sitio i.
    """
    return np.sum(neighbor_J_i * spins[neighbor_idx_i])

@njit
def metropolis_step_numba(spins, neighbor_idx, neighbor_J, T, k_B):
    """
    UN intento de voltear un espín al azar (campo externo h = 0).
    delta_E está en meV; T en K; k_B en meV/K -> el exponente del factor
    de Boltzmann es adimensional, como debe ser.
    """
    N = len(spins)
    i = np.random.randint(0, N)                  # elige un sitio i en 0..N-1
    h_i = calcular_campo_local(spins, neighbor_idx[i], neighbor_J[i])
    delta_E = 2.0 * spins[i] * h_i               # cambio de energía (meV)

    if delta_E <= 0.0:
        spins[i] = -spins[i]                     # baja energía -> aceptar siempre
    elif np.random.random() < np.exp(-delta_E / (k_B * T)):
        spins[i] = -spins[i]                     # sube energía -> aceptar con prob de Boltzmann
    # si no se cumple ninguna condición, NO se voltea (se rechaza el intento)

@njit
def metropolis_sweep_numba(spins, neighbor_idx, neighbor_J, T, k_B):
    """
    UN barrido = N intentos de volteo. En promedio, cada espín de la red
    recibe una oportunidad de voltearse durante un barrido.
    """
    N = len(spins)
    for _ in range(N):
        metropolis_step(spins, neighbor_idx, neighbor_J, T, k_B)

# ========================================================================================




# ============================ OBSERVABLES ================================================

def calcular_M_ord_number(spins, epsilon_array):
    """
    Parámetro de orden por sitio:  M_ord = (1/N) sum_i  epsilon_i * S_i.
    Cuenta (coincidencias con el patrón) menos (discrepancias), todo sobre N.
    Vale ~+-1 si el sistema está ordenado, ~0 si está desordenado.
    """
    return np.mean(epsilon_array * spins)

@njit
def calcular_energia_numba(spins, neighbor_idx, neighbor_J):
    """
    Energía total del sistema (en meV).
    El factor 0.5 corrige el DOBLE CONTEO: cada par (i,j) aparece tanto en
    la libreta de i como en la de j, así que sin el 0.5 contaríamos cada
    interacción dos veces.
    """
    N = len(spins)
    suma = 0.0
    for i in range(N):
        h_i = calcular_campo_local(spins, neighbor_idx[i], neighbor_J[i])
        suma += spins[i] * h_i
    return -0.5 * suma

#==========================================================================================



#============================ BARRIDO DE TEMPERATURA PARA UN TAMANO L======================
def correr_simulacion_numba(L, cell, ti_positions, d_anillos, J_meV, tol,
                      epsilon_base, T_min, T_max, N_temps,
                      N_term, N_skip, N_meas, k_B):
    """Corre el barrido completo de temperaturas para un tamaño L."""

    # 1) Construir red, libreta de vecinos y patrón epsilon.
    supercell      = construir_supercelda_numba(L, cell, ti_positions)
    nbr_idx, nbr_J = construir_libreta_vecinos_numba(supercell, d_anillos, J_meV, tol)
    epsilon        = construir_patron_epsilon_numba(L, epsilon_base)

    # 2) Verificación de seguridad: z_k debe coincidir con el DFT.
    z_coord = verificar_coordinacion_numba(nbr_idx, nbr_J, J_meV)

    # 3) Inicializar espines al azar (+1 o -1 con probabilidad 50/50).
    N = 6 * L * L
    spins = np.random.choice([-1.0, +1.0], size=N)

    # 4) Temperaturas de MAYOR a MENOR (annealing).
    temperaturas = np.linspace(T_max, T_min, N_temps)

    # 5) Barrido en temperatura.
    resultados = {}
    for T in temperaturas:

        # 5a) Termalización: estos barridos NO se miden, se descartan.
        for _ in range(N_term):
            metropolis_sweep_numba(spins, nbr_idx, nbr_J, T, k_B)

        # 5b) Producción: medimos cada N_skip barridos.
        M_abs_acum = 0.0
        M2_acum    = 0.0
        M4_acum    = 0.0
        E_acum     = 0.0
        E2_acum    = 0.0
        for _ in range(N_meas):
            for _ in range(N_skip):
                metropolis_sweep_numba(spins, nbr_idx, nbr_J, T, k_B)
            M = calcular_M_ord_numba(spins, epsilon)
            E = calcular_energia_numba(spins, nbr_idx, nbr_J)
            M_abs_acum += abs(M)
            M2_acum    += M * M
            M4_acum    += M**4
            E_acum     += E
            E2_acum    += E * E

        # 5c) Guardar promedios para esta temperatura.
        resultados[T] = {
                "M_abs": M_abs_acum / N_meas,
                "M2":    M2_acum    / N_meas,
                "M4":    M4_acum    / N_meas,
                "E":     E_acum     / N_meas,
                "E2":    E2_acum    / N_meas,
                "N":     N,
                }

    return resultados, z_coord

#==========================================================================================




#===========================CUMULANTE, CANTIDADES FÍSICAS Y T_C============================

def calcular_cumulante_numba(M2, M4):
    """Cumulante de Binder:  U_L = 1 - <M^4> / (3 <M^2>^2). Adimensional."""
    return 1.0 - M4 / (3.0 * M2**2)

def calcular_susceptibilidad_numba(M2, M_abs, T, N, k_B):
    """
    Susceptibilidad por sitio (unidades clásicas, con k_B explícito):
        chi = (N / (k_B T)) * (<M^2> - <|M|>^2)
    Se usa <|M|> (valor absoluto) y no <M> porque en fase ordenada el
    sistema salta entre +M_0 y -M_0 y <M> daría cero espuriamente.
    """
    return (N / (k_B * T)) * (M2 - M_abs**2)

def calcular_calor_especifico_numba(E, E2, T, N, k_B):
    """
    Calor específico por sitio (unidades clásicas, con k_B explícito):
        C = (<E^2> - <E>^2) / (N * k_B * T^2)
    """
    return (E2 - E**2) / (N * k_B * T**2)

def encontrar_cruce_numba(curva_L1, curva_L2):
    """
    Dadas las curvas U_L(T) de dos tamaños (diccionarios T -> U), interpola
    ambas con splines y resuelve U_L1(T) = U_L2(T). Devuelve la temperatura
    de cruce (en Kelvin). Esa temperatura es una estimación de T_c todavía
    afectada por tamaño finito.
    Implementación real: scipy.interpolate.CubicSpline + scipy.optimize.brentq.
    """
    # ... interpolación spline + búsqueda de raíz de la diferencia ...
    temps = sorted(curva_L1.keys())

    U_L1 = [curva_L1[T] for T in temps]
    U_L2 = [curva_L2[T] for T in temps]

    print(f"  U_L1: {U_L1}")
    print(f"  U_L2: {U_L2}")


    spline_L1 = CubicSpline(temps, U_L1)
    spline_L2 = CubicSpline(temps, U_L2)

    diferencia = lambda T: spline_L1(T) - spline_L2(T)

    # buscar la raíz en el intervalo [T_min, T_max]
    T_cruce = brentq(diferencia, temps[0], temps[-1])

    return T_cruce

def extrapolar_a_infinito_numba(temperaturas_de_cruce, tamanos):
    """
    Ajusta  T_cruce(L) = T_c(infinito) + a/L  (una recta en 1/L) y devuelve
    la ordenada al origen: T_c del sistema infinito, EN KELVIN.
    Como trabajamos en unidades clásicas, este resultado YA está en Kelvin
    (no hace falta ninguna conversión adicional).
    Implementación real: numpy.polyfit(1/L, T_cruce, deg=1).
    """
    # ... ajuste lineal de T_cruce vs (1/L) ...

    inversos = 1.0 / np.array(tamanos)
    coefs = np.polyfit(inversos, temperaturas_de_cruce, deg=1)

    # la ordenada al origen es T_c del sistema infinito
    T_c_infinito = coefs[1]
    return T_c_infinito

