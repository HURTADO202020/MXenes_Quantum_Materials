import numpy as np
#===================== PARÁMETROS DE SIMULACIÓN====================================

# tamanos de celada a simular
L_list = [6, 12]

# Rangos de temepratura del barrido
T_min   = 100.0     # K
T_max   = 450    # K
N_temps = 10       # cuántas temperaturas simular entre T_min y T_max

# termalización
N_term = 2000

# barridos saltados entre medidas consecutivas
N_skip = 5

# medidas efectivas en cada temperatura
N_meas = 2000

#reproducibilidad
seed = 42

#carpeta de resultados 
output_dir = "./resultados"



#===================DATOS DEL MATERIAL ============================================

# constante de Boltzman
k_B = 0.0861733   # meV/K

# Vectores de red de la celda unidad, las primeras filas generan el plano y
#la tercera es la dirección de vacío
cell = np.array([[12.5591, -0.0000,  0.000 ],
                 [-3.1397,  5.4382,  0.000 ],
                 [ 0.0000,  0.0000, 23.928 ]])

#Posiciones de los 6 átomos de Ti dentro de la celda unidad
ti_positions = np.array([[ 1.1238, 2.4778, 15.81],
                         [-0.4358, 5.1792, 15.81],
                         [ 8.9631, 5.1792, 15.81],
                         [ 2.6835, 5.1792, 15.81],
                         [ 7.4034, 2.4778, 15.81],
                         [ 5.8437, 5.1792, 15.81]])


# constante de acoplamiento, orden: [J_1, J_2, J_3]
J_meV = np.array([12.0, -3.5, 1.2])

# distancias a las que viven los acoplamientos verificar con el histograma de distancias de
#su red
d_anillos = np.array([3.12, 5.10, 6.28])

#tolerancia
tol = 0.05

#Patrón del estado fundamental
epsilon_base = np.array([+1, -1, +1, -1, +1, -1])

