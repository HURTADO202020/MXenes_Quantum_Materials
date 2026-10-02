import numpy as np
#===================== PARÁMETROS DE SIMULACIÓN====================================

# tamanos de celada a simular
L_list = [1, 2, 3]

# Rangos de temperatura del barrido
T_min   = 50.0     # K
T_max   = 700.0    # K
N_temps = 30       # cuántas temperaturas simular entre T_min y T_max

# termalización
N_term = 20_000

# barridos saltados entre medidas consecutivas
N_skip = 50

# medidas efectivas en cada temperatura
N_meas = 50_000

#reproducibilidad
seed = 42

#carpeta de resultados 
output_dir = "./resultados"


parallel = False
#===================DATOS DEL MATERIAL ============================================

# constante de Boltzman
k_B = 0.0861733   # meV/K

# Vectores de red de la celda unidad, las primeras filas generan el plano y
#la tercera es la dirección de vacío
cell = np.array([[  12.550147879, -0.003867819,  0.0 ],
                 [ -3.139216741 ,  5.437650478, 0.0  ],
                 [  0.0         ,  0.0        , 28.0 ]])

#Posiciones de los 6 átomos de Ti dentro de la celda unidad
ti_positions = np.array([[  2.57663215,       5.11698331,      15.84977672],
                         [  1.14536426,       2.48228168,      15.99956434],
                         [ -0.42019610,       5.19598340,      15.99791184],
                         [  8.85170649,       5.11504954,      15.84977884],
                         [  7.42043749,       2.48034723,      15.99956474],
                         [  5.85487696,       5.19404942,      15.99791273]])


# constante de acoplamiento, orden: [J_1, J_2, J_3]
J_meV = np.array([0.1679794229567027, -43.74021239206194, -0.16694198548793793])

# distancias a las que viven los acoplamientos verificar con el histograma de distancias de
#su red
d_anillos = np.array([3.00, 3.13, 3.28])

#tolerancia
tol = 0.05

#Patrón del estado fundamental
epsilon_base = np.array([+1, -1, +1, -1, +1, -1])

