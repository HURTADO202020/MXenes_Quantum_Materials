import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.neighborlist import neighbor_list
from parametros import cell, ti_positions
from funciones import construir_supercelda

sc = construir_supercelda(6, cell, ti_positions)
_, _, dist = neighbor_list('ijd', sc, cutoff=7.0)
plt.hist(dist, bins=100); plt.xlabel('Distancia (Å)'); plt.ylabel('Conteo')
plt.savefig('resultados/histograma_distancias.png', dpi=150, bbox_inches='tight')

