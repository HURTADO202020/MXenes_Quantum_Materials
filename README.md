# Simulación Monte Carlo para MXene Ti₆

Simulación de Monte Carlo para predecir la temperatura crítica *T*c a la cual el material 2D MXene Ti₆ pierde su orden magnético. El modelo utiliza el Hamiltoniano de Ising con acoplamientos de canje *J*₁, *J*₂, *J*₃ extraídos de cálculos DFT, y aplica el método del cumulante de Binder con escalamiento de tamaño finito para extrapolar *T*c al límite termodinámico.

## Estructura del proyecto
├── run.py # Punto de entrada — corre la simulación completa
├── funciones.py # Implementación de todos los algoritmos
├── parametros.py # Parámetros de simulación y datos del material
└── Benchmark.ipynb # Análisis comparativo de rendimiento (Python vs Numba)
## Método

- **Modelo**: Ising 2D con primeros, segundos y terceros vecinos
- **Algoritmo**: Metrópolis con annealing en temperatura
- **Análisis**: Cumulante de Binder, susceptibilidad y calor específico
- **Extrapolación**: Escalamiento de tamaño finito (1/L → 0)

## Uso

```bash
python run.py
```

Los parámetros de simulación (tamaños de supercelda, rango de temperaturas, estadística) se configuran en `parametros.py`.

## Dependencias

- Python 3.12
- NumPy
- SciPy
- ASE (Atomic Simulation Environment)
- Matplotlib
