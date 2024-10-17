import numpy as np
import matplotlib.pyplot as plt


# Definir la función de Himmelblau
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


# Parámetros del recocido simulado
def recocido_simulado(func, limites, max_iter=10000, temp=1000, tasa_enfriamiento=0.99):
    # Inicialización de valores aleatorios dentro de los límites
    x_actual = np.random.uniform(limites[0], limites[1])
    y_actual = np.random.uniform(limites[0], limites[1])
    mejor_solucion = (x_actual, y_actual)
    mejor_valor = func(x_actual, y_actual)

    for i in range(max_iter):
        # Generar nuevos candidatos en el vecindario
        x_nuevo = x_actual + np.random.uniform(-0.1, 0.1)
        y_nuevo = y_actual + np.random.uniform(-0.1, 0.1)

        # Mantener los nuevos valores dentro de los límites
        x_nuevo = np.clip(x_nuevo, limites[0], limites[1])
        y_nuevo = np.clip(y_nuevo, limites[0], limites[1])

        # Evaluar la función para los nuevos valores
        valor_actual = func(x_actual, y_actual)
        valor_nuevo = func(x_nuevo, y_nuevo)

        # Si la nueva solución es mejor, se acepta
        if valor_nuevo < valor_actual:
            x_actual, y_actual = x_nuevo, y_nuevo
        else:
            # Aceptar soluciones peores con una probabilidad que depende de la temperatura
            prob_aceptacion = np.exp((valor_actual - valor_nuevo) / temp)
            if np.random.rand() < prob_aceptacion:
                x_actual, y_actual = x_nuevo, y_nuevo

        # Actualizar la mejor solución encontrada
        if valor_nuevo < mejor_valor:
            mejor_solucion = (x_nuevo, y_nuevo)
            mejor_valor = valor_nuevo

        # Reducir la temperatura
        temp *= tasa_enfriamiento

    return mejor_solucion, mejor_valor


# Ejecutar el algoritmo
limites = (-5, 5)
mejor_solucion, mejor_valor = recocido_simulado(himmelblau, limites)

print("Mejor solución (x, y):", mejor_solucion)
print("Valor mínimo de la función:", mejor_valor)

# Graficar la función de Himmelblau y la solución encontrada
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)

plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
plt.plot(mejor_solucion[0], mejor_solucion[1], 'r*', markersize=10)
plt.title("Función de Himmelblau y solución encontrada")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
