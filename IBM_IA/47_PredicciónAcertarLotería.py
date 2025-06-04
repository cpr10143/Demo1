import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class GeneradorSeries:
    @staticmethod
    def generar_series(cantidad):
        series = [np.sort(np.random.choice(range(1, 50), size=6, replace=False)) for _ in range(cantidad)]
        return np.array(series)


class DatosLoteria:
    @staticmethod
    def generar_datos_entrenamiento(cantidad=1000):
        series = GeneradorSeries.generar_series(cantidad)
        exito = [1 if np.random.rand() < 0.1 else 0 for _ in range(cantidad)]
        df = pd.DataFrame(series, columns=[f'num{i+1}' for i in range(6)])
        df['Exito'] = exito
        return df


class ModeloLoteria:
    def __init__(self):
        self.modelo = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()

    def entrenar(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)

    def predecir_probabilidades(self, X):
        X_scaled = self.scaler.transform(X)
        return self.modelo.predict_proba(X_scaled)[:, 1]


class VisualizadorResultados:
    @staticmethod
    def graficar_top_combinaciones(df_series, probabilidades, top_n=10):
        df_resultados = df_series.copy()
        df_resultados['Probabilidad'] = probabilidades
        top_combinaciones = df_resultados.sort_values(by='Probabilidad', ascending=False).head(top_n)
        etiquetas = top_combinaciones[[f'num{i+1}' for i in range(6)]].astype(str).agg('-'.join, axis=1)

        plt.figure(figsize=(10, 6))
        plt.barh(etiquetas[::-1], top_combinaciones['Probabilidad'][::-1], color='skyblue')
        plt.xlabel('Probabilidad de Éxito')
        plt.title(f'Top {top_n} Combinaciones con Mayor Probabilidad')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


class EjecutarSimulacion:
    def ejecutar(self):
        # Generar datos simulados
        datos = DatosLoteria.generar_datos_entrenamiento()
        X = datos[[f'num{i+1}' for i in range(6)]]
        y = datos['Exito']

        # Entrenar el modelo
        modelo = ModeloLoteria()
        modelo.entrenar(X, y)

        # Generar nuevas combinaciones a evaluar
        nuevas_series = GeneradorSeries.generar_series(100)
        df_nuevas_series = pd.DataFrame(nuevas_series, columns=[f'num{i+1}' for i in range(6)])

        # Predecir probabilidades
        probabilidades = modelo.predecir_probabilidades(df_nuevas_series)

        # Mostrar mejor combinación
        mejor_idx = np.argmax(probabilidades)
        mejor_combinacion = nuevas_series[mejor_idx]
        mejor_prob = probabilidades[mejor_idx]

        print("🎯 Mejor serie encontrada:")
        print("Números:", list(mejor_combinacion))
        print(f"Probabilidad estimada de éxito: {mejor_prob:.4f}")

        # Mostrar gráfica con top 10 combinaciones
        VisualizadorResultados.graficar_top_combinaciones(df_nuevas_series, probabilidades)


# Ejecución principal
if __name__ == "__main__":
    simulacion = EjecutarSimulacion()
    simulacion.ejecutar()
