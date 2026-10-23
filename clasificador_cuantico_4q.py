import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

DEVICE_NAME = "lightning.qubit"
NQ = 4
CAPAS = 4

# 1. Cargar datos (copia local del conjunto UCI id=267, sin acceso a archive.ics.uci.edu)
datos = pd.read_csv("banknote.csv", header=None,
                    names=["variance", "skewness", "curtosis", "entropy", "class"])
X = datos[["variance", "skewness", "curtosis", "entropy"]].to_numpy(dtype=float)
y = datos["class"].to_numpy(dtype=int)

print("Atributos usados: variance, skewness, curtosis y entropy")
print("Cantidad de ejemplos:", len(y))
print("Clases encontradas:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y)

if len(X_train) > 320:
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=320, random_state=7, stratify=y_train)

escalador = MinMaxScaler(feature_range=(0, np.pi))
X_train = escalador.fit_transform(X_train)
X_test = escalador.transform(X_test)

y_train_q = pnp.array(2 * y_train - 1, requires_grad=False)

dispositivo = qml.device(DEVICE_NAME, wires=NQ)

@qml.qnode(dispositivo, interface="autograd", diff_method="adjoint")
def clasificador_cuantico(atributos, pesos):
    qml.AngleEmbedding(atributos, wires=range(NQ), rotation="Y")
    for capa in range(pesos.shape[0]):
        for qubit in range(NQ):
            qml.RY(pesos[capa, qubit, 0], wires=qubit)
            qml.RZ(pesos[capa, qubit, 1], wires=qubit)
        for q in range(NQ):
            qml.CNOT(wires=[q, (q + 1) % NQ])
    return qml.expval(qml.PauliZ(0))

def calcular_error(pesos, atributos_lote, clases_lote):
    resultados = pnp.stack([clasificador_cuantico(x, pesos) for x in atributos_lote])
    return pnp.mean((resultados - clases_lote) ** 2)

pnp.random.seed(7)
pesos = pnp.array(0.1 * pnp.random.randn(CAPAS, NQ, 2), requires_grad=True)
optimizador = qml.GradientDescentOptimizer(stepsize=0.08)

epocas = 12
tamano_lote = 32

for epoca in range(epocas):
    indices = np.random.permutation(len(X_train))
    for inicio in range(0, len(X_train), tamano_lote):
        lote = indices[inicio:inicio + tamano_lote]
        pesos, error = optimizador.step_and_cost(
            lambda w: calcular_error(
                w, pnp.array(X_train[lote], requires_grad=False), y_train_q[lote]),
            pesos)
    print(f"Época {epoca + 1:02d} | error: {float(error):.4f}")

puntajes = np.array([
    float(clasificador_cuantico(pnp.array(x, requires_grad=False), pesos))
    for x in X_test])
predicciones = (puntajes > 0).astype(int)

print("\nExactitud:", accuracy_score(y_test, predicciones))
print("Matriz de confusión (filas = clase real; columnas = predicción):")
print(confusion_matrix(y_test, predicciones))

def clasificar_billete(varianza, asimetria, curtosis, entropia):
    nuevo_escalado = escalador.transform(np.array([[varianza, asimetria, curtosis, entropia]]))[0]
    puntaje = float(clasificador_cuantico(pnp.array(nuevo_escalado, requires_grad=False), pesos))
    return ("FALSO" if puntaje > 0 else "VERDADERO"), puntaje
