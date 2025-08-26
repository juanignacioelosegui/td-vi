from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Cargamos el dataset Iris
iris = load_iris()

# Dividimos el dataset en:
# - train+validation (75%)
# - test (25%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

# Luego dividimos train+validation en:
# - entrenamiento (75% de 75%)
# - validación (25% de 75%)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1
)

print(
    "Tamaño de training set: {} | "
    "Tamaño de validation set: {} | "
    "Tamaño de test set: {}\n".format(
        X_train.shape[0], X_valid.shape[0], X_test.shape[0]
    )
)

# Inicializamos la mejor puntuación
best_score = 0

# Recorremos combinaciones de parámetros gamma y C
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # Entrenamos un modelo SVM con los parámetros actuales
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)

        # Evaluamos el modelo en el set de validación
        score = svm.score(X_valid, y_valid)

        # Si este modelo es mejor que los anteriores, lo guardamos
        if score > best_score:
            best_score = score
            best_parameters = {"C": C, "gamma": gamma}

# Con los mejores parámetros encontrados, entrenamos nuevamente
# usando TODO el set de train+validation, y probamos en el test
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)

print("Mejor score en validación: {:.2f}".format(best_score))
print("Mejores parámetros encontrados: ", best_parameters)
print("Score en test con los mejores parámetros: {:.2f}".format(test_score))