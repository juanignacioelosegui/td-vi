#   Importamos Iris
from sklearn.datasets import load_iris
iris = load_iris()
print("Cargado Iris.")

#   División train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris.data,          # X
        iris.target,        # y
        test_size = 0.25,
        random_state = 0,
        stratify = iris.target
    )
print("Dividido el dataset")

#   Pipeline: estandarizar + MLP
from sklearn.preprocessing import StandardScaler
normalizador    = StandardScaler()
X_train_scaled  = normalizador.fit_transform(X_train)
X_test_scaled   = normalizador.transform(X_test)
print("Dataset normalizado")

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
        hidden_layer_sizes  =   (10,),  # 10 neuronas en capa oculta
        activation          =   "relu",
        solver              =   "adam",
        max_iter            =   1000,
        random_state        =   0
    )

#   Train
mlp.fit(X_train_scaled, y_train)
print("Entrenamiento listo")

#   Evaluar
y_pred = mlp.predict(X_test_scaled)
print("Predicciones listas")

#   Resultados
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
