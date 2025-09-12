import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

def warn(*args, **kwargs):
    """
    Función que anula las advertencias, simplemente no realiza ninguna acción.

    Parámetros:
        *args, **kwargs: Argumentos y palabras clave (keywords) pasados a la función. No se utilizan en esta implementación.
    """
    pass

warnings.warn = warn


# Función para cargar los datos de bancarrota
def load_bankruptcy_data():
    """
    Carga los datos de bancarrota desde un archivo y devuelve las características (X) y las etiquetas (y).
    """
    df = pd.read_csv("bankruptcy_data.txt", sep="\t")
    df = df.sample(frac=1, random_state=1234)
    X_train = df.loc[:, df.columns != "class"]
    y_train = df["class"]
    return X_train, y_train

# Carga de datos
X_train, y_train = load_bankruptcy_data()

# División en conjuntos de entrenamiento, validación y prueba
size_val = math.ceil(0.1 * X_train.shape[0])
size_test = math.ceil(0.1 * X_train.shape[0])

X_train_red, X_test, y_train_red, y_test = train_test_split(X_train, y_train, test_size=size_test, random_state=1234)
X_train_red, X_val, y_train_red, y_val = train_test_split(X_train_red, y_train_red, test_size=size_val, random_state=2345)

## Experimento con Bayes ingenuo
np.seterr(divide='ignore')
nb = make_pipeline(KBinsDiscretizer(n_bins=25, encode="ordinal"),
                   CategoricalNB(alpha=0, force_alpha=True))
nb.fit(pd.concat([X_train_red, X_val], axis=0), pd.concat([y_train_red, y_val], axis=0))
preds_probs_nb = nb.predict_proba(X_test)[:, 1]
print("AUC test score - Naïve Bayes:", roc_auc_score(y_test, preds_probs_nb))  # 0.8449174522420071
np.seterr(divide='warn')

## Experimento con vecinos más cercanos
exp_knn_scaled = []
for k in tqdm(range(10, 150, 5)):
    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k, weights="uniform"))
    knn.fit(pd.concat([X_train_red, X_val], axis=0), pd.concat([y_train_red, y_val], axis=0))
    preds_probs_knn = knn.predict_proba(X_test)[:, 1]
    exp_knn_scaled.append({"k": k, "auc_score": roc_auc_score(y_test, preds_probs_knn)})

exp_knn_scaled = pd.DataFrame(exp_knn_scaled)
print(exp_knn_scaled)
