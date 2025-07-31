import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from statistics import mean
from scipy.sparse import hstack

def warn(*args, **kwargs):
    """
    Función que anula las advertencias, simplemente no realiza ninguna acción.

    Parámetros:
        *args, **kwargs: Argumentos y palabras clave (keywords) pasados a la función. No se utilizan en esta implementación.
    """
    pass

warnings.warn = warn

#~ Bloque 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Función para cargar los datos de bancarrota
def load_bankruptcy_data():
    """
    Carga los datos de bancarrota desde un archivo y devuelve las características (X) y las etiquetas (y).
    """
    df = pd.read_csv("bankruptcy_data.txt", sep="\t")
    df = df.sample(frac=1, random_state=1234)
    X = df.loc[:, df.columns != "class"]
    y = df["class"]
    return X, y

# Función para cargar los datos bancarios
def load_bank_data():
    """
    Carga los datos bancarios desde un archivo, divide los datos en conjuntos de entrenamiento y validación,
    y devuelve X_train, X_val, y_train y y_val.
    """
    df = pd.read_csv("bank-full.csv", sep=";")
    df = df.sample(frac=1, random_state=1234)
    X = df.loc[:, df.columns != "y"]
    y = df["y"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3456)
    return X_train, X_val, y_train, y_val

# Cargamos los datos de bancarrota
X, y = load_bankruptcy_data()

# Imprimimos el conteo de las etiquetas
print(y.value_counts())

# Experimento con KNN sin escalado ni transformaciones
exp_knn = []
for k in tqdm([1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300]):
    kcv = KFold(n_splits=10, shuffle=True, random_state=2345)
    knn = KNeighborsClassifier(n_neighbors=k)
    kcv_score = mean(cross_val_score(knn, X, y, scoring="roc_auc", cv=kcv, n_jobs=-1))
    exp_knn.append({"k": k, "kcv_score": kcv_score})

exp_knn = pd.DataFrame(exp_knn)
print(exp_knn)
print(exp_knn.iloc[exp_knn["kcv_score"].idxmax(),:])

# Estandarizando los datos
exp_knn_scaled = []
for k in tqdm([1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300]):
    kcv = KFold(n_splits=10, shuffle=True, random_state=2345)
    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k))
    kcv_score = mean(cross_val_score(knn, X, y, scoring="roc_auc", cv=kcv, n_jobs=-1))
    exp_knn_scaled.append({"k": k, "kcv_score": kcv_score})

exp_knn_scaled = pd.DataFrame(exp_knn_scaled)
print(exp_knn_scaled)
print(exp_knn_scaled.iloc[exp_knn_scaled["kcv_score"].idxmax(),:])

# Tomando logaritmos
class myLogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado que aplica el logaritmo a los datos después de asegurarse de que sean positivos.
    """
    def __init__(self):
        self.mins = {}

    def fit(self, X, y=None):
        """
        Calcula el valor mínimo de cada columna para suavizar el logaritmo.
        """
        for c in X.columns:
            self.mins[c] = min(X[c])
        return self

    def transform(self, X, y=None):
        """
        Aplica el logaritmo a las columnas, asegurándose de que los valores sean positivos.
        """
        X_out = pd.DataFrame()
        for c in X.columns:
            min_c = self.mins[c]
            X.loc[X[c] < min_c, c] = min_c
            X_out[c] = np.log10(X[c] + 1 - min_c)
        return X_out

# Experimento con KNN y logaritmos
exp_knn_log = []
for k in tqdm([1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300]):
    kcv = KFold(n_splits=10, shuffle=True, random_state=2345)
    knn = make_pipeline(myLogTransformer(),
                        KNeighborsClassifier(n_neighbors=k))
    kcv_score = mean(cross_val_score(knn, X, y, scoring="roc_auc", cv=kcv, n_jobs=-1))
    exp_knn_log.append({"k": k, "kcv_score": kcv_score})

exp_knn_log = pd.DataFrame(exp_knn_log)
print(exp_knn_log)
print(exp_knn_log.iloc[exp_knn_log["kcv_score"].idxmax(),:])


# Experimento con KNN y logaritmos escalados
exp_knn_log_and_scaled = []
for k in tqdm([1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300]):
    kcv = KFold(n_splits=10, shuffle=True, random_state=2345)
    knn = make_pipeline(myLogTransformer(),
                        StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k))
    kcv_score = mean(cross_val_score(knn, X, y, scoring="roc_auc", cv=kcv, n_jobs=-1))
    exp_knn_log_and_scaled.append({"k": k, "kcv_score": kcv_score})

exp_knn_log_and_scaled = pd.DataFrame(exp_knn_log_and_scaled)
print(exp_knn_log_and_scaled)
print(exp_knn_log_and_scaled.iloc[exp_knn_log_and_scaled["kcv_score"].idxmax(),:])

#~ Bloque 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Experimento con Bayes ingenuo en datos sin procesar
kcv = KFold(n_splits=10, shuffle=True, random_state=2345)
nb = GaussianNB()
kcv_nb = mean(cross_val_score(nb, X, y, scoring="roc_auc", cv=kcv, n_jobs=-1))
print(kcv_nb)

# Experimento con Bayes ingenuo y discretización de variables
exp_nb_bins = []
for n_bins in tqdm([16, 18, 20, 22, 24, 26, 28]):
    np.seterr(divide='ignore')
    kcv = KFold(n_splits=10, shuffle=True, random_state=2345)
    nb = make_pipeline(KBinsDiscretizer(n_bins=n_bins, encode="ordinal"),
                       CategoricalNB(alpha=0, force_alpha=True))
    kcv_score = mean(cross_val_score(nb, X, y, scoring="roc_auc", cv=kcv, n_jobs=None))
    exp_nb_bins.append({"n_bins": n_bins, "kcv_score": kcv_score})
    np.seterr(divide='warn')

exp_nb_bins = pd.DataFrame(exp_nb_bins)
print(exp_nb_bins)
print(exp_nb_bins.iloc[exp_nb_bins["kcv_score"].idxmax(),:])

#~ Bloque 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Cargamos los datos bancarios
X_train, X_val, y_train, y_val = load_bank_data()

# Imprimimos el conteo de las etiquetas de train
print(y_train.value_counts())

# One-hot-encoding
# Opción 1: utilizar get_dummies de pandas
X_train_ohe_v1 = pd.get_dummies(X_train).head()  # Y ahora cómo lo aplico en validación? (debería unir antes los datos)

# Opción 2: utilizar onehotencoder de sklearn (más conveniente para producción)

# Sobre training
X_train_obj = X_train.select_dtypes(include=["object"])
X_train_num = X_train.select_dtypes(include=["int", "float"])

scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=True)
X_train_ohe_obj = encoder.fit_transform(X_train_obj)
X_train_ohe_v2 = hstack([X_train_ohe_obj, scaler.fit_transform(X_train_num)])

# Sobre validation
X_val_obj = X_val.select_dtypes(include=["object"])
X_val_num = X_val.select_dtypes(include=["int", "float"])

X_val_ohe_obj = encoder.transform(X_val_obj)
X_val_ohe_v2 = hstack([X_val_ohe_obj, scaler.transform(X_val_num)])

# Entrenamos un modelo de regresión logística sobre los datos con OHE sin escalado
lr_exp = []
for C in tqdm([0.001, 0.0025, 0.01, 0.05, 0.075, 0.1, 0.25]):
    lr = LogisticRegression(C=C, max_iter=2000)
    lr.fit(X_train_ohe_v2, y_train)
    preds = lr.predict_proba(X_val_ohe_v2)
    preds = preds[:, lr.classes_ == "yes"]
    auc = roc_auc_score(y_val == "yes", preds)
    lr_exp.append({"C": C, "auc_val": auc})

lr_exp = pd.DataFrame(lr_exp)
print(lr_exp)
print(lr_exp.iloc[lr_exp["auc_val"].idxmax(),:])

# Regresión logística con interacciones
lr_exp_int = []
for C in tqdm([0.001, 0.0025, 0.01, 0.05, 0.075, 0.1, 0.25]):
    lr = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                       LogisticRegression(C=C, max_iter=2000))
    lr.fit(X_train_ohe_v2, y_train)
    preds = lr.predict_proba(X_val_ohe_v2)
    preds = preds[:, lr.classes_ == "yes"]
    auc = roc_auc_score(y_val == "yes", preds)
    lr_exp_int.append({"C": C, "auc_val": auc})

lr_exp_int = pd.DataFrame(lr_exp_int)
print(lr_exp_int)
print(lr_exp_int.iloc[lr_exp_int["auc_val"].idxmax(),:])
