import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import math
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

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

## Experimento con regresión logística
best_auc_lr = float("-inf")
best_c_lr = None
for c in tqdm([0.001, 0.01, 0.1, 1, 5, 10]):
    lr = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                       StandardScaler(),
                       LogisticRegression(penalty="l2", C=c, max_iter=3000))
    lr.fit(X_train_red, y_train_red)
    preds_val_lr = lr.predict_proba(X_val)[:, lr.classes_ == True]
    tmp_auc = roc_auc_score(y_val, preds_val_lr)
    if tmp_auc > best_auc_lr:
        best_auc_lr = tmp_auc
        best_c_lr = c

lr = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                    StandardScaler(),
                    LogisticRegression(penalty="l2", C=best_c_lr, max_iter=3000))
lr.fit(pd.concat([X_train_red, X_val], axis=0), pd.concat([y_train_red, y_val], axis=0))
preds_probs_lr = lr.predict_proba(X_test)[:, 1]
print("AUC test score - Logistic regression:", roc_auc_score(y_test, preds_probs_lr))  # 0.9635143897451244

## Experimento con SVMs
best_auc = float("-inf")
best_c = None
for c in tqdm([0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    svm = make_pipeline(StandardScaler(), SVC(C=c, probability=True, kernel="poly"))
    svm.fit(X_train_red, y_train_red)
    preds_val_svm = svm.predict_proba(X_val)[:, svm.classes_ == True]
    tmp_auc = roc_auc_score(y_val, preds_val_svm)
    if tmp_auc > best_auc:
        best_auc = tmp_auc
        best_c = c

best_svm = make_pipeline(StandardScaler(), SVC(C=best_c, probability=True, degree=2))
best_svm.fit(pd.concat([X_train_red, X_val], axis=0), pd.concat([y_train_red, y_val], axis=0))
preds_test_best_svm = best_svm.predict_proba(X_test)[:, best_svm.classes_ == True]
print("AUC test score - SVM:", roc_auc_score(y_test, preds_test_best_svm)) # 0.9265798792957255
