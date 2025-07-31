import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import math
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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

# Entrenamiento y evaluación del modelo Random Forest
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=6789, verbose=1, oob_score=True)
rf.fit(pd.concat([X_train_red, X_val], axis=0),
       pd.concat([y_train_red, y_val], axis=0))
preds_test_rf = rf.predict_proba(X_test)[:, rf.classes_ == True]
print("ROC test score - Random Forest:", roc_auc_score(y_test, preds_test_rf)) # 0.9511696343957305

# Entrenamiento y evaluación del modelo XGBoost
xgb_params = {'colsample_bytree': 0.75,
              'gamma': 0.5,
              'learning_rate': 0.075,
              'max_depth': 8,
              'min_child_weight': 1,
              'n_estimators': 1200,
              'reg_lambda': 0.5,
              'subsample': 0.75,
              }

clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
                            seed = 1234,
                            eval_metric = 'auc',
                            **xgb_params)

clf_xgb.fit(X_train_red, y_train_red, verbose = 100, eval_set = [(X_train_red, y_train_red), (X_val, y_val)])

preds_test_xgb = clf_xgb.predict_proba(X_test)[:, clf_xgb.classes_ == True]
print("AUC test score - XGBoost:", roc_auc_score(y_test, preds_test_xgb)) # 0.9719437378422864
