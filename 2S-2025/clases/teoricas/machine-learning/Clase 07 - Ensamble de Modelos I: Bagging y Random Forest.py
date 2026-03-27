import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK
import math
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

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

N_TREES = 500

# Definición de la función objetivo para Hyperopt
def objective(params):
    tree = DecisionTreeClassifier(**params, random_state=2345)
    tree.fit(X_train_red, y_train_red)
    y_preds_val_prob = tree.predict_proba(X_val)[:, tree.classes_ == True]
    score = roc_auc_score(y_val, y_preds_val_prob)
    return {'loss': -1 * score, 'status': STATUS_OK}

# Espacio de búsqueda para los hiperparámetros
space = {'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
         'splitter': hp.choice('splitter', ['best', 'random']),
         'max_depth': hp.uniformint('max_depth', 3, 30),
         'min_samples_split': hp.uniformint('min_samples_split', 2, 20),
         'min_samples_leaf': hp.uniformint('min_samples_leaf', 1, 20),
         'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.1)}

# Búsqueda de hiperparámetros "óptimos" con Hyperopt
best = fmin(objective, space,
            algo=tpe.suggest,
            max_evals=N_TREES,
            rstate=np.random.default_rng(3456)) # best loss: -0.8834860828241683
best_tree_params = space_eval(space, best)

# Creación y entrenamiento del mejor árbol
best_tree = DecisionTreeClassifier(**best_tree_params, random_state=4567)
best_tree.fit(pd.concat([X_train_red, X_val], axis=0),
              pd.concat([y_train_red, y_val], axis=0))

# Predicción y evaluación del mejor árbol en el conjunto de test
preds_test_tree = best_tree.predict_proba(X_test)[:, best_tree.classes_ == True]
print("ROC AUC Score - Best Tree:", roc_auc_score(y_test, preds_test_tree)) # 0.8403661030475336

# Entrenamiento y evaluación del modelo Bagging
base_model = DecisionTreeClassifier()
bag = BaggingClassifier(base_model, n_estimators=N_TREES, n_jobs=-1, random_state=5678, verbose=1)
bag.fit(pd.concat([X_train_red, X_val], axis=0),
        pd.concat([y_train_red, y_val], axis=0))
preds_test_bag = bag.predict_proba(X_test)[:, bag.classes_ == True]
print("ROC AUC Score - Bagging:", roc_auc_score(y_test, preds_test_bag)) # 0.9617250236919546

# Entrenamiento y evaluación del modelo Random Forest
rf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=6789, verbose=1, oob_score=True)
rf.fit(pd.concat([X_train_red, X_val], axis=0),
       pd.concat([y_train_red, y_val], axis=0))
preds_test_rf = rf.predict_proba(X_test)[:, rf.classes_ == True]
print("ROC AUC Score - Random Forest:", roc_auc_score(y_test, preds_test_rf)) # 0.9506521522270438

# Performance oob
preds_oob_rf = rf.oob_decision_function_[:, rf.classes_ == True]
print("OOB ROC AUC Score - Random Forest:", roc_auc_score(pd.concat([y_train_red, y_val]), preds_oob_rf)) # 0.916999750777964

# Importancia de atributos con random forest
def plot_importance(model, n_vars):
    # Sort the DataFrame by 'Importance' column in descending order
    imp_df = pd.DataFrame({"Variable": model.feature_names_in_, "Importance": model.feature_importances_})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)

    # Take only the top 10 rows
    top_imp_df = imp_df.head(n_vars).copy()

    # Scale the importance values to have the max as 100
    max_importance = top_imp_df['Importance'].max()
    top_imp_df['Scaled_Importance'] = (top_imp_df['Importance'] / max_importance) * 100

    # Create the horizontal bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_imp_df['Variable'], top_imp_df['Scaled_Importance'], color='skyblue')
    plt.xlabel('Scaled Importance (Max = 100)')
    plt.ylabel('Variable')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

plot_importance(rf, 10)
