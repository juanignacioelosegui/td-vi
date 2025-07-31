import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import make_pipeline
from math import sqrt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# Definición de una función personalizada para suprimir las advertencias
def warn(*args, **kwargs):
    """
    Función que anula las advertencias, simplemente no realiza ninguna acción.

    Parámetros:
        *args, **kwargs: Argumentos y palabras clave (keywords) pasados a la función. No se utilizan en esta implementación.
    """
    pass

warnings.warn = warn

# ~ Bloque 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def rmse(y_true, y_pred):
    """
    Calcula la raíz del error cuadrático medio (RMSE) entre las predicciones y los valores reales.

    Parámetros:
        y_true (array-like): Valores reales.
        y_pred (array-like): Valores predichos.

    Returns:
        float: Raíz del error cuadrático medio.
    """
    mse = mean_squared_error(y_true, y_pred)
    return sqrt(mse)

def load_data():
    """
    Carga y procesa los datos de entrenamiento y evaluación.

    Returns:
        X_train (DataFrame): Datos de entrenamiento.
        X_eval (DataFrame): Datos de evaluación.
        y_train (Series): Etiquetas de entrenamiento.
        y_eval (Series): Etiquetas de evaluación.
    """
    # Carga de los datos de entrenamiento
    X_train = pd.read_csv("cars_X_train.csv")
    X_train["is_train"] = 1
    y_train = pd.read_csv("cars_y_train.csv")
    df_train = X_train.merge(y_train, on="carID")

    # Carga de los datos de evaluación
    X_eval = pd.read_csv("cars_X_eval.csv")
    X_eval["is_train"] = 0
    y_eval = pd.read_csv("cars_y_eval.csv")
    df_eval = X_eval.merge(y_eval, on="carID")

    # Concatenación de los datos de entrenamiento y evaluación
    df = pd.concat([df_train, df_eval], axis=0)
    y = df["price"]
    df = pd.get_dummies(df.drop(columns="price"))
    df["price"] = y

    # Separación de los datos de entrenamiento y evaluación
    df_train = df[df["is_train"] == 1]
    df_train = df_train.drop(columns="is_train")
    df_eval = df[df["is_train"] == 0]
    df_eval = df_eval.drop(columns="is_train")
    X_train = df_train.drop(columns="price")
    y_train = df_train["price"]
    X_eval = df_eval.drop(columns="price")
    y_eval = df_eval["price"]

    return X_train, X_eval, y_train, y_eval

def plot_rmse_by_n_vars(df):
    """
    Genera una gráfica de RMSE en función del número de variables utilizadas.

    Parámetros:
        df (DataFrame): Dataframe con resultados de RMSE en función del número de variables.
    """
    plt.figure()
    plt.plot(df["n_preds"], df["val_rmse"], color="blue", linestyle="-")
    plt.title("Relación entre el número de variables y el RMSE de evaluación")
    plt.xlabel("Número de Variables")
    plt.ylabel("RMSE de Evaluación")
    plt.grid(True)
    plt.show()

# Carga de datos de entrenamiento y evaluación
X_train, X_eval, y_train, y_eval = load_data()

# Modelo de regresión lineal
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds_eval = lr.predict(X_eval)
lr_rmse = rmse(y_eval, lr_preds_eval)
print(f"Regresión lineal RMSE: {lr_rmse:.2f}")  # 6032.84

# Búsqueda del mejor valor de alpha para Ridge Regression
best_alpha_ridge = None
best_rmse_ridge = float("inf")
for alpha in tqdm([0.01, 0.1, 1.0, 10., 15., 20., 25., 30.]):
    ridge = make_pipeline(StandardScaler(),
                          Ridge(alpha=alpha))
    ridge.fit(X_train, y_train)
    ridge_preds_eval = ridge.predict(X_eval)
    ridge_rmse = rmse(y_eval, ridge_preds_eval)
    if ridge_rmse < best_rmse_ridge:
        best_rmse_ridge, best_alpha_ridge = ridge_rmse, alpha

print(f"Ridge regression RMSE: {best_rmse_ridge:.2f}")  # 6032.07

# Búsqueda del mejor valor de alpha para Lasso Regression
best_alpha_lasso = None
best_rmse_lasso = float("inf")
for alpha in tqdm([0.01, 0.1, 1., 1.25]):
    lasso = make_pipeline(StandardScaler(),
                          Lasso(alpha=alpha, max_iter=10000))
    lasso.fit(X_train, y_train)
    lasso_preds_eval = lasso.predict(X_eval)
    lasso_rmse = rmse(y_eval, lasso_preds_eval)
    if lasso_rmse < best_rmse_lasso:
        best_rmse_lasso, best_alpha_lasso = lasso_rmse, alpha

print(f"Lasso regression RMSE: {best_rmse_lasso:.2f}")  # 6032.84

# Regresión lineal con interacciones
lr_int = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                       LinearRegression())
lr_int.fit(X_train, y_train)
lr_int_preds_eval = lr_int.predict(X_eval)
lr_int_rmse = rmse(y_eval, lr_int_preds_eval)
print(f"Regresión lineal con interacciones RMSE: {lr_int_rmse:.2f}")  # 251213361.60

# Búsqueda del mejor valor de alpha para Ridge Regression con interacciones
best_alpha_int_ridge = None
best_rmse_int_ridge = float("inf")
for alpha in tqdm([0.01, 0.1, 1.0, 10., 15., 20., 25., 30.]):
    ridge = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                          StandardScaler(),
                          Ridge(alpha=alpha))
    ridge.fit(X_train, y_train)
    ridge_preds_eval = ridge.predict(X_eval)
    ridge_rmse = rmse(y_eval, ridge_preds_eval)
    if ridge_rmse < best_rmse_int_ridge:
        best_rmse_int_ridge, best_alpha_int_ridge = ridge_rmse, alpha

print(f"Ridge regression con interacciones RMSE: {best_rmse_int_ridge:.2f}")  # 4156.87

# Búsqueda del mejor valor de alpha para Lasso Regression con interacciones
best_alpha_int_lasso = None
best_rmse_int_lasso = float("inf")
for alpha in tqdm([0.01, 0.1, 1., 1.25]):
    lasso = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                          StandardScaler(),
                          Lasso(alpha=alpha))
    lasso.fit(X_train, y_train)
    lasso_preds_eval = lasso.predict(X_eval)
    lasso_rmse = rmse(y_eval, lasso_preds_eval)
    if lasso_rmse < best_rmse_int_lasso:
        best_rmse_int_lasso, best_alpha_int_lasso = lasso_rmse, alpha

print(f"Lasso regression con interacciones RMSE: {best_rmse_int_lasso:.2f}")  # 4585.19

# ~ Bloque 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Modelo de regresión por árboles de decisión
tree = DecisionTreeRegressor(random_state=2345)
tree.fit(X_train, y_train)
tree_preds_eval = tree.predict(X_eval)
tree_rmse = rmse(y_eval, tree_preds_eval)
print("Árbol de decisión RMSE:", tree_rmse)  # 4342.87

# Eliminación hacia atrás de atributos en árbol de decisión
tree_exp = []
preds_attrs = X_train.columns
for _ in tqdm(range(X_train.shape[1])):
    tree = DecisionTreeRegressor(random_state=2345)
    tree.fit(X_train[preds_attrs], y_train)
    tree_preds = tree.predict(X_eval[preds_attrs])
    tree_exp.append({"n_preds": len(preds_attrs),
                     "val_rmse": rmse(y_eval, tree_preds)})
    to_drop = preds_attrs[tree.feature_importances_ == min(tree.feature_importances_)]
    to_drop = np.random.choice(to_drop)
    preds_attrs = preds_attrs[preds_attrs != to_drop]

tree_exp = pd.DataFrame(tree_exp)
print(f"Árbol con eliminación hacía atrás RMSE: {tree_exp['val_rmse'].min():.2f}")  # 4162.27

plot_rmse_by_n_vars(tree_exp)

# ~ Bloque 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Experimentación de regresión lineal con selección de atributos utilizando filtros
lr_filter_exp = []
for i in tqdm(range(X_train.shape[1])):
    attr_selector = SelectKBest(f_regression, k=X_train.shape[1] - i)
    attr_selector.fit(X_train, y_train)
    lr_filtered = LinearRegression()
    lr_filtered.fit(attr_selector.transform(X_train), y_train)
    lr_filtered_preds_eval = lr_filtered.predict(attr_selector.transform(X_eval))
    lr_filter_exp.append({"n_preds": X_train.shape[1] - i,
                          "val_rmse": rmse(y_eval, lr_filtered_preds_eval)})

lr_filter_exp = pd.DataFrame(lr_filter_exp)
plot_rmse_by_n_vars(lr_filter_exp)

# ~ Bloque 4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Experimentación con filtering
X_sim = np.random.normal(size=(2500, 10000))
y_sim = np.where(np.random.choice([0, 1], size=2500), "y", "n")

attr_selector = SelectKBest(f_classif, k=200)
attr_selector.fit(X_sim, y_sim)

X_sim_train, X_sim_test, y_sim_train, y_sim_test = train_test_split(X_sim, y_sim, test_size=0.5)
tree = RandomForestClassifier(n_estimators=500, n_jobs=-1)
tree.fit(attr_selector.transform(X_sim_train), y_sim_train)
y_sim_pred_test = tree.predict(attr_selector.transform(X_sim_test))

print(f"La precisión en validación es de: {(y_sim_test == y_sim_pred_test).mean():.2f}")
