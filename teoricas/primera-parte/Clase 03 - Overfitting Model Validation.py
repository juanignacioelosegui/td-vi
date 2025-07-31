import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import math
from statistics import mean, stdev
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
import matplotlib.pyplot as plt

TREE_SEED = 1234
N_SIMS = 200

#~ Bloque 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# El proceso generador de datos
class data_generating_process():

    def __init__(self, n_features, noise):
        self.n_features = n_features
        self.noise = noise
    
    def draw_samples(self, size, random_state=None):
        X, y = make_regression(size, 
                               n_features = self.n_features,
                               noise=self.noise,
                               random_state=random_state)
        return X, y


def simulate_problem(dgp, model, seed_tr, seed_ts):

    # Se samplea el training set con el que se trabajará (primera instancia de aleatoriedad)
    X_tr, y_tr = dgp.draw_samples(5000, random_state=seed_tr)

    # Se entrena al modelo predictivo
    model.fit(X_tr, y_tr)

    # Se obtienen las instancias del test set (segunda instancia de aleatorieda)
    X_ts, y_ts = dgp.draw_samples(5000, random_state=seed_ts)

    # Se predice sobre el test set
    preds_ts = model.predict(X_ts)

    # Se calcula el error en test
    return math.sqrt(mean_squared_error(y_ts, preds_ts))


def plot_rmse_histogram(rmse_values):
    # Crea un histograma
    plt.hist(rmse_values, bins=20, edgecolor='black', alpha=0.7)

    # Agrega labels y títulos
    plt.xlabel('RMSE Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of RMSE Values')

    # Muestra el histograma
    plt.show()


# Se instancia al proceso generador de datos
dgp = data_generating_process(30, 2)

# Caso 1 (caso de interés): Variabilidad proveniente del testing set (test error / generalization error)
test_errors_ts = []
for _ in tqdm(range(N_SIMS), desc="N simulación"):
    tree = DecisionTreeRegressor(max_depth=5, random_state=TREE_SEED)
    test_errors_ts.append(simulate_problem(dgp, tree, seed_tr=1234, seed_ts=None))

print (f"Desvío estándar del rmse debido sólo a diferentes test sets: {stdev(test_errors_ts):.2f}")
plot_rmse_histogram(test_errors_ts)

# Caso 2 (raro): Variabilidad proveniente del train set
test_errors_tr = []
for _ in tqdm(range(N_SIMS), desc="N simulación"):
    tree = DecisionTreeRegressor(max_depth=5, random_state=TREE_SEED)
    test_errors_tr.append(simulate_problem(dgp, tree, seed_tr=None, seed_ts=5678))

print (f"Desvío estándar del rmse debido sólo a diferentes train sets: {stdev(test_errors_tr):.2f}")
plot_rmse_histogram(test_errors_tr)

# Caso 3: Variabilidad proveniente del ambos sets (expected prediction error / expected test error)
test_errors_tr_ts = []
for _ in tqdm(range(N_SIMS), desc="N simulación"):
    tree = DecisionTreeRegressor(max_depth=5, random_state=TREE_SEED)
    test_errors_tr_ts.append(simulate_problem(dgp, tree, seed_tr=None, seed_ts=None))

print (f"Desvío estándar del rmse debido a ambos sets: {stdev(test_errors_tr_ts):.2f}")
plot_rmse_histogram(test_errors_tr_ts)

#~ Bloque 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def load_data():
    """
    Carga y procesa los datos de entrenamiento y evaluación.
    
    Returns:
    X_train (DataFrame): Datos de entrenamiento.
    X_eval (DataFrame): Datos de evaluación.
    y_train (Series): Etiquetas de entrenamiento.
    y_eval (Series): Etiquetas de evaluación.
    """
    # Se cargan los datos de entrenamiento
    X_train = pd.read_csv("cars_X_train.csv")
    X_train["is_train"] = 1
    y_train = pd.read_csv("cars_y_train.csv")
    df_train = X_train.merge(y_train, on="carID")

    # Se cargan los datos de evaluación
    X_eval = pd.read_csv("cars_X_eval.csv")
    X_eval["is_train"] = 0
    y_eval = pd.read_csv("cars_y_eval.csv")
    df_eval = X_eval.merge(y_eval, on="carID")

    # Se concatenan los datos de entrenamiento y evaluación
    df = pd.concat([df_train, df_eval], axis=0)
    y = df["price"]
    df = pd.get_dummies(df.drop(columns="price"))
    df["price"] = y

    # Se separan los datos de entrenamiento y evaluación
    df_train = df[df["is_train"] == 1]
    df_train = df_train.drop(columns="is_train")
    df_eval = df[df["is_train"] == 0]
    df_eval = df_eval.drop(columns="is_train")
    X_train = df_train.drop(columns="price")
    y_train = df_train["price"]
    X_eval = df_eval.drop(columns="price")
    y_eval = df_eval["price"]

    return X_train, X_eval, y_train, y_eval


def plot_exp(exp_results):
    exp_results = pd.DataFrame(exp_results)
    plt.figure(figsize=(10, 6))
    plt.plot(exp_results["max_depth"], exp_results["rmse_val"], marker='o')
    plt.title('Estimated rmse')
    plt.xlabel('Tree Depth')
    plt.ylabel('rmse')
    plt.grid(True)
    plt.show()


# Se cargan los datos (no se guarda y_eval)
X_train, X_eval, y_train, _ = load_data()

# Se entrena un árbol de decisión sobre los datos de entrenamiento
tree = DecisionTreeRegressor(max_depth=10, random_state=TREE_SEED)
tree.fit(X_train, y_train)
tree.predict(X_eval)

# ¿Qué performance tendrá este árbol en evaluación? (recordemos que se supone que no los conocemos)

# Holdout/Validation set
X_train_red, X_val, y_train_red, y_val = train_test_split(X_train, y_train, test_size=0.2)
tree = DecisionTreeRegressor(max_depth=10, random_state=TREE_SEED)
tree.fit(X_train_red, y_train_red)
preds_val = tree.predict(X_val)
print(f"RMSE estimado mediante velidation set: {math.sqrt(mean_squared_error(y_val, preds_val)):.2f}")

# Repeated holdout/validation set
rmse_rep_val = []
for _ in tqdm(range(50), desc="Repetición"):
    X_train_red, X_val, y_train_red, y_val = train_test_split(X_train, y_train, test_size=0.2)
    tree = DecisionTreeRegressor(max_depth=10, random_state=TREE_SEED)
    tree.fit(X_train_red, y_train_red)
    preds_val = tree.predict(X_val)
    rmse_rep_val.append(math.sqrt(mean_squared_error(y_val, preds_val)))

print(f"RMSE estimado mediante validación repetida (repeated validation set): {mean(rmse_rep_val):.2f}")

#~ Bloque 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Leave-one-out cross validation (LOOCV)
loo = LeaveOneOut()
rmse_loo = []
for train_ix, test_ix in tqdm(loo.split(X_train), total=X_train.shape[0]):
    X_train_red, X_val = X_train.iloc[train_ix,:], X_train.iloc[test_ix,:]
    y_train_red, y_val = y_train[train_ix], y_train[test_ix]
    tree = DecisionTreeRegressor(max_depth=10, random_state=TREE_SEED)
    tree.fit(X_train_red, y_train_red)
    preds_val = tree.predict(X_val)
    rmse_loo.append(math.sqrt(mean_squared_error(y_val, preds_val)))

print(f"RMSE estimado mediante LOOCV: {mean(rmse_loo):.2f}") # Esto da 2811.51

#~ Bloque 4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# k-fold cross validation
kcv = KFold(n_splits=10, shuffle=True)
rmse_kcv1 = []
for train_ix, test_ix in tqdm(kcv.split(X_train), total=10, desc="Fold"):
    X_train_red, X_val = X_train.iloc[train_ix,:], X_train.iloc[test_ix,:]
    y_train_red, y_val = y_train[train_ix], y_train[test_ix]
    tree = DecisionTreeRegressor(max_depth=10, random_state=TREE_SEED)
    tree.fit(X_train_red, y_train_red)
    preds_val = tree.predict(X_val)
    rmse_kcv1.append(math.sqrt(mean_squared_error(y_val, preds_val)))

print(f"RMSE estimado mediante k-fold CV (Versión 1): {mean(rmse_kcv1):.2f}")

# Validación cruzada k-fold utilizando cross_val_score
kcv = KFold(n_splits=10, shuffle=True)
tree = DecisionTreeRegressor(max_depth=10, random_state=TREE_SEED)
rmse_kcv2 = cross_val_score(tree, X_train, y_train, scoring='neg_root_mean_squared_error', cv=kcv, n_jobs=-1)

print(f"RMSE estimado mediante k-fold CV (Versión 2): {-1 * mean(rmse_kcv2):.2f}")

#~ Bloque 5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Training, holdout y testing sets
size_val = math.ceil(0.1 * X_train.shape[0])
size_test = math.ceil(0.1 * X_train.shape[0])

X_train_red, X_test, y_train_red, y_test = train_test_split(X_train, y_train, test_size=size_test)
X_train_red, X_val, y_train_red, y_val = train_test_split(X_train_red, y_train_red, test_size=size_val)

# Prueba con distintos niveles de profundidad (model selection)
exp_results = []
for md in tqdm(range(1, 51), desc="Probando profundidades"):
    tree = DecisionTreeRegressor(max_depth=md, random_state=TREE_SEED)
    tree.fit(X_train_red, y_train_red)
    preds_val = tree.predict(X_val)
    exp_results.append({"max_depth": md,
                        "rmse_val": math.sqrt(mean_squared_error(y_val, preds_val))})

exp_results = pd.DataFrame(exp_results)
plot_exp(exp_results)

# Se entrena el árbol con la mejor profundidad encontrada sobre train set + validation set
best_md = exp_results[exp_results["rmse_val"].min() == exp_results["rmse_val"]]
best_md = best_md.sort_values("max_depth").iloc[0,:]
print(f"Performance del mejor modelo: {best_md['rmse_val']:.2f}")

tree = DecisionTreeRegressor(max_depth=int(best_md["max_depth"]), random_state=TREE_SEED)
tree.fit(pd.concat([X_train_red, X_val], axis=0),
         pd.concat([y_train_red, y_val], axis=0))
preds_test = tree.predict(X_test)

print(f"RMSE estimado en test: {math.sqrt(mean_squared_error(y_test, preds_test)):.2f}")

# Evaluación final en el conjunto de evaluación (model assestment)
tree = DecisionTreeRegressor(max_depth=int(best_md["max_depth"]), random_state=TREE_SEED)
tree.fit(X_train, y_train)
preds_eval = tree.predict(X_eval)
_, _, _, y_eval = load_data()

print(f"Performance de evaluación: {math.sqrt(mean_squared_error(y_eval, preds_eval)):.2f}")
