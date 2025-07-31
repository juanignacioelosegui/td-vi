import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_fscore_support, precision_recall_curve, average_precision_score
import numpy as np
from tqdm import tqdm

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

# Cargamos los datos bancarios
X_train, X_val, y_train, y_val = load_bank_data()

# Entreno un modelo (random forest) y predigo sobre el conjunto validación
obj_vars = X_train.select_dtypes(include=["object"]).columns.tolist()
num_vars = X_train.select_dtypes(include=["int", "float"]).columns.tolist()
rf = make_pipeline(ColumnTransformer(transformers=[("num", "passthrough", num_vars),
                                                   ("cat", OneHotEncoder(), obj_vars)]),
                   RandomForestClassifier(n_estimators=1000, n_jobs=-1))

rf.fit(X_train, y_train)

preds_on_val = pd.DataFrame({"y_true": y_val,
                             "y_pred": rf.predict_proba(X_val)[:,rf.classes_ == "yes"].flatten()})

# Vemos cómo se distribuyen las probabilidades predichas en validación
def plot_preds(df):
    plt.figure(figsize=(10, 6))
    # Grafico las estimaciones de densidad para "yes"
    sns.kdeplot(data=df.loc[df["y_true"] == "yes", "y_pred"], label="Yes", fill=True)

    # Grafico las estimaciones de densidad para "no"
    sns.kdeplot(data=df.loc[df["y_true"] == "no", "y_pred"], label="no", fill=True)

    plt.xlim(0, 1)
    plt.xlabel("Predicted Probability for Class Yes")
    plt.ylabel("Density")
    plt.title("Density Plot of Predicted Probabilities")
    plt.legend()
    plt.show()

plot_preds(preds_on_val)

# Veamos cómo se comporta precision, recall y F1 para diferentes thresholds
precision_recall_fscore_support(preds_on_val["y_true"],
                                ["yes" if e > 0.5 else "no" for e in preds_on_val["y_pred"]],
                                pos_label="yes", average="binary")

precision_recall_fscore_support(preds_on_val["y_true"],
                                ["yes" if e > 0.3 else "no" for e in preds_on_val["y_pred"]],
                                pos_label="yes", average="binary")

precision_recall_fscore_support(preds_on_val["y_true"],
                                ["yes" if e > 0.7 else "no" for e in preds_on_val["y_pred"]],
                                pos_label="yes", average="binary")

# Visualización de la Curva ROC
def plot_roc_curve(df):
    """
    Grafica la curva ROC utilizando la tasa de falsos positivos (FPR) y la tasa de verdaderos positivos (TPR).
    """
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_pred"], pos_label="yes")
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Característica de Operación del Receptor (ROC)")
    plt.show()

# Graficamos la Curva ROC
plot_roc_curve(preds_on_val)

# Calculamos el área bajo la curva ROC (AUC-ROC)
roc_auc_score(preds_on_val["y_true"], preds_on_val["y_pred"])

# Experimento para interpretar probabilísticamente la Curva ROC
def calculate_roc_as_prob(df, size):
    """
    Realiza un experimento para interpretar probabilísticamente la Curva ROC.
    """
    y = df.loc[df["y_true"] == "yes", "y_pred"].values
    n = df.loc[df["y_true"] == "no", "y_pred"].values

    is_greater = 0
    for _ in tqdm(range(size)):
        if (np.random.choice(y, 1) > np.random.choice(n, 1))[0]:
            is_greater += 1

    return is_greater / size

# Calculamos el área bajo la Curva ROC como una interpretación probabilística
calculate_roc_as_prob(preds_on_val, 1500000)

# Visualización de la Curva PR
def plot_pr_curve(df):
    """
    Grafica la curva precision-recall utilizando la precisión y el recall.
    """
    precision, recall, _ = precision_recall_curve(df["y_true"], df["y_pred"], pos_label="yes")
    plt.figure()
    plt.plot(recall, precision, color="darkorange", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precisión")
    plt.title("Curva Precisión-Recall")
    plt.show()

# Graficamos la Curva PR
plot_pr_curve(preds_on_val)

# Calculamos el Área bajo la Curva PR (AUC-PR)
average_precision_score(preds_on_val["y_true"], preds_on_val["y_pred"], pos_label="yes")

# Ejercicio asumiendo que conocemos la matriz de costos
cost_matrix = np.array([[0, 120], [1000, 120]])

# Calculamos la matriz de confusión con umbral de 0.5 y evaluamos el costo total
cm = confusion_matrix(preds_on_val["y_true"],
                      ["yes" if e > 0.5 else "no" for e in preds_on_val["y_pred"]])

print("Costo total con umbral de 0.5:", (cost_matrix * cm).sum())

# Calculamos la matriz de confusión con umbral de 0.12 y evaluamos el costo total
cm = confusion_matrix(preds_on_val["y_true"],
                      ["yes" if e > 0.12 else "no" for e in preds_on_val["y_pred"]])

print("Costo total con umbral de 0.12:", (cost_matrix * cm).sum())

# Función para graficar la curva de costo en función del umbral
def plot_cost_curve(df, cost_matrix):
    """
    Grafica el costo en validación en función del threshold.
    """
    cost_data = []
    for t in tqdm(np.arange(0, 1, 0.01)):
        cm = confusion_matrix(df["y_true"],
                              ["yes" if e > t else "no" for e in df["y_pred"]])
        cost_data.append({"umbral": t,
                          "costo": (cm * cost_matrix).sum()})

    cost_data = pd.DataFrame(cost_data)

    plt.plot(cost_data["umbral"], cost_data["costo"], marker=None, linestyle="-")

    # Agregar etiquetas y título
    plt.xlabel("Umbral")
    plt.ylabel("Costo")
    plt.title("Costo vs. Umbral")
    plt.show()

# Graficamos la curva de costo en validación en función del umbral
plot_cost_curve(preds_on_val, cost_matrix)
