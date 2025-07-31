import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import make_pipeline

# Cargar el conjunto de datos de Boston
def carga_datos_boston():
    """
    Carga el conjunto de datos de Boston desde una fuente en línea y realiza algunas transformaciones.
    
    Returns:
        DataFrame: El conjunto de datos de Boston procesado.
    """
    data = pd.read_csv(
        filepath_or_buffer="http://lib.stat.cmu.edu/datasets/boston",
        delim_whitespace=True,
        skiprows=21,
        header=None,
    )

    columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
               "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

    # Aplanar todos los valores en una única lista larga y eliminar los valores nulos
    valores_con_nulos = data.values.flatten()
    todos_los_valores = valores_con_nulos[~np.isnan(valores_con_nulos)]

    # Reorganizar los valores para tener 14 columnas y crear un nuevo DataFrame
    data = pd.DataFrame(
        data = todos_los_valores.reshape(-1, len(columns)),
        columns = columns,
    )

    return data

data = carga_datos_boston()

# Mostrar las primeras filas y estadísticas resumidas
print(data.head())
print(data.info())
print(data.describe())

# Graficar las primeras 5 columnas
pd.plotting.scatter_matrix(data.iloc[:, :5])
plt.show()

# Clustering con K-means
k = 7
kmeans = KMeans(n_clusters=k, max_iter=30, n_init=20)
kmeans.fit(data)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

# Graficar la evolución de la función objetivo a medida que K aumenta
evol_variabilidad = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, max_iter=30, n_init=20)
    kmeans.fit(data)
    evol_variabilidad.append({"k": k, "var": kmeans.inertia_})

evol_variabilidad = pd.DataFrame(evol_variabilidad)
plt.figure()
plt.plot(evol_variabilidad["k"], evol_variabilidad["var"], marker="o")
plt.xlabel("# Clusters")
plt.ylabel("tot.withinss")
plt.show()

# Graficar los clusters asignados (K=4 en este caso)
k = 4
kmeans = KMeans(n_clusters=k, max_iter=30, n_init=20)
kmeans.fit(data)
data["cluster"] = kmeans.labels_
pd.plotting.scatter_matrix(data.drop(columns="cluster"), c=data["cluster"], cmap="viridis")
plt.show()

# Interpretar los clusters usando un árbol de decisión
tree_classifier = DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.1)
tree_classifier.fit(data.drop(columns="cluster"), data["cluster"])
plt.figure(figsize=(12, 6))
plot_tree(tree_classifier, feature_names=data.columns[:-1], class_names=[str(i) for i in range(k)], filled=True)
plt.show()

pd.plotting.scatter_matrix(data[["TAX", "B"]], c=data["cluster"], cmap="viridis")
plt.show()

# Graficar la evolución de la función objetivo para los datos estandarizados
evol_variabilidad_scaled = []
for k in range(1, 21):
    kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=k, max_iter=30, n_init=20))
    kmeans.fit(data.drop(columns="cluster"))
    evol_variabilidad_scaled.append({"k": k, "var": kmeans["kmeans"].inertia_})

evol_variabilidad_scaled = pd.DataFrame(evol_variabilidad_scaled)
plt.figure()
plt.plot(evol_variabilidad_scaled["k"], evol_variabilidad_scaled["var"], marker="o")
plt.xlabel("# Clusters")
plt.ylabel("tot.withinss")
plt.show()

# Clustering con K-means para los datos estandarizados (K=5 en este caso)
k = 5
kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=k, max_iter=30, n_init=20))
kmeans.fit(data.drop(columns="cluster"))
data["cluster"] = kmeans["kmeans"].labels_
tree_classifier = DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.01)
tree_classifier.fit(StandardScaler().fit_transform(data.drop(columns="cluster")), data["cluster"])
plt.figure(figsize=(12, 6))
plot_tree(tree_classifier, feature_names=data.columns[:-1], class_names=[str(i) for i in range(k)], filled=True)
plt.show()

# Realizar clustering jerárquico con diferentes métodos de enlace
hc_complete = linkage(StandardScaler().fit_transform(data.drop(columns="cluster")), method='complete')
hc_average = linkage(StandardScaler().fit_transform(data.drop(columns="cluster")), method='average')
hc_single = linkage(StandardScaler().fit_transform(data.drop(columns="cluster")), method='single')

# Visualizar los dendrogramas verticales sin colores
plt.figure(figsize=(15, 5))
plt.subplot(131)
dendrogram(hc_complete, no_labels=True, orientation='top', color_threshold=np.inf)
plt.title("Complete linkage")
plt.subplot(132)
dendrogram(hc_average, no_labels=True, orientation='top', color_threshold=np.inf)
plt.title("Average linkage")
plt.subplot(133)
dendrogram(hc_single, no_labels=True, orientation='top', color_threshold=np.inf)
plt.title("Single linkage")
plt.tight_layout()
plt.show()

# Asignar clusters con 4 cortes usando enlace completo
num_clusters = 4
asignaciones = cut_tree(hc_complete, n_clusters=num_clusters).flatten()
print(asignaciones)
