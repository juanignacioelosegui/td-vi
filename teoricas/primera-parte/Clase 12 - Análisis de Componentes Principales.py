from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

#~ Bloque 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Carga los datos de USArrests desde el conjunto de datos R
USArrests = get_rdataset("USArrests").data

# Mínimo análisis del DataFrame
USArrests.head()
USArrests.describe()
pd.plotting.scatter_matrix(USArrests)
plt.show()

# Obtiene la matriz de varianzas y covarianzas
USArrests.cov(ddof=0)

# Obtiene la matriz de correlaciones
USArrests.corr()

# Confirmamos que las matriz de covarianza de las variables estandarizadas es igual a la de correlación de las originales
scaler = StandardScaler(with_std=True, with_mean=True)
USArrests_scaled = pd.DataFrame(scaler.fit_transform(USArrests), columns=USArrests.columns)
USArrests_scaled.cov(ddof=0)

#~ Bloque 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Realiza el análisis de componentes principales (PCA) en los datos escalados
pcaUS = PCA()
pcaUS.fit(USArrests_scaled)

# Crea un DataFrame para almacenar los resultados del PCA
scores = pd.DataFrame(pcaUS.transform(USArrests_scaled), index=USArrests.index)
scores.head()

# Veamos qué valores de los loadings se obtuvieron (cada fila es un vector de loadins)
print(pcaUS.components_)

def plot_biplot(pca_res, scores):
    """
    Función para crear un gráfico biplot a partir de los resultados del PCA.
    """
    i, j = 0, 1
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Grafica las puntuaciones en el espacio de las dos primeras componentes principales
    ax.scatter(scores.values[:, 0], scores.values[:, 1])
    ax.set_xlabel('PC%d' % (i + 1))
    ax.set_ylabel('PC%d' % (j + 1))
    
    # Agrega flechas y etiquetas para las cargas de las variables en el biplot
    for k in range(pca_res.components_.shape[1]):
        ax.arrow(0, 0, pca_res.components_[i, k], pca_res.components_[j, k])
        ax.text(pca_res.components_[i, k], pca_res.components_[j, k], USArrests.columns[k])
    
    # Agrega etiquetas a las observaciones
    for k, txt in enumerate(scores.index):
        ax.annotate(txt, (scores.values[k, 0], scores.values[k, 1]), fontsize=8, color='gray')
    
    # Agrega líneas verticales y horizontales en el origen
    ax.axvline(0, color='lightgray', linestyle='--')
    ax.axhline(0, color='lightgray', linestyle='--')

    plt.show()

# Llama a la función plot_biplot para crear el biplot
plot_biplot(pcaUS, scores)

#~ Bloque 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Crea un gráfico de scree plot para visualizar la varianza explicada por cada componente
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(pcaUS.explained_variance_ratio_) + 1),
        pcaUS.explained_variance_ratio_, align='center')
plt.xlabel('Componente Principal')
plt.ylabel('Ratio de Varianza Explicada')
plt.title('Scree Plot')
plt.xticks([e + 1 for e in range(len(pcaUS.explained_variance_ratio_))])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Crea un gráfico de la varianza explicada acumulativa
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(pcaUS.explained_variance_ratio_.cumsum()) + 1),
         pcaUS.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulativa')
plt.title('Gráfico de Varianza Explicada Acumulativa')
plt.grid(linestyle='--', alpha=0.7)
plt.show()
