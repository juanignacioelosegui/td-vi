import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

# Dataset de ejemplo:
# df = pd.read_csv("OnlineNewsPopularity.csv", sep=", ")
# X = df[["n_tokens_content"]].to_numpy()

# Demo sintética:
rng = np.random.default_rng(0)
X = rng.normal(400, 150, size=(5000, 1)).clip(min=10)  # "conteo de palabras"

# Min-Max [0,1]
minmax = MinMaxScaler().fit_transform(X)

# Estandarización (media 0, var 1)
standardized = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

# L2-normalización por columna (norma de la columna = 1)
l2_col = normalize(X, axis=0, norm="l2")

# Armar DataFrame para graficar
df_plot = pd.DataFrame({
    "original": X.ravel(),
    "minmax": minmax.ravel(),
    "standardized": standardized.ravel(),
    "l2_normalized": l2_col.ravel(),
})

# Hist: escalar no cambia la forma, solo la escala
fig, axes = plt.subplots(4, 1, figsize=(6, 10))
df_plot["original"].hist(ax=axes[0], bins=100);      axes[0].set_xlabel("Original");        axes[0].set_ylabel("Conteo")
df_plot["minmax"].hist(ax=axes[1], bins=100);        axes[1].set_xlabel("Min-Max")
df_plot["standardized"].hist(ax=axes[2], bins=100);  axes[2].set_xlabel("Estandarizado")
df_plot["l2_normalized"].hist(ax=axes[3], bins=100); axes[3].set_xlabel("L2 normalizado (columna)")
for ax in axes: ax.tick_params(labelsize=12)
plt.tight_layout(); plt.show()