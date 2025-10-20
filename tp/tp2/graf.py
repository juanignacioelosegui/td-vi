# -*- coding: utf-8 -*-
# Requisitos: pandas, numpy, matplotlib
# pip install pandas numpy matplotlib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
PATH = "competition_data/train_data.txt"  # <-- Cambiá por la ruta a tu .txt
# Si tu archivo tiene otro nombre, ej.: "spotify_train.txt" o "data.txt"

# === CARGA ROBUSTA DEL TXT ===
# sep=None con engine='python' intenta inferir delimitador (coma, tab, pipe, etc.)
df = pd.read_csv(
    PATH,
    sep=None,
    engine="python",
)
# Normalizamos nombres de columnas a minúsculas para evitar errores
df.columns = [c.strip().lower() for c in df.columns]

# === UTILIDADES ===
def has_cols(*cols):
    return all(c in df.columns for c in cols)

# ===== CASO 1: TENEMOS ETIQUETAS (reason_end) =====
if "reason_end".lower() in df.columns:
    # Variable objetivo: forward = 1 si reason_end == "fwdbtn"
    y = (df["reason_end"].astype(str) == "fwdbtn").astype(int)
    df["is_forward"] = y

    # Parseo de timestamp
    ts_col = None
    for cand in ["ts", "timestamp", "time", "datetime"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError(
            "No se encontró columna temporal (ej.: ts). Añadí una columna de tiempo para construir sesiones."
        )
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    if df[ts_col].isna().all():
        raise ValueError("No se pudo parsear el timestamp. Revisa el formato de la columna de tiempo.")

    # Usuario (para segmentar sesiones)
    user_col = None
    for cand in ["username", "user_id", "user", "uid"]:
        if cand in df.columns:
            user_col = cand
            break
    if user_col is None:
        # Si no hay usuario, creamos un usuario único y hacemos sesiones globales
        user_col = "_tmp_user"
        df[user_col] = "user_1"

    # Si no existen session_id / pos_in_session, los construimos
    if not has_cols("session_id", "pos_in_session"):
        df = df.sort_values([user_col, ts_col]).reset_index(drop=True)
        # Diferencia temporal por usuario
        dt = df.groupby(user_col)[ts_col].diff()
        # Nueva sesión si cambia usuario o gap > 30 minutos
        new_session = (dt.isna()) | (dt.dt.total_seconds() > 1800)
        # ID de sesión incremental
        df["session_id"] = (new_session).groupby(df[user_col]).cumsum().astype(int)
        # Posición en la sesión (1, 2, 3, …)
        df["pos_in_session"] = df.groupby([user_col, "session_id"]).cumcount() + 1

    # =======================
    # GRÁFICO 1: Forward rate por posición en la sesión
    # =======================
    # Para no ir muy lejos, limitamos a las primeras 20 posiciones
    max_pos = 20
    g_pos = (
        df.loc[df["pos_in_session"] <= max_pos]
        .groupby("pos_in_session")["is_forward"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(g_pos["pos_in_session"], g_pos["is_forward"], marker="o")
    plt.title("Tasa de forward por posición en la sesión (≤ 20)")
    plt.xlabel("Posición en la sesión")
    plt.ylabel("Tasa de forward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("forward_rate_by_session_position.png", dpi=150)
    plt.show()

    # =======================
    # GRÁFICO 2: Forward rate por hora del día
    # =======================
    # Convertimos a hora local si querés (acá se deja en UTC para simplicidad)
    df["hour"] = df[ts_col].dt.hour
    g_hour = df.groupby("hour")["is_forward"].mean().reset_index()

    plt.figure(figsize=(8, 4.5))
    plt.plot(g_hour["hour"], g_hour["is_forward"], marker="o")
    plt.title("Tasa de forward por hora del día")
    plt.xlabel("Hora (0–23)")
    plt.ylabel("Tasa de forward")
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("forward_rate_by_hour.png", dpi=150)
    plt.show()

    print(
        "Listo. Se guardaron:\n"
        " - forward_rate_by_session_position.png\n"
        " - forward_rate_by_hour.png"
    )

# ===== CASO 2: NO HAY ETIQUETAS PERO TENEMOS PREDICCIONES (pred_proba) =====
elif "pred_proba" in df.columns:
    # Distribución de probabilidades → Histograma + CDF
    proba = df["pred_proba"].astype(float).clip(0, 1)

    # Gráfico A: Histograma
    plt.figure(figsize=(8, 4.5))
    plt.hist(proba, bins=30)
    plt.title("Distribución de probabilidades predichas (pred_proba)")
    plt.xlabel("pred_proba")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pred_proba_hist.png", dpi=150)
    plt.show()

    # Gráfico B: CDF
    sorted_vals = np.sort(proba.values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    plt.figure(figsize=(8, 4.5))
    plt.plot(sorted_vals, cdf)
    plt.title("Función de distribución acumulada (CDF) de pred_proba")
    plt.xlabel("pred_proba")
    plt.ylabel("Proporción acumulada")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pred_proba_cdf.png", dpi=150)
    plt.show()

    print(
        "No se encontraron etiquetas (reason_end). Generé:\n"
        " - pred_proba_hist.png\n"
        " - pred_proba_cdf.png"
    )

# ===== CASO 3: FALTA TODO LO ANTERIOR =====
else:
    raise ValueError(
        "El archivo no tiene 'reason_end' ni 'pred_proba'. "
        "Para gráficos con patrones, incluí al menos una de esas columnas."
    )
