import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_columns", None)

COMPETITION_PATH = os.path.expanduser("~/td-vi/tp/tp2/competition_data")
TRAIN_DATA_PATH = "train_data.txt"
TEST_DATA_PATH = "test_data.txt"

def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    print("Cargando datasets de competencia desde:", data_dir)
    train_file = os.path.join(data_dir, TRAIN_DATA_PATH)
    test_file = os.path.join(data_dir, TEST_DATA_PATH)

    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → DataFrame concatenado: {combined.shape[0]} rows")
    return combined


def cast_column_types(df):
    
    print("Casteando tipos de columnas y parseando campos de datetime...")
    
    #   Fechas
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "offline_timestamp" in df.columns:
        df["offline_timestamp"] = pd.to_datetime(
            df["offline_timestamp"], unit="s", errors="coerce", utc=True
        )

    #   Categóricas “manejables” (low-card)
    for col in ["platform", "conn_country", "reason_end",
                "master_metadata_track_name", "master_metadata_album_artist_name",
                "master_metadata_album_album_name", "username", "ip_addr"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    #   Flags
    for col in ["shuffle", "offline", "incognito_mode"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    #   Claves/ids
    for col in ["obs_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    print("  → Tipos de columna casteados.")
    return df


def build_features(df):
    '''
    Features básicas sin fuga de info.
    '''
    print("Armando features...")

    #   Orden temporal por usuario
    df = df.sort_values(["username", "ts"], kind="mergesort")
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1

    #   Separación de calendario
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")

    #   Tiempo desde la reproducción anterior del mismo usuario (en segundos)
    df["ts_unix"] = df["ts"].astype("int64") // 10**9
    df["last_ts_unix"] = df.groupby("username", observed=True)["ts_unix"].shift(1)
    df["secs_since_prev"] = (df["ts_unix"] - df["last_ts_unix"]).clip(lower=0)
    df["secs_since_prev"] = df["secs_since_prev"].fillna(df["secs_since_prev"].median())

    #   Target e indicador de test
    df["target"] = (df["reason_end"] == "fwdbtn").astype("Int8")
    df["is_test"] = df["reason_end"].isna()

    #   === Tasa histórica por usuario SIN ver el futuro, y SIN merge ===
    #   s = target desplazado una posición (lo que pasó "antes" para ese usuario)
    s = df.groupby("username", observed=True)["target"].shift(1)

    #   Conteo acumulado de observaciones NO nulas en s (o sea, cuántas "anteriores" tengo)
    cnt_nonnull = s.notna().groupby(df["username"], observed=True).cumsum()

    #   Suma acumulada de forwards históricos (NaN lo tomamos como 0)
    sum_hist = s.fillna(0).groupby(df["username"], observed=True).cumsum()

    #   Ratio histórico: suma / conteo (solo cuando hay al menos una previa)
    df["user_forward_rate_hist"] = (sum_hist / cnt_nonnull).astype("float32")

    #   Imputación: si el usuario no tiene historia, uso la tasa global del TRAIN
    global_rate = df.loc[~df["is_test"], "target"].mean()
    df["user_forward_rate_hist"] = df["user_forward_rate_hist"].fillna(global_rate).astype("float32")

    #   Imputo con promedio global de train si no hay historial
    global_rate = df.loc[~df["is_test"], "target"].mean()
    df["user_forward_rate_hist"] = df["user_forward_rate_hist"].fillna(global_rate)

    #   Variables finales (raw + derivadas low-card)
    cat_cols = []
    for c in ["platform", "conn_country", "shuffle", "offline", "incognito_mode"]:
        if c in df.columns:
            cat_cols.append(c)

    num_cols = ["user_order",
                "hour",
                "dow",
                "is_weekend",
                "secs_since_prev",
                "user_forward_rate_hist"]

    num_cols = [c for c in num_cols if c in df.columns]

    keep = ["obs_id", "username", "is_test", "target"] + cat_cols + num_cols

    df = df[keep].copy()

    print("  → Features listas.")
    return df, cat_cols, num_cols


def split_train_test(df):
    '''
    Usar la máscara is_test ya creada.
    '''

    print("Splitteando datos en train/test sets...")
    train_df = df.loc[~df["is_test"]].copy()
    test_df = df.loc[df["is_test"]].copy()

    print(f"  → Training set: {train_df.shape[0]} rows")
    print(f"  → Test set:     {test_df.shape[0]} rows")
    return train_df, test_df


def train_and_predict(train_df, test_df, cat_cols, num_cols):
    '''
    Pipeline: OHE (categóricas low-card) + LR; predicción de probas para test.
    '''
    
    X_train = train_df[cat_cols + num_cols]
    y_train = train_df["target"].astype(int)

    X_test = test_df[cat_cols + num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    #   Modelo estable y rápido para AUC
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    print("Entrenando modelo (Pipeline: OHE + Regresión Logística)...")
    pipe.fit(X_train, y_train)
    print("  → Entrenamiento completado.")

    print("Generando predicciones para el test set...")
    preds_proba = pipe.predict_proba(X_test)[:, 1]
    out = pd.DataFrame({"obs_id": test_df["obs_id"].astype(int), "pred_proba": preds_proba})
    out.to_csv("modelo_benchmark.csv", index=False)
    print("  → Predicciones escritas a 'modelo_benchmark.csv'")
    return pipe


def main():
    print("=== Arrancando pipeline ===")

    df = load_competition_datasets(
        COMPETITION_PATH,
        sample_frac=0.8,   #    Si subís a 1.0 entrenás con todo. No necesariamente dé mejor el resultado.
        random_state=1234
    )
    df = cast_column_types(df)

    #  Predictores 
    df, cat_cols, num_cols = build_features(df)

    #   Split real (train vs test Kaggle)
    train_df, test_df = split_train_test(df)

    # Entrenar y predecir
    _ = train_and_predict(train_df, test_df, cat_cols, num_cols)

    print("=== Pipeline listo ===")

'''
    NOTA: con sample_frac = 0.8 da 0.76666
        con sample_frac = 0.4 da 0.76711
'''

if __name__ == "__main__":
    main()
