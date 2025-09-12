
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

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
    s = df.groupby("username", observed=True)["target"].shift(1)
    cnt_nonnull = s.notna().groupby(df["username"], observed=True).cumsum()
    sum_hist = s.fillna(0).groupby(df["username"], observed=True).cumsum()
    df["user_forward_rate_hist"] = (sum_hist / cnt_nonnull).astype("float32")

    global_rate = df.loc[~df["is_test"], "target"].mean()
    df["user_forward_rate_hist"] = df["user_forward_rate_hist"].fillna(global_rate).astype("float32")
    
    df["log1p_secs_since_prev"] = np.log1p(df["secs_since_prev"]).astype("float32")

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)
  
    if "master_metadata_album_artist_name" in df.columns:
        freq_artist = df.loc[~df["is_test"], "master_metadata_album_artist_name"].value_counts()
        df["artist_freq"] = df["master_metadata_album_artist_name"].map(freq_artist).fillna(1).astype("float32")
    else:
        df["artist_freq"] = 1.0


    # === Comportamiento histórico del usuario (sin fuga) ===
    # target anterior del mismo usuario
    prev_t = df.groupby("username", observed=True)["target"].shift(1).astype("float32")

    # medias móviles de las últimas K acciones del usuario
    for k in (3, 5, 10):
        df[f"user_roll_mean_{k}"] = (
                prev_t.groupby(df["username"], observed=True).rolling(k, min_periods=1).mean().reset_index(level=0, drop=True)
                ).astype("float32")

    # tiempo desde el último *forward* del mismo usuario
    last_forward_ts = df["ts_unix"].where(df["target"] == 1)
    last_forward_ts = last_forward_ts.groupby(df["username"], observed=True).shift(1)
    last_forward_cummax = last_forward_ts.groupby(df["username"], observed=True).cummax()
    df["secs_since_last_forward"] = (df["ts_unix"] - last_forward_cummax).clip(lower=0).fillna(df["secs_since_prev"]).astype("float32")
    df["log1p_secs_since_last_forward"] = np.log1p(df["secs_since_last_forward"]).astype("float32")


    # === Afinidad usuario–artista (sin fuga) ===
    if "master_metadata_album_artist_name" in df.columns:
        u = df["username"]
        a = df["master_metadata_album_artist_name"]

        # historial de ese par user-artist (antes del evento actual)
        ua_prev = df["target"].groupby([u, a], observed=True).shift(1)

        cnt_ua = ua_prev.notna().groupby([u, a], observed=True).cumsum()
        sum_ua = ua_prev.fillna(0).groupby([u, a], observed=True).cumsum()
        ua_rate = (sum_ua / cnt_ua).astype("float32")

        # backoff bayesiano hacia:
        #   - tasa del usuario
        #   - tasa global
        user_rate = df["user_forward_rate_hist"]
        global_rate = float(df.loc[~df["is_test"], "target"].mean())

        # peso según cuánta historia tenemos del par
        w = (cnt_ua / (cnt_ua + 20)).astype("float32")  # 20 = suavizado
        df["user_artist_rate_hist"] = (w * ua_rate + (1 - w) * (0.7 * user_rate + 0.3 * global_rate)).fillna(user_rate).astype("float32")

        # cuántas veces escuchó ese user-artist
        df["user_artist_cnt"] = cnt_ua.fillna(0).astype("float32")
    else:
        df["user_artist_rate_hist"] = df["user_forward_rate_hist"]
        df["user_artist_cnt"] = 0.0


    # === Sesiones por usuario (corte a 30 minutos) ===
    gap = df["secs_since_prev"].fillna(0)
    new_session = (gap > 1800).astype("int8")
    df["session_id"] = (new_session.groupby(df["username"], observed=True).cumsum()).astype("int32")

    # posición y tamaño de sesión
    df["pos_in_session"] = df.groupby(["username", "session_id"], observed=True).cumcount() + 1
    sess_len = df.groupby(["username", "session_id"], observed=True)["obs_id"].transform("size")
    df["session_len"] = sess_len.astype("int32")
    df["is_first_in_session"] = (df["pos_in_session"] == 1).astype("int8")
    df["is_long_session"] = (df["session_len"] >= 10).astype("int8")
    
    

    cat_cols = []
    
    for c in ["platform", "conn_country", "shuffle", "offline", "incognito_mode"]:
        if c in df.columns:
            cat_cols.append(c)

    
    # reemplazá tu num_cols por algo así:
    num_cols = [
        "user_order", "is_weekend",
        "secs_since_prev", "log1p_secs_since_prev",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "user_forward_rate_hist",
        "user_roll_mean_3","user_roll_mean_5","user_roll_mean_10",
        "secs_since_last_forward","log1p_secs_since_last_forward",
        "user_artist_rate_hist","user_artist_cnt",
        "pos_in_session","session_len","is_first_in_session","is_long_session",
        "artist_freq"  # si la agregaste previamente
    ]

    num_cols = [c for c in num_cols if c in df.columns]

    keep = ["obs_id", "username", "ts", "is_test", "target"] + cat_cols + num_cols
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


# ------------------------------------------------------------------
#   Split temporal para AUC de validación + early stopping
# ------------------------------------------------------------------

def temporal_valid_split(train_df, quantile=0.9):
    cutoff = train_df["ts"].quantile(quantile)
    trn = train_df.loc[train_df["ts"] < cutoff].copy()
    val = train_df.loc[train_df["ts"] >= cutoff].copy()
    print(f"  → Split temporal: train={trn.shape[0]}  valid={val.shape[0]} (cutoff={cutoff})")
    return trn, val


def train_and_predict(train_df, test_df, cat_cols, num_cols):
    '''
    Pipeline: OHE (categóricas low-card) + XGBoost; predicción de probas para test.
    También imprime AUC en validación temporal.
    '''

    X_train = train_df[cat_cols + num_cols]
    y_train = train_df["target"].astype(int)
    X_test  = test_df[cat_cols + num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    #   validación temporal
    trn, val = temporal_valid_split(train_df, quantile=0.9)
    X_trn = trn[cat_cols + num_cols]
    y_trn = trn["target"].astype(int)
    X_val = val[cat_cols + num_cols]
    y_val = val["target"].astype(int)

    #   ⚠️ IMPORTANTE: el eval_set de XGBoost NO pasa por el ColumnTransformer del Pipeline.
    #   Por eso transformamos MANUALMENTE train/val/test con 'pre' y entrenamos el XGB solo.
    print("Ajustando preprocesador (OHE)...")
    X_trn_t = pre.fit_transform(X_trn, y_trn)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    
    # después de crear X_trn, y_trn, X_val, y_val (ya lo tenés así)
    pos_rate = float(y_trn.mean())
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=4500,      # más árboles
        learning_rate=0.03,     # LR más chica
        max_depth=4,            # árboles más bajos generalizan mejor
        min_child_weight=3,     # regulariza
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.1,
        gamma=0.0,
        tree_method="hist",
        max_bin=512,            # hist más fino
        eval_metric="auc",
        random_state=1234,
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=100,   # más paciencia para LR más chica
        scale_pos_weight=spw,        # ⬅️ clave si hay desbalance

            )

    print("Entrenando modelo (XGBoost) con early stopping...")
    clf.fit(
        X_trn_t,
        y_trn,
        eval_set=[(X_val_t, y_val)],
    )

    #   AUC de validación (para que se imprima y puedas decidir si cambiar o no)
    val_pred = clf.predict_proba(X_val_t)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"  → AUC valid (temporal 10%): {val_auc:.5f}")

    print("Generando predicciones para el test set...")
    preds_proba = clf.predict_proba(X_test_t)[:, 1]
    out = pd.DataFrame({"obs_id": test_df["obs_id"].astype(int), "pred_proba": preds_proba})
    out.to_csv("modelo_xgb.csv", index=False)
    print("  → Predicciones escritas a 'modelo_xgb.csv'")

    #   Devolvemos un dict liviano con objetos útiles por si luego querés reusar
    return {"pre": pre, "clf": clf}


def main():
    print("=== Arrancando pipeline (XGBoost sin grid search) ===")

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
