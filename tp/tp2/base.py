import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)

COMPETITION_PATH = os.path.expanduser("~/td-vi/tp/tp2/competition_data")
TRAIN_DATA_PATH = "train_data.txt"
TEST_DATA_PATH  = "test_data.txt"

# ------------------------------------------------------------------
#   Carga de datasets
# ------------------------------------------------------------------

def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    print("Cargando datasets de competencia desde:", data_dir)
    train_file = os.path.join(data_dir, TRAIN_DATA_PATH)
    test_file  = os.path.join(data_dir, TEST_DATA_PATH)

    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → DataFrame concatenado: {combined.shape[0]} rows")
    return combined


def cast_column_types(df):
    print("Casteando tipos de columnas y parseando campos de datetime...")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "offline_timestamp" in df.columns:
        df["offline_timestamp"] = pd.to_datetime(
            df["offline_timestamp"], unit="s", errors="coerce", utc=True
        )
    for col in ["platform", "conn_country", "reason_end",
                "master_metadata_track_name", "master_metadata_album_artist_name",
                "master_metadata_album_album_name", "username", "ip_addr"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in ["shuffle", "offline", "incognito_mode"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    if "obs_id" in df.columns:
        df["obs_id"] = pd.to_numeric(df["obs_id"], errors="coerce").astype("Int64")
    print("  → Tipos de columna casteados.")
    return df

# ------------------------------------------------------------------
#   Features
# ------------------------------------------------------------------

def build_features(df):
    print("Armando features...")
    df = df.sort_values(["username", "ts"], kind="mergesort")
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1

    # calendario
    df["hour"] = df["ts"].dt.hour
    df["dow"]  = df["ts"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)

    # gaps temporales
    df["ts_unix"] = df["ts"].astype("int64") // 10**9
    last_ts = df.groupby("username", observed=True)["ts_unix"].shift(1)
    df["secs_since_prev"] = (df["ts_unix"] - last_ts).clip(lower=0)
    df["secs_since_prev"] = df["secs_since_prev"].fillna(df["secs_since_prev"].median())
    df["log1p_secs_since_prev"] = np.log1p(df["secs_since_prev"]).astype("float32")

    # target y máscara
    df["target"] = (df["reason_end"] == "fwdbtn").astype("Int8")
    df["is_test"] = df["reason_end"].isna()

    # historial usuario
    s = df.groupby("username", observed=True)["target"].shift(1)
    cnt_hist = s.notna().groupby(df["username"], observed=True).cumsum()
    sum_hist = s.fillna(0).groupby(df["username"], observed=True).cumsum()
    user_rate_raw = (sum_hist / cnt_hist).astype("float32")
    global_rate = float(df.loc[~df["is_test"], "target"].mean())
    w_user = (cnt_hist / (cnt_hist + 100)).astype("float32")
    df["user_forward_rate_hist"] = (w_user * user_rate_raw + (1 - w_user) * global_rate)\
                                      .fillna(global_rate).astype("float32")

    # rolling recientes
    prev_t = df.groupby("username", observed=True)["target"].shift(1).astype("float32")
    for k in (3, 5, 10):
        df[f"user_roll_mean_{k}"] = (
            prev_t.groupby(df["username"], observed=True)
                  .rolling(k, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
        ).astype("float32")

    # tiempo desde último forward
    last_fwd_ts = df["ts_unix"].where(df["target"] == 1)
    last_fwd_ts = last_fwd_ts.groupby(df["username"], observed=True).shift(1)
    last_fwd_cum = last_fwd_ts.groupby(df["username"], observed=True).cummax()
    df["secs_since_last_forward"] = (df["ts_unix"] - last_fwd_cum).clip(lower=0)\
                                      .fillna(df["secs_since_prev"]).astype("float32")
    df["log1p_secs_since_last_forward"] = np.log1p(df["secs_since_last_forward"]).astype("float32")

    # afinidad user–artist
    if "master_metadata_album_artist_name" in df.columns:
        u = df["username"]
        a = df["master_metadata_album_artist_name"]
        ua_prev = df["target"].groupby([u, a], observed=True).shift(1)
        cnt_ua = ua_prev.notna().groupby([u, a], observed=True).cumsum()
        sum_ua = ua_prev.fillna(0).groupby([u, a], observed=True).cumsum()
        ua_rate = (sum_ua / cnt_ua).astype("float32")
        w_ua = (cnt_ua / (cnt_ua + 20)).astype("float32")
        df["user_artist_rate_hist"] = (
            w_ua * ua_rate + (1 - w_ua) * (0.7 * df["user_forward_rate_hist"] + 0.3 * global_rate)
        ).fillna(df["user_forward_rate_hist"]).astype("float32")
        df["user_artist_cnt"] = cnt_ua.fillna(0).astype("float32")
    else:
        df["user_artist_rate_hist"] = df["user_forward_rate_hist"]
        df["user_artist_cnt"] = 0.0

    # popularidad artista
    if "master_metadata_album_artist_name" in df.columns:
        freq_artist = df.loc[~df["is_test"], "master_metadata_album_artist_name"].value_counts()
        df["artist_freq"] = df["master_metadata_album_artist_name"].map(freq_artist)\
                                .fillna(1).astype("float32")
    else:
        df["artist_freq"] = 1.0

    # popularidad artista ya calculada arriba

    # sesiones mínimas
    gap = df["secs_since_prev"].fillna(0)
    new_session = (gap > 1800).astype("int8")
    df["session_id"] = new_session.groupby(df["username"], observed=True).cumsum().astype("int32")
    df["pos_in_session"] = df.groupby(["username", "session_id"], observed=True).cumcount() + 1
    df["is_first_in_session"] = (df["pos_in_session"] == 1).astype("int8")

    cat_cols = [c for c in ["platform", "conn_country", "shuffle", "offline", "incognito_mode"] if c in df.columns]
    num_cols = [
        "user_order","is_weekend",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "secs_since_prev","log1p_secs_since_prev",
        "user_forward_rate_hist",
        "user_roll_mean_3","user_roll_mean_5","user_roll_mean_10",
        "secs_since_last_forward","log1p_secs_since_last_forward",
        "user_artist_rate_hist","user_artist_cnt",
        "artist_freq",
        "pos_in_session","is_first_in_session",
    ]
    keep = ["obs_id","username","ts","is_test","target"] + cat_cols + num_cols
    df = df[keep].copy()
    print("  → Features listas.")
    return df, cat_cols, num_cols

# ------------------------------------------------------------------
#   Splits
# ------------------------------------------------------------------

def split_train_test(df):
    print("Splitteando datos en train/test sets...")
    train_df = df.loc[~df["is_test"]].copy()
    test_df  = df.loc[df["is_test"]].copy()
    print(f"  → Training set: {train_df.shape[0]} rows")
    print(f"  → Test set:     {test_df.shape[0]} rows")
    return train_df, test_df

def temporal_valid_split(train_df, quantile=0.9):
    cutoff = train_df["ts"].quantile(quantile)
    trn = train_df.loc[train_df["ts"] < cutoff].copy()
    val = train_df.loc[train_df["ts"] >= cutoff].copy()
    print(f"  → Split temporal: train={trn.shape[0]}  valid={val.shape[0]} (cutoff={cutoff})")
    return trn, val

# ------------------------------------------------------------------
#   Train + Predict
# ------------------------------------------------------------------

def train_and_predict(train_df, test_df, cat_cols, num_cols):
    X_train = train_df[cat_cols + num_cols]; y_train = train_df["target"].astype(int)
    X_test  = test_df[cat_cols + num_cols]

    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat_cols),
                      ("num","passthrough",num_cols)],
        remainder="drop", sparse_threshold=1.0,
    )

    trn, val = temporal_valid_split(train_df, quantile=0.9)
    X_trn = trn[cat_cols + num_cols]; y_trn = trn["target"].astype(int)
    X_val = val[cat_cols + num_cols]; y_val = val["target"].astype(int)

    print("Ajustando preprocesador (OHE)...")
    X_trn_t = pre.fit_transform(X_trn, y_trn)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    pos_rate = float(y_trn.mean())
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    pos_rate = float(y_trn.mean())
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.1,
        gamma=0.0,
        tree_method="hist",
        eval_metric="auc",
        random_state=1234,
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=100,
        scale_pos_weight=spw,
    )

    print("Entrenando modelo (XGBoost baseline)...")
    clf.fit(X_trn_t, y_trn, eval_set=[(X_val_t, y_val)])

    val_pred = clf.predict_proba(X_val_t)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"  → AUC valid (temporal 10%): {val_auc:.5f}")

    preds_proba = clf.predict_proba(X_test_t)[:, 1]
    out = pd.DataFrame({"obs_id": test_df["obs_id"].astype(int), "pred_proba": preds_proba})
    out.to_csv("test.csv", index=False)
    print("  → Predicciones escritas a 'modelo_xgb.csv'")
    return {"pre":pre,"clf":clf}

def main():
    print("=== Arrancando pipeline (XGBoost mejorado con bagging) ===")
    df = load_competition_datasets(COMPETITION_PATH, sample_frac=1.0, random_state=1234)
    df = cast_column_types(df)
    df, cat_cols, num_cols = build_features(df)
    train_df, test_df = split_train_test(df)
    _ = train_and_predict(train_df, test_df, cat_cols, num_cols)
    print("=== Pipeline listo ===")

if __name__ == "__main__":
    main()

