# ============================================================
# (Colab) ONE-SHOT (FINAL):
#   Featurize(THV, FULL schema) -> Train DET -> Train CLF -> Save ONE unified_model.pkl
#
# ✅ 핵심 목표(네 요구 반영)
# 1) 피처링 코드는 "하나"만 사용 (어떤 데이터든 동일한 FULL 컬럼 생성)
# 2) DET/CLF는 "같은 피처 DF"에서 각자 필요한 컬럼만 골라서 학습
#    - DET: BASE_FEATURE_COLS
#    - CLF: FULL_FEATURE_COLS (BASE + 추가2개)
# 3) phase 컬럼은 "사용 안 함"
#    - 대신 시간(t_end) 기반으로 DET를 구간별(STARTUP/STEADY/SHUTDOWN)로 나눠 학습/임계값 적용 가능
#    - 실무 데이터에 phase 컬럼 없어도 동작
# 4) 분류 학습은 옵션으로 onset~recovery_start 구간만 쓰도록 필터(ON/OFF 가능)
#
# - rail_speed 컬럼명 고정: "rail_speed" (피처 계산에는 사용 X, 최종 저장에서도 제거)
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

import xgboost as xgb


# ============================================================
# [1] Featurizer (THV only, FULL schema)
#   - 어떤 RAW든 동일한 FULL 피처 컬럼 생성
#   - passthrough: 윈도우 끝 row 그대로(단, 원시센서 4개는 제거)
#   - 항상 생성되는 피처:
#     BASE: THV stats(8개) * 3 + corr_vib_temp
#     EXTRA: temp_hum_slope_diff, vib_spike_strength
# ============================================================

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def linear_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x.astype(float), y.astype(float), 1)[0])

def window_stats(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
        "delta": float(x[-1] - x[0]),
        "slope": float(linear_slope(x)),
        "diff_std": float(np.std(np.diff(x))) if len(x) >= 2 else 0.0,
    }

def build_base_feature_cols_thv() -> List[str]:
    sensors = ["temperature", "humidity", "vibration"]
    stats = ["mean", "std", "min", "max", "range", "delta", "slope", "diff_std"]
    cols = []
    for s in sensors:
        for st in stats:
            cols.append(f"{s}_{st}")
    cols.append("corr_vib_temp")
    return cols

BASE_FEATURE_COLS = build_base_feature_cols_thv()
EXTRA_FEATURE_COLS = ["temp_hum_slope_diff", "vib_spike_strength"]
FULL_FEATURE_COLS = BASE_FEATURE_COLS + EXTRA_FEATURE_COLS

def extract_X(df_feat: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"FEAT 피처 컬럼 누락(스펙 불일치): {missing}")
    return df_feat[feature_cols].copy()

def featurize_raw_to_windows_full(
    df_raw: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    win_size: int,
    stride: int,
    onset_col: str,
    recovery_col: str,
    apply_segment_filter: bool,
    drop_raw_cols: List[str],   # ["temperature","humidity","vibration","rail_speed"]
) -> pd.DataFrame:
    # 필수 컬럼 체크 (THV + id/time)
    required = [id_col, time_col, "temperature", "humidity", "vibration"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"RAW 필수 컬럼 누락: {missing}")

    df_raw = df_raw.sort_values([id_col, time_col]).reset_index(drop=True)
    rows: List[Dict[str, Any]] = []

    for sid, g in df_raw.groupby(id_col, sort=True):
        g = g.sort_values(time_col).reset_index(drop=True)
        n = len(g)
        if n < win_size:
            continue

        onset = float(g[onset_col].iloc[0]) if onset_col in g.columns else None
        recovery = float(g[recovery_col].iloc[0]) if recovery_col in g.columns else None

        for start in range(0, n - win_size + 1, stride):
            end = start + win_size
            win = g.iloc[start:end]
            t = float(win[time_col].iloc[-1])


            # (옵션) 분류 학습용: onset~recovery만 쓰기
            if apply_segment_filter:
                if onset is None or onset < 0:
                    continue
                if t < onset + win_size:
                    continue
                if (recovery is not None) and (recovery >= 0) and (t > recovery):
                    continue

            out: Dict[str, Any] = {}

            # (1) passthrough: 윈도우 끝 row 기준(센서 raw 4개 제거)
            end_row = win.iloc[-1].to_dict()
            for k, v in end_row.items():
                if k in drop_raw_cols:
                    continue
                out[k] = v

            # (2) 메타: id + t (항상 저장)
            out[id_col] = int(win[id_col].iloc[0])
            out["t"] = t

            # (3) THV arrays
            temp = win["temperature"].to_numpy(dtype=float)
            hum  = win["humidity"].to_numpy(dtype=float)
            vib  = win["vibration"].to_numpy(dtype=float)

            # (4) THV stats (BASE)
            for name, arr in [("temperature", temp), ("humidity", hum), ("vibration", vib)]:
                st = window_stats(arr)
                for kk, vv in st.items():
                    out[f"{name}_{kk}"] = vv

            # (5) corr (BASE)
            out["corr_vib_temp"] = safe_corr(vib, temp)

            # (6) EXTRA (항상 생성)
            # temp_hum_slope_diff = humidity_slope - temperature_slope
            out["temp_hum_slope_diff"] = float(out["humidity_slope"] - out["temperature_slope"])
            # vib_spike_strength = vibration_max - vibration_mean
            out["vib_spike_strength"]  = float(out["vibration_max"] - out["vibration_mean"])

            rows.append(out)

    df_feat = pd.DataFrame(rows)

    # 스펙 체크(항상 FULL이 있어야 함)
    if len(df_feat) > 0:
        _ = df_feat[FULL_FEATURE_COLS]

    return df_feat


# ============================================================
# [2] Detection (IsolationForest) Train
#   - phase 컬럼 사용 안 함
#   - 대신 시간(t_end) 기반으로 구간(STARTUP/STEADY/SHUTDOWN) 나눠 학습/임계값 가능
# ============================================================

def make_detector_pipeline(
    n_estimators: int,
    max_samples,
    contamination,
    random_state: int
) -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

def compute_anomaly_score(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    imputer = pipe.named_steps["imputer"]
    scaler  = pipe.named_steps["scaler"]
    iforest = pipe.named_steps["iforest"]
    X_imp = imputer.transform(X)
    X_scl = scaler.transform(X_imp)
    s = iforest.score_samples(X_scl)
    return (-s).astype(float)

def _assign_time_segment(t: float, segment_rules: List[Tuple[str, float, float]]) -> str:
    """
    segment_rules: [(name, start_inclusive, end_inclusive), ...]
    """
    for name, a, b in segment_rules:
        if (t >= a) and (t <= b):
            return str(name)
    return "ALL"

def train_detector_time_segment(
    df_feat_train: pd.DataFrame,
    *,
    feature_cols: List[str],               # DET는 보통 BASE_FEATURE_COLS
    time_segment_enabled: bool,
    segment_rules: List[Tuple[str, float, float]],
    warning_q: float,
    scale_by_segment: Dict[str, float],
    n_estimators: int,
    max_samples,
    contamination,
    random_state: int,
) -> Dict[str, Any]:
    if len(df_feat_train) == 0:
        raise ValueError("DET 학습 피처가 비었습니다. 데이터/윈도우 설정을 확인하세요.")

    X_all = extract_X(df_feat_train, feature_cols)
    models: Dict[str, Any] = {}
    thresholds: Dict[str, Any] = {}

    if time_segment_enabled:
        if "t" not in df_feat_train.columns:
            raise ValueError("시간 구간 DET를 쓰려면 FEAT에 't' 컬럼이 필요합니다.")

        seg_series = df_feat_train["t"].astype(float).apply(lambda t: _assign_time_segment(t, segment_rules))
        segments = sorted(seg_series.unique().tolist())

        for seg in segments:
            idx = (seg_series == seg).values
            X = X_all.loc[idx].copy()
            if len(X) < 10:
                # 너무 적으면 학습/threshold 불안정 → 스킵하고 ALL로 fallback 가능
                continue

            pipe = make_detector_pipeline(n_estimators, max_samples, contamination, random_state)
            pipe.fit(X)

            scores = compute_anomaly_score(pipe, X)
            base_warn = float(np.quantile(scores, warning_q))
            scale = float(scale_by_segment.get(str(seg), 1.0))
            final_warn = float(base_warn * scale)

            models[str(seg)] = pipe
            thresholds[str(seg)] = {
                "warning_q": warning_q,
                "base_warning_threshold": base_warn,
                "scale": scale,
                "warning_threshold": final_warn,
                "n_samples": int(len(X)),
            }

        # 안전장치: 세그먼트 모델이 하나도 못 만들어졌으면 ALL로 학습
        if len(models) == 0:
            time_segment_enabled = False

    if not time_segment_enabled:
        pipe = make_detector_pipeline(n_estimators, max_samples, contamination, random_state)
        pipe.fit(X_all)

        scores = compute_anomaly_score(pipe, X_all)
        base_warn = float(np.quantile(scores, warning_q))

        models["ALL"] = pipe
        thresholds["ALL"] = {
            "warning_q": warning_q,
            "base_warning_threshold": base_warn,
            "scale": 1.0,
            "warning_threshold": base_warn,
            "n_samples": int(len(X_all)),
        }

    return {
        "time_segment_enabled": bool(time_segment_enabled),
        "segment_rules": segment_rules,
        "models": models,
        "thresholds": thresholds,
        "params": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "random_state": random_state,
        }
    }


# ============================================================
# [3] Classification (XGBoost) Train
#   - CLF는 보통 FULL_FEATURE_COLS (BASE + EXTRA)
# ============================================================

def train_classifier_xgb(
    df_feat_train: pd.DataFrame,
    df_feat_test: pd.DataFrame,
    *,
    feature_cols: List[str],   # CLF는 보통 FULL_FEATURE_COLS
    target_col: str,
    xgb_params: Dict[str, Any],
) -> Dict[str, Any]:
    if len(df_feat_train) == 0 or len(df_feat_test) == 0:
        raise ValueError("CLF 학습/테스트 피처가 비었습니다. 구간필터/윈도우 설정을 확인하세요.")

    if target_col not in df_feat_train.columns:
        raise ValueError(f"TRAIN FEAT에 target_col({target_col})이 없습니다.")
    if target_col not in df_feat_test.columns:
        raise ValueError(f"TEST FEAT에 target_col({target_col})이 없습니다.")

    X_train = extract_X(df_feat_train, feature_cols)
    X_test  = extract_X(df_feat_test, feature_cols)

    y_train = df_feat_train[target_col].astype(str)
    y_test  = df_feat_test[target_col].astype(str)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    params = dict(xgb_params)
    params["num_class"] = int(len(le.classes_))

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train_enc, eval_set=[(X_test, y_test_enc)], verbose=False)

    return {
        "model": model,
        "label_encoder": le,
        "target_col": target_col,
        "classes_": list(le.classes_),
        "xgb_params": params,
        "feature_cols": feature_cols,
    }


# ============================================================
# [4] UnifiedModel (단일 파일에 저장될 객체)
#   - 실무용: raw 들어오면 full featurize -> DET 컬럼 선택 -> CLF 컬럼 선택
#   - phase 컬럼 없음(사용 안 함)
#   - DET는 시간 구간 기반으로 모델/threshold 선택 가능
# ============================================================

@dataclass
class UnifiedConfig:
    id_col: str = "session_id"
    time_col: str = "t"
    speed_col: str = "rail_speed"   # 고정(사용 안 함, 제거만)
    win_size: int = 10
    stride: int = 1
    onset_col: str = "onset"
    recovery_col: str = "recovery_start"
    drop_raw_cols: List[str] = None

class UnifiedModel:
    def __init__(
        self,
        cfg: UnifiedConfig,
        *,
        det_feature_cols: List[str],
        clf_feature_cols: List[str],
        detector: Dict[str, Any],
        classifier: Dict[str, Any],
    ):
        self.cfg = cfg
        self.det_feature_cols = det_feature_cols
        self.clf_feature_cols = clf_feature_cols
        self.detector = detector
        self.classifier = classifier

    def featurize(self, df_raw: pd.DataFrame, *, apply_segment_filter: bool = False) -> pd.DataFrame:
        return featurize_raw_to_windows_full(
            df_raw,
            id_col=self.cfg.id_col,
            time_col=self.cfg.time_col,
            win_size=self.cfg.win_size,
            stride=self.cfg.stride,
            onset_col=self.cfg.onset_col,
            recovery_col=self.cfg.recovery_col,
            apply_segment_filter=apply_segment_filter,
            drop_raw_cols=self.cfg.drop_raw_cols,
        )

    def predict_feat(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        if len(df_feat) == 0:
            raise ValueError("피처 DF가 비었습니다.")

        # -------------------------
        # (A) DET
        # -------------------------
        X_det = extract_X(df_feat, self.det_feature_cols)

        det = self.detector
        models = det["models"]
        ths = det["thresholds"]
        time_seg = det["time_segment_enabled"]
        seg_rules = det["segment_rules"]

        scores = np.zeros(len(df_feat), dtype=float)
        pred_warn = np.zeros(len(df_feat), dtype=int)

        if time_seg and ("t" in df_feat.columns):
            seg_series = df_feat["t"].astype(float).apply(lambda t: _assign_time_segment(t, seg_rules))
            for seg in seg_series.unique():
                idx = (seg_series == seg).values
                key = str(seg) if str(seg) in models else ("ALL" if "ALL" in models else list(models.keys())[0])
                pipe = models[key]
                thv = ths[key]["warning_threshold"]
                sc = compute_anomaly_score(pipe, X_det.loc[idx])
                scores[idx] = sc
                pred_warn[idx] = (sc >= thv).astype(int)
        else:
            key = "ALL" if "ALL" in models else list(models.keys())[0]
            pipe = models[key]
            thv = ths[key]["warning_threshold"]
            sc = compute_anomaly_score(pipe, X_det)
            scores[:] = sc
            pred_warn[:] = (sc >= thv).astype(int)

        # -------------------------
        # (B) CLF
        #   - 실무에서는 보통 pred_warn==1인 구간만 분류를 쓰는 걸 추천하지만,
        #     여기서는 전체에 대해 proba/pred_label을 계산함
        # -------------------------
        X_clf = extract_X(df_feat, self.clf_feature_cols)

        clf = self.classifier["model"]
        le = self.classifier["label_encoder"]

        proba = clf.predict_proba(X_clf)
        pred_enc = np.argmax(proba, axis=1)
        pred_label = le.inverse_transform(pred_enc)

        out = df_feat.copy()
        out["if_score"] = scores
        out["pred_warning"] = pred_warn
        out["pred_label"] = pred_label

        for ci, cls in enumerate(le.classes_):
            out[f"proba_{cls}"] = proba[:, ci]

        return out

    def predict_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df_feat = self.featurize(df_raw, apply_segment_filter=False)
        if len(df_feat) == 0:
            raise ValueError("피처링 결과가 비었습니다. WIN_SIZE/데이터 길이/session_id 확인.")
        return self.predict_feat(df_feat)


# ============================================================
# ================== [수정하기 좋은 설정 구역] ==================
# ============================================================

# ---- 입력 선택: FEAT를 쓰면 featurize 단계 스킵됨 ----
USE_DET_FEAT_DIRECT = False
USE_CLF_FEAT_DIRECT = False

# detection data (정상 학습용)
DET_RAW_TRAIN  = "/content/data/raw_sensor_train.csv"
DET_FEAT_TRAIN = "/content/data/feat_sensor_train.csv"

# classification data (이상 유형 분류용)
CLF_RAW_TRAIN  = "/content/data/raw_anom_train.csv"
CLF_RAW_TEST   = "/content/data/raw_anom_test.csv"
CLF_FEAT_TRAIN = "/content/data/feat_anom_train.csv"
CLF_FEAT_TEST  = "/content/data/feat_anom_test.csv"

# save
OUT_UNIFIED = "/content/models/unified_model.pkl"

# columns
ID_COL = "session_id"
TIME_COL = "t"

# window
WIN_SIZE = 10
STRIDE = 1

# 분류 학습에서만 onset~recovery_start 구간만 쓰고 싶으면 True
# (피처링 코드는 하나지만, 학습 데이터 만들 때 필터를 적용하는 옵션)
CLF_APPLY_SEGMENT_FILTER = False
ONSET_COL = "onset"
RECOVERY_COL = "recovery_start"

# 최종 저장에서 제거할 원시 센서 컬럼(4개)
DROP_RAW_COLS = ["temperature", "humidity", "vibration", "rail_speed"]

# ----------------------------
# DET 학습 설정
# - phase 컬럼 사용 안 함
# - 시간(t_end)으로 STARTUP/STEADY/SHUTDOWN 구간 나누기(원하면 True)
# ----------------------------
DET_TIME_SEGMENT_ENABLED = True

# 네가 말한 시간 구간(초) 그대로
# 0~119 STARTUP, 120~479 STEADY, 480~600 SHUTDOWN
DET_SEGMENT_RULES = [
    ("STARTUP",  0.0, 119.0),
    ("STEADY", 120.0, 479.0),
    ("SHUTDOWN", 480.0, 600.0),
]

DET_WARNING_Q = 0.95
DET_SCALE_BY_SEGMENT = {"STARTUP": 1.15, "STEADY": 1.0, "SHUTDOWN": 1.15, "ALL": 1.0}
DET_N_ESTIMATORS = 300
DET_MAX_SAMPLES = "auto"
DET_CONTAMINATION = "auto"
DET_RANDOM_STATE = 42

# ----------------------------
# CLF 학습 설정
# ----------------------------
CLF_TARGET_COL = "label"
CLF_XGB_PARAMS = dict(
    objective="multi:softprob",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
)

# ----------------------------
# 모델별 feature cols 선택 (중요)
# ----------------------------
DET_FEATURE_COLS = BASE_FEATURE_COLS
CLF_FEATURE_COLS = FULL_FEATURE_COLS

if __name__ == "__main__":
    # ============================================================
    # [RUN] ONE-SHOT
    # ============================================================

    # (0) 먼저 FEAT를 직접 쓸 경우 스펙 체크용 로더
    def _load_and_check_feat(path: str, required_cols: List[str]) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{os.path.basename(path)}] FEAT에 필요한 컬럼이 없습니다: {missing}")
        return df

    # (1) DET train feat 준비
    if USE_DET_FEAT_DIRECT:
        df_det_train = _load_and_check_feat(DET_FEAT_TRAIN, FULL_FEATURE_COLS + ["t"])
    else:
        df_raw_det = pd.read_csv(DET_RAW_TRAIN)
        df_det_train = featurize_raw_to_windows_full(
            df_raw_det,
            id_col=ID_COL, time_col=TIME_COL,
            win_size=WIN_SIZE, stride=STRIDE,
            onset_col=ONSET_COL, recovery_col=RECOVERY_COL,
            apply_segment_filter=False,                 # ✅ DET는 전체 구간(정상 분포 학습)
            drop_raw_cols=DROP_RAW_COLS,
        )
        os.makedirs(os.path.dirname(DET_FEAT_TRAIN), exist_ok=True)
        df_det_train.to_csv(DET_FEAT_TRAIN, index=False)
        print(f"[OK] DET feat saved -> {DET_FEAT_TRAIN} shape={df_det_train.shape}")

    # (2) CLF train/test feat 준비
    if USE_CLF_FEAT_DIRECT:
        df_clf_train = _load_and_check_feat(CLF_FEAT_TRAIN, FULL_FEATURE_COLS)
        df_clf_test  = _load_and_check_feat(CLF_FEAT_TEST,  FULL_FEATURE_COLS)
    else:
        df_raw_train = pd.read_csv(CLF_RAW_TRAIN)
        df_raw_test  = pd.read_csv(CLF_RAW_TEST)

        df_clf_train = featurize_raw_to_windows_full(
            df_raw_train,
            id_col=ID_COL, time_col=TIME_COL,
            win_size=WIN_SIZE, stride=STRIDE,
            onset_col=ONSET_COL, recovery_col=RECOVERY_COL,
            apply_segment_filter=CLF_APPLY_SEGMENT_FILTER,   # ✅ 분류 학습용 옵션
            drop_raw_cols=DROP_RAW_COLS,
        )
        df_clf_test = featurize_raw_to_windows_full(
            df_raw_test,
            id_col=ID_COL, time_col=TIME_COL,
            win_size=WIN_SIZE, stride=STRIDE,
            onset_col=ONSET_COL, recovery_col=RECOVERY_COL,
            apply_segment_filter=CLF_APPLY_SEGMENT_FILTER,   # ✅ 분류 학습용 옵션
            drop_raw_cols=DROP_RAW_COLS,
        )

        os.makedirs(os.path.dirname(CLF_FEAT_TRAIN), exist_ok=True)
        df_clf_train.to_csv(CLF_FEAT_TRAIN, index=False)
        df_clf_test.to_csv(CLF_FEAT_TEST, index=False)
        print(f"[OK] CLF feat saved -> {CLF_FEAT_TRAIN} shape={df_clf_train.shape}")
        print(f"[OK] CLF feat saved -> {CLF_FEAT_TEST} shape={df_clf_test.shape}")

    # (3) Train detector (시간구간 기반)
    detector_obj = train_detector_time_segment(
        df_det_train,
        feature_cols=DET_FEATURE_COLS,
        time_segment_enabled=DET_TIME_SEGMENT_ENABLED,
        segment_rules=DET_SEGMENT_RULES,
        warning_q=DET_WARNING_Q,
        scale_by_segment=DET_SCALE_BY_SEGMENT,
        n_estimators=DET_N_ESTIMATORS,
        max_samples=DET_MAX_SAMPLES,
        contamination=DET_CONTAMINATION,
        random_state=DET_RANDOM_STATE,
    )

    # (4) Train classifier (FULL 피처)
    classifier_obj = train_classifier_xgb(
        df_clf_train,
        df_clf_test,
        feature_cols=CLF_FEATURE_COLS,
        target_col=CLF_TARGET_COL,
        xgb_params=CLF_XGB_PARAMS,
    )
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # 1) 테스트셋 예측
    X_test_clf = extract_X(df_clf_test, CLF_FEATURE_COLS)      # CLF 입력 컬럼
    y_test_true = df_clf_test[CLF_TARGET_COL].astype(str).values

    proba = classifier_obj["model"].predict_proba(X_test_clf)
    y_pred_enc = np.argmax(proba, axis=1)
    y_pred = classifier_obj["label_encoder"].inverse_transform(y_pred_enc)

    # 2) 정확도/리포트 출력
    acc = accuracy_score(y_test_true, y_pred)
    print("\n================ CLF TEST METRICS ================")
    print(f"CLF accuracy: {acc:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_test_true, y_pred, zero_division=0))
    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test_true, y_pred))
    print("==================================================\n")


    # (5) Save ONE unified model file
    cfg = UnifiedConfig(
        id_col=ID_COL,
        time_col=TIME_COL,
        speed_col="rail_speed",
        win_size=WIN_SIZE,
        stride=STRIDE,
        onset_col=ONSET_COL,
        recovery_col=RECOVERY_COL,
        drop_raw_cols=DROP_RAW_COLS,
    )
    unified = UnifiedModel(
        cfg,
        det_feature_cols=DET_FEATURE_COLS,
        clf_feature_cols=CLF_FEATURE_COLS,
        detector=detector_obj,
        classifier=classifier_obj,
    )

    os.makedirs(os.path.dirname(OUT_UNIFIED), exist_ok=True)
    joblib.dump(unified, OUT_UNIFIED)

    print(f"\n[OK] ONE unified model saved -> {OUT_UNIFIED}")
    print(f"     DET_FEATURE_COLS({len(DET_FEATURE_COLS)}): {DET_FEATURE_COLS[:6]} ...")
    print(f"     CLF_FEATURE_COLS({len(CLF_FEATURE_COLS)}): {CLF_FEATURE_COLS[:6]} ...")
    print(f"     detector.time_segment_enabled: {detector_obj['time_segment_enabled']}")
    print(f"     detector.models: {list(detector_obj['models'].keys())}")
    print(f"     classifier.classes: {classifier_obj['classes_']}")