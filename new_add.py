import streamlit as st
import datetime
import re
import joblib
import pandas as pd
import numpy as np
import holidays
from streamlit_option_menu import option_menu


# 사용 함수 정리

# 광고 시작일로부터 시간 관련 feature 생성
kr_holidays = holidays.KR(years=[2024, 2025])
def make_time_features(start_date):
    month = start_date.month
    quarter = (month - 1) // 3 + 1
    is_month_start = 1 if start_date.day <= 10 else 0
    is_month_end = 1 if start_date.day >= 25 else 0
    is_weekday_holiday = int(start_date.weekday() < 5 and start_date in kr_holidays)
    return month, quarter, is_month_start, is_month_end, is_weekday_holiday

# Lookup 값 가져오기
def get_lookup_value(table, key, default=0):
    value = table.get(key, default)
    return default if pd.isna(value) else value

# 모델에 들어갈 컬럼 정의
def make_feature_row(user_inputs: dict, lookup_tables: dict, start_date: datetime.date, final_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    # 1. 날짜 기반 feature 생성
    month, quarter, is_month_start, is_month_end, is_weekday_holiday = make_time_features(start_date)
    row = {
        "week": 1,
        "mda_idx": user_inputs["mda_idx"],
        "domain": user_inputs["domain"],
        "ads_3step": user_inputs["ads_3step"],
        "ads_rejoin_type": user_inputs["ads_rejoin_type"],
        "ads_os_type": user_inputs["ads_os_type"],
        "ads_payment": user_inputs["ads_payment"],
        "ads_length": user_inputs["ads_length"],
        "age_limit": user_inputs["age_limit"],
        "gender_limit": user_inputs["gender_limit"],
        "active_days": user_inputs["active_days"],
        "month": month,
        "quarter": quarter,
        "is_month_start": is_month_start,
        "is_month_end": is_month_end,
        "is_weekday_holiday": is_weekday_holiday,
    }


    # 2. lookup_tables 기반 피처
    # mda_idx 따로 꺼내기
    mda = str(user_inputs["mda_idx"])

    # --- ① 매체 단위
    for col in ["mda_mean_acost", "mda_mean_earn", "mda_mean_clk", "mda_mean_turn", "mda_cost_ratio"]:
        row[col] = get_lookup_value(lookup_tables[col], mda)

    # --- ② 도메인 단위
    for col in ["domain_acost_mean", "domain_earn_mean", "domain_cvr", "domain_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], user_inputs["domain"])

    # --- ③ ads_3step 단위
    for col in ["ads_3step_acost_mean", "ads_3step_earn_mean", "ads_3step_cvr", "ads_3step_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], str(user_inputs["ads_3step"]))

    # --- ④ ads_os_type 단위
    for col in ["ads_os_type_acost_mean", "ads_os_type_earn_mean", "ads_os_type_cvr", "ads_os_type_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], str(user_inputs["ads_os_type"]))

    # --- ⑤ mda_idx 단위
    for col in ["mda_idx_cvr", "mda_idx_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], mda)

    # --- ⑥ ads_os_type + mda_idx
    for col in ["ads_os_type_mda_idx_acost_mean", "ads_os_type_mda_idx_earn_mean", "ads_os_type_mda_idx_cvr", "ads_os_type_mda_idx_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (str(user_inputs["ads_os_type"]), mda))

    # --- ⑦ domain + mda_idx
    for col in ["domain_mda_idx_acost_mean", "domain_mda_idx_earn_mean", "domain_mda_idx_cvr", "domain_mda_idx_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (user_inputs["domain"], mda))

    # --- ⑧ domain + ads_os_type
    for col in ["domain_ads_os_type_acost_mean", "domain_ads_os_type_earn_mean", "domain_ads_os_type_cvr", "domain_ads_os_type_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (user_inputs["domain"], str(user_inputs["ads_os_type"])))

    # --- ⑨ domain + ads_3step
    for col in ["domain_ads_3step_acost_mean", "domain_ads_3step_earn_mean", "domain_ads_3step_cvr", "domain_ads_3step_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (user_inputs["domain"], str(user_inputs["ads_3step"])))

    # --- ⑩ ads_3step + ads_os_type
    for col in ["ads_3step_ads_os_type_acost_mean", "ads_3step_ads_os_type_earn_mean", "ads_3step_ads_os_type_cvr", "ads_3step_ads_os_type_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (str(user_inputs["ads_3step"]), str(user_inputs["ads_os_type"])))

    # --- ⑪ ads_3step + mda_idx
    for col in ["ads_3step_mda_idx_acost_mean", "ads_3step_mda_idx_earn_mean", "ads_3step_mda_idx_cvr", "ads_3step_mda_idx_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (str(user_inputs["ads_3step"]), mda))

    # --- ⑫ domain + ads_3step + ads_os_type
    for col in ["domain_ads_3step_ads_os_type_acost_mean", "domain_ads_3step_ads_os_type_earn_mean", "domain_ads_3step_ads_os_type_cvr", "domain_ads_3step_ads_os_type_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (user_inputs["domain"], str(user_inputs["ads_3step"]), str(user_inputs["ads_os_type"])))

    # --- ⑬ domain + ads_3step + mda_idx
    for col in ["domain_ads_3step_mda_idx_acost_mean", "domain_ads_3step_mda_idx_earn_mean", "domain_ads_3step_mda_idx_cvr", "domain_ads_3step_mda_idx_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (user_inputs["domain"], str(user_inputs["ads_3step"]), mda))

    # --- ⑭ ads_3step + ads_os_type + mda_idx
    for col in ["ads_3step_ads_os_type_mda_idx_acost_mean", "ads_3step_ads_os_type_mda_idx_earn_mean", "ads_3step_ads_os_type_mda_idx_cvr", "ads_3step_ads_os_type_mda_idx_turn_per_day"]:
        row[col] = get_lookup_value(lookup_tables[col], (str(user_inputs["ads_3step"]), str(user_inputs["ads_os_type"]), mda))


    # 3. 교차 카테고리 피처
    row["domain_ads3step"] = f"{user_inputs['domain']}_{user_inputs['ads_3step']}"
    row["domain_mda"] = f"{user_inputs['domain']}_{mda}"
    row["ads3step_mda"] = f"{user_inputs['ads_3step']}_{mda}"
    row["domain_os"] = f"{user_inputs['domain']}_{user_inputs['ads_os_type']}"
    row["ads3step_os"] = f"{user_inputs['ads_3step']}_{user_inputs['ads_os_type']}"
    row["mda_os"] = f"{mda}_{user_inputs['ads_os_type']}"


    # 4. flag 피처
    mda = str(user_inputs["mda_idx"])
    domain = user_inputs["domain"]
    ads_3step = user_inputs["ads_3step"]
    ads_os_type = user_inputs["ads_os_type"]

    # 단일 단위
    row["is_first_domain"] = 0 if domain in final_df["domain"].unique() else 1
    row["is_first_ads_3step"] = 0 if ads_3step in final_df["ads_3step"].unique() else 1
    row["is_first_ads_os_type"] = 0 if ads_os_type in final_df["ads_os_type"].unique() else 1
    row["is_first_mda_idx"] = 0 if mda in final_df["mda_idx"].astype(str).unique() else 1
    row["is_small_mda_idx"] = 1 if (final_df["mda_idx"].astype(str).value_counts().get(mda, 0) < 5) else 0
    
    # 교차 단위
    def check_first(df, cols, values):
        return 0 if values in df.set_index(cols).index else 1

    def check_small(df, cols, values):
        return 1 if df.groupby(cols).size().get(values, 0) < 5 else 0

    row["is_first_domain_mda_idx"] = check_first(final_df, ["domain","mda_idx"], (domain, mda))
    row["is_first_domain_ads_os_type"] = check_first(final_df, ["domain","ads_os_type"], (domain, ads_os_type))
    row["is_first_domain_ads_3step"] = check_first(final_df, ["domain","ads_3step"], (domain, ads_3step))
    row["is_first_ads_3step_ads_os_type"] = check_first(final_df, ["ads_3step","ads_os_type"], (ads_3step, ads_os_type))
    row["is_first_ads_3step_mda_idx"] = check_first(final_df, ["ads_3step","mda_idx"], (ads_3step, mda))
    row["is_first_ads_os_type_mda_idx"] = check_first(final_df, ["ads_os_type","mda_idx"], (ads_os_type, mda))
    row["is_first_domain_ads_3step_ads_os_type"] = check_first(final_df, ["domain","ads_3step","ads_os_type"], (domain, ads_3step, ads_os_type))
    row["is_first_domain_ads_3step_mda_idx"] = check_first(final_df, ["domain","ads_3step","mda_idx"], (domain, ads_3step, mda))
    row["is_first_ads_3step_ads_os_type_mda_idx"] = check_first(final_df, ["ads_3step","ads_os_type","mda_idx"], (ads_3step, ads_os_type, mda))
    
    row["is_small_domain_mda_idx"] = check_small(final_df, ["domain","mda_idx"], (domain, mda))
    row["is_small_ads_3step_mda_idx"] = check_small(final_df, ["ads_3step","mda_idx"], (ads_3step, mda))
    row["is_small_ads_os_type_mda_idx"] = check_small(final_df, ["ads_os_type","mda_idx"], (ads_os_type, mda))
    row["is_small_domain_ads_3step_mda_idx"] = check_small(final_df, ["domain","ads_3step","mda_idx"], (domain, ads_3step, mda))
    row["is_small_ads_3step_ads_os_type_mda_idx"] = check_small(final_df, ["ads_3step","ads_os_type","mda_idx"], (ads_3step, ads_os_type, mda))
    

    df = pd.DataFrame([row])

    # 카테고리형 
    cat_cols = [
        "domain", "ads_rejoin_type", "ads_os_type", "mda_idx", "ads_3step",
        "domain_ads3step","domain_mda","ads3step_mda","domain_os","ads3step_os","mda_os"
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # 학습 시 사용한 feature 순서와 맞추기
    df = df.reindex(columns=feature_cols, fill_value=0)
        
    return df

# 예상 전환수, 클릭수, 수익 구할 때 이전의 기록 사용하기 위해 가져올 값 정의하는 함수
def get_fallback_value(lookup_tables, keys, target, default=0):
    def valid(v):
        return v is not None and not pd.isna(v)

    # 1순위: domain + ads_3step + ads_os_type
    if target == "acost":
        val = lookup_tables["domain_ads_3step_ads_os_type_acost_mean"].get(
            (keys["domain"], str(keys["ads_3step"]), str(keys["ads_os_type"]))
        )
    elif target == "earn":
        val = lookup_tables["domain_ads_3step_ads_os_type_earn_mean"].get(
            (keys["domain"], str(keys["ads_3step"]), str(keys["ads_os_type"]))
        )
    elif target == "turn_per_day":
        val = lookup_tables["domain_ads_3step_ads_os_type_turn_per_day"].get(
            (keys["domain"], str(keys["ads_3step"]), str(keys["ads_os_type"]))
        )
    if valid(val):
        return val

    # 2순위: domain
    if target == "acost":
        val = lookup_tables["domain_acost_mean"].get(keys["domain"])
    elif target == "earn":
        val = lookup_tables["domain_earn_mean"].get(keys["domain"])
    elif target == "turn_per_day":
        val = lookup_tables["domain_turn_per_day"].get(keys["domain"])
    if valid(val):
        return val

    # 3순위: ads_3step
    if target == "acost":
        val = lookup_tables["ads_3step_acost_mean"].get(str(keys["ads_3step"]))
    elif target == "earn":
        val = lookup_tables["ads_3step_earn_mean"].get(str(keys["ads_3step"]))
    elif target == "turn_per_day":
        val = lookup_tables["ads_3step_turn_per_day"].get(str(keys["ads_3step"]))
    if valid(val):
        return val

    # 4순위: mda_idx
    if target == "acost":
        val = lookup_tables["mda_mean_acost"].get(str(keys["mda_idx"]))
    elif target == "earn":
        val = lookup_tables["mda_mean_earn"].get(str(keys["mda_idx"]))
    elif target == "turn_per_day":
        val = lookup_tables["mda_mean_clk"].get(str(keys["mda_idx"]))
    if valid(val):
        return val

    return default

# 과거 평균 클릭수를 조합별 우선순위로 가져오고, 진행일수(active_days) 비율로 보정.
def get_baseline_clicks(lookup_tables, domain, ads_3step, ads_os_type, mda, active_days):
    def _valid(v):
        return v is not None and not pd.isna(v)

    base = None
    # ① domain + ads_3step + mda_idx
    if base is None:
        base = lookup_tables.get("domain_ads_3step_mda_idx_clk_mean", {}).get((domain, str(ads_3step), str(mda)))
    # ② domain + mda_idx
    if not _valid(base):
        base = lookup_tables.get("domain_mda_idx_earn_mean", {}).get((domain, str(mda)))  # 클릭이 없으면 유사 대체
    # ③ mda 전체 평균 클릭
    if not _valid(base):
        base = lookup_tables.get("mda_mean_clk", {}).get(str(mda))
    # ④ 마지막 fallback
    if not _valid(base):
        base = 30.0  # 주당 최소 기대 클릭(임계값)

    # 진행일수(1~7일) 반영
    scale = max(1, min(7, int(active_days))) / 7.0
    return float(base) * scale

# 예산이 충분하면 예산기반에 가깝게, 예산이 작으면 과거기반에 가깝게.
def estimate_clicks(ad_budget, mean_acost, baseline_clicks):
    mean_acost = max(1e-6, float(mean_acost))
    budget_clicks = float(ad_budget) / mean_acost

    typical_cost = baseline_clicks * mean_acost
    # 예산/필요예산 비율(0~1) → 가중치
    w = 0.0 if typical_cost <= 0 else min(1.0, float(ad_budget) / typical_cost)

    blended = w * budget_clicks + (1.0 - w) * baseline_clicks

    # 너무 작은 값은 표시에서 0으로 보이니 살짝 바닥값
    return max(blended, 0.1)

# 매체별 1주일 기준 예측 결과를 계산하고 Top-N 추천
def predict_and_rank(user_inputs: dict, lookup_tables: dict, start_date, cvr_model, ranker_model, final_df, ad_budget: float, feature_cols: list, top_n: int = 10):
    results = []

    # 유사 광고 필터링 (domain + ads_3step + ads_os_type 기준)
    similar_ads = final_df[
        (final_df["domain"] == user_inputs["domain"]) &
        (final_df["ads_3step"] == user_inputs["ads_3step"]) &
        (final_df["ads_os_type"] == user_inputs["ads_os_type"])
    ]
    similar_mda = similar_ads["mda_idx"].unique().tolist()

    # 가능한 매체 목록 (유사 광고 매체만, 없으면 전체)
    if len(similar_mda) > 0:
        all_mda = [str(m) for m in similar_mda]
    else:
        all_mda = list(lookup_tables["mda_mean_acost"].keys())

    for mda in all_mda:
        temp_inputs = user_inputs.copy()
        temp_inputs["mda_idx"] = str(mda)

        # feature row 생성
        feature_row = make_feature_row(
            temp_inputs, lookup_tables, start_date, final_df, feature_cols
        )

        # 1. 예측 전환율 (CVR 모델)
        pred_cvr = cvr_model.predict(feature_row)[0]

        # 2. 랭킹 점수 (Ranker 모델)
        rank_score = ranker_model.predict(feature_row)[0]

        # 3. 평균 광고비용 & 평균 매체사비용 & capacity
        keys = {
            "domain": user_inputs["domain"],
            "ads_3step": user_inputs["ads_3step"],
            "ads_os_type": user_inputs["ads_os_type"],
            "mda_idx": mda
        }

        mean_acost = get_fallback_value(lookup_tables, keys, "acost", default=1)
        mean_earn = get_fallback_value(lookup_tables, keys, "earn", default=0)
        capacity = get_fallback_value(lookup_tables, keys, "turn_per_day", default=np.inf)

        if mean_acost is None or mean_acost <= 0:
            mean_acost = 1

        # 4. 과거 평균 클릭(주간) 기준 가져오기 + 진행일 보정
        baseline_clicks = get_baseline_clicks(
            lookup_tables,
            domain=user_inputs["domain"],
            ads_3step=user_inputs["ads_3step"],
            ads_os_type=user_inputs["ads_os_type"],
            mda=mda,
            active_days=user_inputs.get("active_days", 7),
        )

        # 5. 블렌딩 클릭 추정
        expected_clicks = estimate_clicks(ad_budget, mean_acost, baseline_clicks)

        # 6. 상한선(과도한 과대추정 방지) – 과거 평균의 3배
        expected_clicks = min(expected_clicks, 3.0 * baseline_clicks)

        # 6. 전환수
        expected_conversions = expected_clicks * pred_cvr

        # 7. 비용/성과 계산
        expected_acost = expected_clicks * mean_acost
        expected_earn = expected_clicks * mean_earn
        expected_profit = expected_acost - expected_earn

        # 8. 최소 기준 필터링
        if expected_profit <= 0:
            continue

        # 9. 결과 저장
        results.append({
            "mda_idx": mda,
            "rank_score": rank_score,
            "predicted_cvr": pred_cvr,
            "expected_clicks": expected_clicks,
            "expected_conversions": expected_conversions,
            "expected_acost": expected_acost,
            "expected_earn": expected_earn,
            "ive_expected_profit": expected_profit
        })

    # 10. 정렬
    results_df = (
        pd.DataFrame(results)
        .sort_values(by="rank_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    results_df.index = results_df.index + 1

    # 11. 광고주 ROI 추가 (성과/비용)
    results_df["ROI"] = results_df["expected_earn"] / (results_df["expected_acost"] + 1e-6)

    return results_df

# 상위 5개 매체에 대해 ROI 기반으로 예산 배분 추천
def recommend_budget_allocation(results_df, total_budget):
    # 상위 5개 매체만 사용
    top5 = results_df.head(5).copy()

    # ROI 비율 정규화
    top5["allocation_ratio"] = top5["ROI"] / top5["ROI"].sum()

    # 권장 예산 배분
    top5["allocated_budget"] = top5["allocation_ratio"] * total_budget

    return top5[[
        "mda_idx",
        "ROI",
        "allocation_ratio",
        "allocated_budget",
        "expected_clicks",
        "expected_conversions",
        "ive_expected_profit"   
    ]]

# -------------------------------------------------

# 모델 & lookup 불러오기
@st.cache_resource
def load_model_and_lookup():
    cvr_model = joblib.load("lgbm_final_model.pkl")
    ranker_model = joblib.load("ranker_model.pkl")
    lookup_tables = joblib.load("lookup_tables.pkl")
    final_df = joblib.load("final_df.pkl")
    feature_cols = joblib.load("feature_cols.pkl") 
    
    # 성능 지표 (CVR 모델 기준)
    cvr_metrics = {"MAE": 0.1768, "RMSE": 0.2547} 
    # 성능 지표 (랭킹 모델 기준)
    rank_metrics = {"hit@10": 0.4538, "ndcg@10": 0.6020}
    return cvr_model, ranker_model, lookup_tables, final_df, feature_cols, cvr_metrics, rank_metrics 

cvr_model, ranker_model, lookup_tables, final_df, feature_cols, cvr_metrics, rank_metrics  = load_model_and_lookup()


# Streamlit UI
st.set_page_config(page_title='신규 광고 매체 추천 시스템')


# 배경 색상
st.markdown(
    """
    <style>
    .stApp {background-color: #2D2D2D;}
    </style>
    """,
    unsafe_allow_html=True
)


# 상단 여백 
st.markdown(
    """
    <style>
        /* 전체 페이지 상단 여백 제거 */
        .block-container {
            padding-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# 공통 CSS 스타일 정의
st.markdown(
    """
    <style>
    /* --- Streamlit 헤더 숨기기 --- */
    [data-testid="stHeader"] {
        background-color: transparent; /* 헤더 배경을 투명하게 만듦 */
    }

    /* ===== 공통 컨테이너 (들여쓰기 적용) ===== */
    .content-container {
        padding-left: 30px;   /* 좌측 여백 */
        padding-right: 30px;  /* 우측 여백 */
    }

    /* ===== 제목 스타일 ===== */
    .section-title {
        color: white;
        font-size: 23px;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 15px;
    }

    /* ===== Expander 전체 박스 ===== */
    div[data-testid="stExpander"] {
        background-color: #1A1A1A !important;
        border-radius: 10px !important;
        border: 1px solid #444 !important;
        color: white !important;
        margin-bottom: 15px !important;
        margin-left: 30px !important;  
        margin-right: 30px !important;  
        width: calc(100% - 60px) !important; 

    }
    div[data-testid="stExpander"] summary {
        color: #fff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    div[data-testid="stExpander"] div[role="region"] {
        background-color: #1E1E1E !important;
        color: #ddd !important;
        padding: 10px !important;
    }

    /* ===== Selectbox ===== */
    div[data-baseweb="select"] > div {
        background-color: #1A1A1A !important;
        border-radius: 9px !important;
        border: 1px solid #444 !important;
        color: white !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    div[role="listbox"] {
        background-color: #1E1E1E !important;
        border-radius: 8px !important;
        border: 1px solid #444 !important;
        padding: 4px !important;
    }
    div[role="option"] {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        line-height: 1.8em !important;
        color: #ddd !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    div[role="option"]:hover {
        background-color: #333 !important;
        color: white !important;
    }

    /* ===== 실행 버튼 ===== */
    div.stButton > button:first-child {
        background-color: #E9353E;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4em 1.2em;
        font-size: 16px;
        font-weight: 900;
        margin-top: 10px;
        transition: 0.3s;
        margin-left: 30px ;
        width: 80px;
    }
    div.stButton > button:first-child:hover {
        background-color: #c62828;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 세션 상태 초기화
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "광고 정보" 


menu_options = ["광고 정보", "추천 매체", "매체 상세 분석"]
menu_icons = ["house", "bar-chart", "collection"]


# 현재 세션 상태에 맞는 탭의 인덱스를 계산
try:
    default_index = menu_options.index(st.session_state.active_tab)
except ValueError:
    default_index = 0

selected = option_menu(
    None,
    menu_options,
    icons=menu_icons,
    orientation="horizontal",
    default_index=default_index,
    styles={
        "container": {"padding": "0!important", "background-color": "transparent", "border": "none"},
        "icon": {"font-size": "18px"},
        "nav-link": {"font-size": "16px", "font-weight": "700", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#E9353E", "color": "white", "border-radius": "8px"},
    }
)

# 탭이 변경되었는지 확인하고, 변경되었다면 세션 상태를 업데이트 후 rerun
if selected != st.session_state.active_tab:
    st.session_state.active_tab = selected
    st.rerun()



# 광고 정보
if st.session_state.active_tab == '광고 정보':
    # 부제목 크기
    with st.container():
        st.markdown("<br>", unsafe_allow_html=True) 
        st.markdown(
            "<div style='text-align:left; color:white; font-size:17px; padding-left:30px; font-weight: 600;'>"
            "신규 광고 기본 정보 입력"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("𝟭. 광고 기본 정보"):
            # 광고 도메인
            domain = st.selectbox("▶︎\u00A0\u00A0 광고 도메인", 
                                ['금융/보험', '게임', '상품소비', '생활서비스', '플랫폼',
                                'SNS/커뮤니케이션', '콘텐츠', '앱테크/리워드',
                                '유틸리티/툴', '지역/상점', '기타'], key="domain_select")
            st.markdown("<br>", unsafe_allow_html=True) 

            # 광고 분류 3단계
            labels1 = {
                1 : "1단계 : 단순 노출 및 클릭",
                2 : "2단계 : 행동 유도 (설치, 실행, 참여, 퀴즈, 구독 등)",
                3 : "3단계 : 최종 수익 창출 (구매, 게임 내 특정 퀘스트 달성 등)"
            }
            ads_3step = st.selectbox("▶︎\u00A0\u00A0 광고 분류 3단계",options=[1, 2, 3], format_func=lambda x: labels1[x])
            st.markdown("<br>", unsafe_allow_html=True) 

            # 앱/웹 광고
            labels2 = {
                0 : "APP(앱)",
                1 : "WEB(웹)"
            }
            ads_os_type = st.selectbox("▶︎\u00A0\u00A0 앱/웹 광고", options=[0, 1], format_func=lambda x: labels2[x])
            st.markdown("<br>", unsafe_allow_html=True) 

            # 참여 제한 조건
            labels3 = {
                'NONE' : "재참여 불가 (1인 1회)", 
                'ADS_CODE_DAILY_UPDATE' : "매일 재참여 가능 (1인 1일 1회)",
                'REJOINABLE' : "계속 재참여 가능 (1인 1일 무제한)"
            }
            ads_rejoin_type = st.selectbox("▶︎\u00A0\u00A0 참여 제한 조건", options=['NONE', 'ADS_CODE_DAILY_UPDATE', 'REJOINABLE'], format_func=lambda x: labels3[x])
            st.markdown("<br>", unsafe_allow_html=True) 

            # 광고 길이
            ads_length = st.number_input('▶︎\u00A0\u00A0 광고 내용 길이', min_value=1, value=200, step=1, key="length_input",
                                         help='광고 문구의 글자 수(빈칸 포함)를 입력해주세요.')
    
        with st.expander("𝟮. 광고 조건"):
            # 광고 집행 개시일
            start_date = st.date_input('▶︎\u00A0\u00A0 광고 집행 개시일', datetime.date.today())
            st.markdown("<br>", unsafe_allow_html=True) 

            # 광고 진행 일수
            active_days = st.number_input('▶︎\u00A0\u00A0 주간 광고 진행 일수', min_value=1, max_value=7, step=1,
                                        help='일주일(7일) 중 광고 진행 예정 일수')
            st.markdown("<br>", unsafe_allow_html=True) 
            
            # 연령 제한 여부
            labels4 = {
                0 : "무",
                1 : "유"
            }
            age_limit = st.selectbox('▶︎\u00A0\u00A0 연령 제한 여부', options=[0, 1], format_func=lambda x: labels4[x])
            st.markdown("<br>", unsafe_allow_html=True) 

            # 성별 제한 여부
            labels5 = {
                0 : "무", 
                1 : "유"
            }
            gender_limit = st.selectbox('▶︎\u00A0\u00A0 성별 제한 여부', options=[0, 1], format_func=lambda x: labels5[x])
            st.markdown("<br>", unsafe_allow_html=True) 

            # 유저 광고 참여 비용
            ads_payment = st.number_input('▶︎\u00A0\u00A0 유저 광고 참여 비용(원)', min_value=0, value=0, step=10, key="payment_input")

        with st.expander("𝟯. 예산 및 추천 설정"):
            # 일주일 광고 예산
            ad_budget_str = st.text_input("▶︎\u00A0\u00A0 일주일 광고 예산 (원)", "1,000,000", key="budget_input")
            ad_budget = int(re.sub(r'[^0-9]', '', ad_budget_str)) if ad_budget_str else 0
            st.markdown("<br>", unsafe_allow_html=True) 

            # 추천 매체 개수
            top_n = st.slider('▶︎\u00A0\u00A0 추천 매체 개수', min_value=3, max_value=50, value=10, key="topn_slider")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 실행 버튼
        if st.button("실행", key="run_button"):
            user_inputs = {
                "domain" : domain,
                "ads_3step" : ads_3step,
                "ads_rejoin_type": ads_rejoin_type,
                "ads_os_type": ads_os_type,
                "ads_length": ads_length,
                "age_limit": age_limit,
                "gender_limit": gender_limit,
                "ads_payment": ads_payment,
                "active_days" : active_days,
            }
            results_df = predict_and_rank(
                user_inputs=user_inputs,
                lookup_tables=lookup_tables,
                start_date=start_date,
                cvr_model=cvr_model,
                ranker_model=ranker_model,
                final_df=final_df,
                ad_budget=ad_budget,
                feature_cols=feature_cols,
                top_n=top_n
            )
            st.session_state.results_df = results_df
            st.session_state.user_inputs = user_inputs
            
            # '추천 매체' 탭으로 이동하도록 상태 변경 후 rerun
            st.session_state.active_tab = "추천 매체"
            st.rerun()




# 추천 매체
if st.session_state.active_tab == '추천 매체':
    if "results_df" in st.session_state and st.session_state.results_df is not None:
        # 필터
        spacer, col1, spacer, col2 = st.columns([0.4, 2, 1.3, 5]) 

        # 최소 클릭수 슬라이더
        with col1:
            st.markdown(
                """
                <style>
                .filter-title {
                    font-size: 18px;
                    font-weight: 700;
                    color: white;
                    margin-bottom: 0px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<p class='filter-title'>최소 클릭수 필터</p>", unsafe_allow_html=True)
            min_clicks = st.slider(
                label="", min_value=0, max_value=100, value=30, step=10, key="min_clicks_slider"
            )

        # 정렬 기준 라디오
        with col2:
            st.markdown(
                """
                <style>
                .sort-title {
                    font-size: 15px;
                    font-weight: 700;
                    color: white;
                    margin-bottom: -10px !important;
                }
                div[role="radiogroup"] label p {
                    font-size: 13px !important;
                    font-weight: 500 !important;
                    margin: 0px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<p class='sort-title'>추천 정렬 기준</p>", unsafe_allow_html=True)
            sort_option = st.radio(
                label="",
                options=("랭킹 점수", "예상 전환율 (%)", "ive 예상 수익 (원)"),
                index=0,
                horizontal=True,
                key="sort_option_radio"
            )

        # 결과 테이블
        st.markdown("<br>", unsafe_allow_html=True)
        results_df = st.session_state.results_df

        # 필요한 컬럼만 선택
        df_display = results_df[[
            "mda_idx",
            "rank_score",
            "predicted_cvr",
            "expected_clicks",
            "expected_conversions",
            "ive_expected_profit"
        ]].copy()

        # 전환율 퍼센트 변환
        df_display["predicted_cvr"] = df_display["predicted_cvr"] * 100

        # 한글 컬럼명으로 교체
        df_display = df_display.rename(columns={
            "mda_idx": "매체 번호",
            "rank_score": "랭킹 점수",
            "predicted_cvr": "예상 전환율 (%)",
            "expected_clicks": "예상 클릭수",
            "expected_conversions": "예상 전환수",
            "ive_expected_profit": "ive 예상 수익 (원)"
        })

        # 최소 클릭수 필터 적용
        df_display = df_display[df_display["예상 클릭수"] >= min_clicks]

        # 정렬 기준 적용
        if sort_option == "랭킹 점수":
            df_display = df_display.sort_values(by="랭킹 점수", ascending=False)
        elif sort_option == "예상 전환율 (%)":
            df_display = df_display.sort_values(by="예상 전환율 (%)", ascending=False)
        elif sort_option == "ive 예상 수익 (원)":
            df_display = df_display.sort_values(by="ive 예상 수익 (원)", ascending=False)

        # 모든 값이 양수인 행만 출력
        cols_check = ["예상 클릭수", "예상 전환율 (%)", "ive 예상 수익 (원)"]
        df_display = df_display[(df_display[cols_check] > 0).all(axis=1)]

        if df_display.empty:
            st.warning("⚠️ 조건을 만족하는 추천 매체가 없습니다.")
        else:
            # 포맷 지정
            styled = df_display.style.format({
                "랭킹 점수": "{:.4f}",
                "예상 전환율 (%)": "{:.2f}",
                "예상 클릭수": "{:,.0f}",
                "예상 전환수": "{:,.0f}",
                "ive 예상 수익 (원)": "{:,.0f}"
            })

            # 최고 랭킹 점수를 가진 행 찾기
            max_rank_idx = df_display["랭킹 점수"].idxmax()

            # 스타일 적용
            styled = styled.applymap(
                lambda _: "background-color: #E9353E; color: white; font-weight: 700;",
                subset=pd.IndexSlice[[max_rank_idx], ["랭킹 점수"]]
            )

            st.dataframe(styled, use_container_width=True)
                # 모델 성능

        st.markdown(
            f"<div style='text-align:right; color:gray; font-size:12px;'>"
            f"※ 위 지표들은 1주일 기준 추정값입니다."
            f"<br>"
            f"전환율 예측 모델 (LightGBM) : MAE: {cvr_metrics['MAE']:.4f} | RMSE: {cvr_metrics['RMSE']:.4f}"
            f"<br>"
            f"랭킹 모델 (LGBMRanker) : Hit@10: {rank_metrics['hit@10']:.4f} | NDCG@10: {rank_metrics['ndcg@10']:.4f}"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("👉 먼저 '광고 정보' 탭에서 실행 버튼을 눌러주세요.")




# 매체 상세 분석
elif selected == "매체 상세 분석":
    if "results_df" in st.session_state and st.session_state["results_df"] is not None and "user_inputs" in st.session_state:
        
        st.markdown(
            """
            <style>
            /* --- Selectbox 제목 스타일 --- */
            .filter-title { font-size: 18px; font-weight: 700; color: white; margin-bottom: -90px; margin-top: -5px; padding-left: 33px; }
            div[data-testid="stSelectbox"] { width: 90% !important; margin: 0 auto; }
            
            /* --- 섹션 제목과 경고창 여백 --- */
            .section-header, [data-testid="stAlert"] { margin-left: 30px !important; margin-right: 30px !important; }
            
            /* --- Flexbox 컨테이너 스타일 --- */
            .kpi-container {
                display: flex;
                justify-content: space-between;
                margin: 0 30px; /* 좌우 30px 여백 */
            }
            
            /* --- KPI 카드 개별 스타일 --- */
            .kpi-card { 
                background-color: #1C1C1C; 
                border-radius: 12px; 
                padding: 15px;
                flex: 1; /* 모든 카드가 동일한 너비를 차지하도록 설정 */
                margin: 0 5px; /* 카드 사이 간격 */
                text-align: center;
            }
            .kpi-title { font-size: 14px; color: #aaa; margin-bottom: 8px; }
            .kpi-value { font-size: 20px; color: #E9353E; font-weight: 700; }

            /* --- 섹션 제목 개별 스타일 --- */
            .section-header { font-size: 16px; font-weight: bold; color: white; margin-top: 25px; margin-bottom: 15px; border-bottom: 2px solid #444; padding-bottom: 5px; }
            </style>
            """, unsafe_allow_html=True)
        
        # Selectbox와 제목 
        st.markdown("<p class='filter-title'>매체 선택</p>", unsafe_allow_html=True)
        all_media = sorted(lookup_tables["mda_mean_acost"].keys(), key=lambda x: int(x))
        mda_choice = st.selectbox(label="", options=all_media, key="mda_detail_select")

        if mda_choice:
            # 1. 입력한 광고 조합 기준 성과
            st.markdown("<div class='section-header'>입력하신 광고 조합 기준 매체 평균 성과</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:left; color:gray; font-size:12px; padding-left:30px;'>"
                "※ 광고 조합 : 광고 도메인 + 광고 분류 3단계"
                "<br>"
                "※ 아래 지표들은 조합별로 집계된 현재까지의 매체별 일주일 누적 성과입니다."
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            domain_key = st.session_state["user_inputs"]["domain"]
            ads_3step_key = st.session_state["user_inputs"]["ads_3step"]
            key_tuple = (domain_key, ads_3step_key, mda_choice)

            combo_acost = lookup_tables['domain_ads_3step_mda_idx_acost_mean'].get(key_tuple)
            combo_clk = lookup_tables['domain_ads_3step_mda_idx_clk_mean'].get(key_tuple, 0)
            combo_earn = lookup_tables['domain_ads_3step_mda_idx_earn_mean'].get(key_tuple, 0)
            combo_turn = lookup_tables['domain_ads_3step_mda_idx_turn_mean'].get(key_tuple, 0)

            if combo_acost is None:
                st.warning("선택하신 매체는 입력하신 광고 조합과 일치하는 유의미한 과거 데이터가 없습니다.")
            else:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-card">
                        <div class="kpi-title">광고 단가</div>
                        <div class="kpi-value">{combo_acost:,.0f} 원</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">매체사 수익</div>
                        <div class="kpi-value">{combo_earn:,.0f} 원</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">클릭수</div>
                        <div class="kpi-value">{combo_clk:.2f} 회</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">전환수</div>
                        <div class="kpi-value">{combo_turn:.2f} 회</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            
            # 2. 해당 매체의 전체 평균 성과 
            st.markdown("<div class='section-header'>매체 전체 평균 성과</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:left; color:gray; font-size:12px; padding-left:30px;'>"
                "※ 아래 지표들은 현재까지 집계된 매체별 일주일 누적 성과입니다."
                "</div>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            overall_acost = lookup_tables['mda_mean_acost'].get(mda_choice, 0)
            overall_earn = lookup_tables['mda_mean_earn'].get(mda_choice, 0)
            overall_clk = lookup_tables['mda_mean_clk'].get(mda_choice, 0)
            overall_turn = lookup_tables['mda_mean_turn'].get(mda_choice, 0)
            
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-title">광고 단가</div>
                    <div class="kpi-value">{overall_acost:,.0f} 원</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">매체사 수익</div>
                    <div class="kpi-value">{overall_earn:,.0f} 원</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">클릭수</div>
                    <div class="kpi-value">{overall_clk:,.2f} 회</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">전환수</div>
                    <div class="kpi-value">{overall_turn:.2f} 회</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("먼저 '광고 정보' 탭에서 정보를 입력하고 추천 결과를 확인해주세요.")