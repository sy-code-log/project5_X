
import streamlit as st
import datetime
import joblib
import pandas as pd
import numpy as np
import holidays
from streamlit_option_menu import option_menu


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
    return table.get(key, default)


# 모델에 들어갈 컬럼 정의
def make_feature_row(user_inputs: dict, lookup_tables: dict, start_date: datetime.date) -> pd.DataFrame:
    # 날짜 기반 feature 생성
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
        "month": month,
        "quarter": quarter,
        "is_month_start": is_month_start,
        "is_month_end": is_month_end,
        "is_weekday_holiday": is_weekday_holiday,
    }

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

    # DataFrame 반환
    df = pd.DataFrame([row])

    # 카테고리형 맞추기
    cat_cols = ["domain", "ads_rejoin_type", "ads_os_type", "mda_idx", "ads_3step"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


# 새 광고 정보와 광고 예산을 받아서 매체별 1주일 기준 예측 결과를 계산하고 Top-N 추천
def predict_and_rank(user_inputs: dict, lookup_tables: dict, start_date, model, final_df, ad_budget: float, top_n: int = 5):
    results = []

    # 유사 광고 필터링 (domain + ads_3step 기준)
    similar_ads = final_df[
        (final_df["domain"] == user_inputs["domain"]) &
        (final_df["ads_3step"] == user_inputs["ads_3step"])
    ]
    similar_mda = similar_ads["mda_idx"].unique().tolist()

    # 가능한 매체 목록 (유사 광고 매체만, 없으면 전체)
    if len(similar_mda) > 0:
        all_mda = [str(m) for m in similar_mda]
    else:
        all_mda = list(lookup_tables["mda_mean_acost"].keys())

    # 전체 평균 클릭수 (fallback)
    global_mean_clk = final_df["mda_mean_clk"].mean()

    for mda in all_mda:
        temp_inputs = user_inputs.copy()
        temp_inputs["mda_idx"] = str(mda)  

        # feature row 생성
        feature_row = make_feature_row(temp_inputs, lookup_tables, start_date)
  
        # 모델 예측 (예상 전환율)
        pred_cvr = model.predict(feature_row)[0]

        # 매체 평균 클릭당 비용 (없으면 1원)
        mean_acost = lookup_tables["mda_mean_acost"].get(mda, 1)
        if mean_acost is None or mean_acost <= 0:
            mean_acost = 1

        # 클릭수 추정 (예산 기반)
        expected_clicks = ad_budget / mean_acost

        # 예측 전환수 
        expected_conversions = expected_clicks * pred_cvr

        # 아이브 수익 추정 
        # 광고주 지출 = 예상 클릭수 × 평균 단가(acost)
        expected_acost = expected_clicks * mean_acost
        # 매체 정산액 = 예상 클릭수 × 평균 정산액(earn)
        mean_earn = lookup_tables["mda_mean_earn"].get(str(mda), 0)
        expected_earn = expected_clicks * mean_earn
        # 순수익 = 광고주 지출 - 매체 정산액
        expected_profit = expected_acost - expected_earn

        # 최소 기준 적용
        if expected_profit <= 0:
            continue
        if expected_clicks < 30:
            continue

        results.append({
            "mda_idx": mda,
            "predicted_cvr": pred_cvr,
            "expected_clicks": expected_clicks,
            "expected_conversions": expected_conversions,
            "ive_expected_profit": expected_profit
        })

    # 정렬 후 Top-N 반환
    results_df = pd.DataFrame(results).sort_values(by="predicted_cvr", ascending=False).head(top_n).reset_index(drop=True)
    results_df.index = results_df.index + 1
    return results_df


# -------------------------------------------------

# 모델 & lookup 불러오기
@st.cache_resource
def load_model_and_lookup():
    model = joblib.load("lgbm_final_model.pkl")
    lookup_tables = joblib.load("lookup_tables.pkl")
    final_df = joblib.load("final_df.pkl")
    metrics = {"MAE": 0.1768, "RMSE": 0.2547} 
    return model, lookup_tables, final_df, metrics

model, lookup_tables, final_df, metrics = load_model_and_lookup()


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
            padding-top: 1rem;
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
            ad_budget = int(ad_budget_str.replace(",", ""))
            st.markdown("<br>", unsafe_allow_html=True) 

            # 추천 매체 개수
            top_n = st.slider('▶︎\u00A0\u00A0 추천 매체 개수', min_value=3, max_value=50, value=5, key="topn_slider")
        
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
                model=model,
                final_df=final_df,
                ad_budget=ad_budget,
                top_n=top_n
            )
            st.session_state.results_df = results_df
            st.session_state.user_inputs = user_inputs
            
            # '추천 매체' 탭으로 이동하도록 상태 변경 후 rerun
            st.session_state.active_tab = "추천 매체"
            st.rerun()


# 추천 매체
elif st.session_state.active_tab == '추천 매체':
    if "results_df" in st.session_state and st.session_state.results_df is not None:
        # 추가 필터
        spacer, col1, spacer, col2 = st.columns([0.4, 2, 1.3, 5]) 

        with col1:
            st.markdown(
                """
                <style>
                .filter-title {
                    font-size: 18px;       /* 글씨 크기 */
                    font-weight: 700;      /* 굵기 */
                    color: white;          /* 색상 */
                    margin-bottom: 0px;    /* 라벨과 간격 */
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # 최소 클릭수 필터 제목
            st.markdown("<p class='filter-title'>최소 클릭수 필터</p>", unsafe_allow_html=True)

            # 슬라이더
            min_clicks = st.slider(
                label="",   # 라벨 비우고
                min_value=0,
                max_value=100,
                value=30,
                step=10,
                key="min_clicks_slider"
            )

        with col2:
            # 라디오 옵션 글씨 크기 & 간격 조절
            st.markdown(
                """
                <style>
                /* 추천 정렬 기준 제목 */
                .sort-title {
                    font-size: 15px;
                    font-weight: 700;
                    color: white;
                    margin-bottom: -10px !important;  /* 아래쪽 간격 */
                }
                /* 라디오 버튼 옵션 텍스트 */
                div[role="radiogroup"] label p {
                    font-size: 13px !important;   /* 글씨 크기 */
                    font-weight: 500 !important;  /* 두께 */
                    margin: 0px !important;       /* 여백 줄이기 */
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<p class='sort-title'>추천 정렬 기준</p>", unsafe_allow_html=True)

            sort_option = st.radio(
                label="",
                options=("예측 전환율 (%)", "예상 전환수", "아이브 예상 수익 (원)"),
                index=0,
                horizontal=True,
                key="sort_option_radio"
            )
        

        st.markdown("<br>", unsafe_allow_html=True) 

        # 데이터 복사 후 변환
        df_display = st.session_state.results_df.copy()
        df_display["predicted_cvr"] = df_display["predicted_cvr"] * 100
        df_display = df_display.rename(columns={"predicted_cvr": "예측 전환율 (%)",
                                                "expected_clicks": "예상 클릭수",
                                                "expected_conversions": "예상 전환수",
                                                "ive_expected_profit": "아이브 예상 수익 (원)"})

        # 필터링
        df_display = df_display[df_display["예상 클릭수"] >= min_clicks]

        # 정렬
        if sort_option == "예측 전환율 (%)":
            df_display = df_display.sort_values(by="예측 전환율 (%)", ascending=False)
        elif sort_option == "예상 전환수":
            df_display = df_display.sort_values(by="예상 전환수", ascending=False)
        elif sort_option == "아이브 예상 수익 (원)":
            df_display = df_display.sort_values(by="아이브 예상 수익 (원)", ascending=False)

        # 표 출력
        st.dataframe(
            df_display.style.format({
                "예측 전환율 (%)": "{:.2f}",
                "예상 클릭수": "{:,.0f}",
                "예상 전환수": "{:,.0f}",
                "아이브 예상 수익 (원)": "{:,.0f}",
            }),
            use_container_width=True
        )

        # 모델 성능
        st.markdown(
            f"<div style='text-align:right; color:gray; font-size:12px;'>"
            f"※ 위 지표들은 1주일 기준 추정값입니다."
            f"<br>"
            f"모델: LightGBM<br>MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f}"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("먼저 '광고 정보' 탭에서 정보를 입력하고 실행해주세요.")


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
        st.markdown("<br>", unsafe_allow_html=True)

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

            st.markdown("<br>", unsafe_allow_html=True)
            
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