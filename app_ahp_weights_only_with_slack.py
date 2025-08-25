# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import datetime

# ==============
# App Config
# ==============
st.set_page_config(
    page_title="AHP Weights + Benchmarks + Scoring (0–1)",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AHP – Weights → Benchmarks → Camp Scoring (0–1 scale)")
st.caption(
    "Upload CSV → Chọn tiêu chí → So sánh cặp (Saaty) → CI nhanh → Trọng số (λmax, CI, CR) → "
    "Benchmark (0–1) → Tính điểm, đóng góp, xếp hạng."
)

# ==============
# Helpers
# ==============
SAATY_LABELS = ["1/9","1/8","1/7","1/6","1/5","1/4","1/3","1/2","1","2","3","4","5","6","7","8","9"]
SAATY_MAP = {
    "1/9": 1/9, "1/8": 1/8, "1/7": 1/7, "1/6": 1/6, "1/5": 1/5, "1/4": 1/4, "1/3": 1/3, "1/2": 1/2,
    "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "6": 6.0, "7": 7.0, "8": 8.0, "9": 9.0
}
_RI = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}

def _weights_eigen(A: np.ndarray) -> Tuple[np.ndarray, float]:
    eigvals, eigvecs = np.linalg.eig(A)
    k = np.argmax(eigvals.real)
    lam_max = eigvals[k].real
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()
    return w, lam_max

def _weights_geo(A: np.ndarray) -> Tuple[np.ndarray, float]:
    gm = np.prod(A, axis=1) ** (1.0 / A.shape[1])
    w = gm / gm.sum()
    lam_max = float(np.mean((A @ w) / w))
    return w, lam_max

def ahp_priority(pairwise_df: pd.DataFrame, method: str = "eigen"):
    A = pairwise_df.values.astype(float)
    n = A.shape[0]
    if method == "geo":
        w, lam_max = _weights_geo(A)
    else:
        w, lam_max = _weights_eigen(A)
    CI = (lam_max - n) / (n - 1) if n > 2 else 0.0
    RI = _RI.get(n, None)
    CR = (CI / RI) if (RI and RI > 0) else 0.0
    result_df = pd.DataFrame({"Criteria": pairwise_df.index, "Weight": w})
    return result_df, lam_max, CI, CR

@st.cache_data(show_spinner=True)
def read_csv_safely(file) -> pd.DataFrame:
    # try default
    try:
        return pd.read_csv(file)
    except Exception:
        pass
    # utf-8-sig
    try:
        file.seek(0)
        return pd.read_csv(file, encoding="utf-8-sig")
    except Exception:
        pass
    # latin-1
    file.seek(0)
    return pd.read_csv(file, encoding="latin1")

@st.cache_data(show_spinner=True)
def load_data(uploaded, use_sample: bool):
    if uploaded is not None:
        df = read_csv_safely(uploaded)
        # normalize decimals if commas used
        for c in df.columns[1:]:
            if df[c].dtype == object:
                df[c] = (
                    df[c].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df, "CSV uploaded"
    if use_sample:
        np.random.seed(7)
        df = pd.DataFrame({
            "campaign_id": [f"C{i+1:02d}" for i in range(8)],
            # metrics on 0–1 scale
            "cpi": np.round(np.random.uniform(0.5, 3.0, 8), 3),
            "roas_d0": np.round(np.random.uniform(0.05, 0.5, 8), 3),
            "roas_d3": np.round(np.random.uniform(0.1, 0.8, 8), 3),
        })
        return df, "(sample data)"
    return pd.DataFrame(), "(no data)"

# ==============
# Sidebar
# ==============
with st.sidebar:
    st.header("📄 Dữ liệu")
    use_sample = st.toggle("Dùng dữ liệu mẫu", value=True)
    uploaded = st.file_uploader("Upload CSV (cột 1 = Camp; các cột sau = metrics)", type=["csv"])

df, note = load_data(uploaded, use_sample)
with st.expander("🔎 Thông tin nguồn dữ liệu", expanded=False):
    st.write(note)

if df.empty:
    st.warning("Chưa có dữ liệu. Upload CSV hoặc bật 'Dùng dữ liệu mẫu'.")
    st.stop()

# ==============
# 1) Chọn tiêu chí
# ==============
st.subheader("1) Chọn tiêu chí (columns)")
first_col = df.columns[0]
candidate_cols = list(df.columns[1:])
criteria = st.multiselect(
    "Chọn metric để tính AHP",
    options=candidate_cols,
    default=candidate_cols,
)
if not criteria:
    st.stop()

st.dataframe(df[[first_col] + criteria].head(20), use_container_width=True)

# ==============
# 2) So sánh cặp
# ==============
st.subheader("2) So sánh cặp tiêu chí (AHP)")
n = len(criteria)
st.caption("Chọn mức ưu tiên của **tiêu chí bên trái so với bên phải** (Saaty 1/9…9).")

# persist selection boxes
for i in range(n):
    for j in range(i+1, n):
        key = f"pw_{i}_{j}"
        if key not in st.session_state:
            st.session_state[key] = "1"  # equal

for i in range(n):
    for j in range(i+1, n):
        colL, colC, colR = st.columns([1.5, 1, 1.5])
        with colL: st.write(f"**{criteria[i]}**")
        with colC:
            st.session_state[f"pw_{i}_{j}"] = st.selectbox(
                f"{criteria[i]} vs {criteria[j]}",
                SAATY_LABELS,
                index=SAATY_LABELS.index(st.session_state[f"pw_{i}_{j}"]),
                key=f"sb_{i}_{j}",
                label_visibility="collapsed",
                help=">1: trái quan trọng hơn phải; <1: ngược lại."
            )
        with colR: st.write(f"**{criteria[j]}**")

# build pairwise matrix
A = np.ones((n, n), dtype=float)
for i in range(n):
    for j in range(i+1, n):
        v = SAATY_MAP[st.session_state[f"pw_{i}_{j}"]]
        A[i, j] = v
        A[j, i] = 1.0 / v
pairwise_df = pd.DataFrame(A, index=criteria, columns=criteria)
with st.expander("Xem ma trận so sánh (A)", expanded=False):
    st.dataframe(pairwise_df.style.format("{:.4f}"), use_container_width=True)

# quick CI
st.subheader("3) Kiểm tra nhất quán nhanh (CI – Eigen)")
_, lam_max_quick = _weights_eigen(A)
CI_quick = (lam_max_quick - n) / (n - 1) if n > 2 else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("λ_max (quick)", f"{lam_max_quick:.4f}")
c2.metric("n", f"{n}")
c3.metric("CI (quick)", f"{CI_quick:.4f}")

# ==============
# 4) Trọng số (chi tiết)
# ==============
st.subheader("4) Tính trọng số & kiểm tra nhất quán (chi tiết)")
method = st.radio("Phương pháp", ["eigen", "geo"], index=0, horizontal=True)
weights_df, lam_max, CI, CR = ahp_priority(pairwise_df, method=method)

m1, m2, m3, m4 = st.columns(4)
m1.metric("λ_max", f"{lam_max:.4f}")
m2.metric("CI", f"{CI:.4f}")
m3.metric("CR", f"{CR:.4f}", "✅ OK (<0.10)" if CR < 0.10 else "⚠️ Cao (≥0.10)")
m4.metric("Số tiêu chí", f"{n}")

st.markdown("**Trọng số AHP**")
st.dataframe(weights_df, use_container_width=True)
st.download_button("⬇️ Tải weights.csv",
    data=weights_df.to_csv(index=False).encode("utf-8"),
    file_name="weights.csv", mime="text/csv")

# weight chart
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(weights_df["Criteria"], weights_df["Weight"])
ax.set_xlabel("Weight"); ax.set_ylabel("Criteria"); ax.set_title("AHP Priority Weights")
ax.invert_yaxis()
for i, v in enumerate(weights_df["Weight"]):
    ax.text(float(v) + 0.005, i, f"{v:.3f}", va="center")
st.pyplot(fig, clear_figure=True)

# ==============
# 5) Benchmarks (0–1)
# ==============
st.subheader("5) Bảng trọng số & Benchmark cho từng metric (0–1)")
w_map = dict(zip(weights_df["Criteria"], weights_df["Weight"]))
bench_df_default = pd.DataFrame({
    "Metric": criteria,
    "Weight": [float(w_map.get(m, 0.0)) for m in criteria],
    "Benchmark tốt (100%) (0–1)": [None for _ in criteria],
    "Benchmark xấu (0%) (0–1)": [None for _ in criteria],
})
st.caption("Nhập benchmark theo **thang 0–1**. Ví dụ 0.05 = 5%. Với metric 'thấp là tốt' (vd CPI), đặt Best < Worst.")

edited = st.data_editor(
    bench_df_default,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Weight": st.column_config.NumberColumn("Trọng số", step=0.001, help="Trọng số AHP (read-only)"),
        "Benchmark tốt (100%) (0–1)": st.column_config.NumberColumn("Benchmark tốt (100%) (0–1)", step=0.001),
        "Benchmark xấu (0%) (0–1)": st.column_config.NumberColumn("Benchmark xấu (0%) (0–1)", step=0.001),
    },
    disabled=["Metric", "Weight"],
)

st.download_button(
    "⬇️ Tải benchmarks.csv",
    data=edited.to_csv(index=False).encode("utf-8"),
    file_name="benchmarks.csv", mime="text/csv",
)

# ==============
# 6) Scoring
# ==============
st.subheader("6) Tính điểm cho từng camp (chuẩn hoá theo benchmark)")
camp_df = df[[first_col] + criteria].copy()
# to numeric
for c in criteria:
    camp_df[c] = pd.to_numeric(camp_df[c], errors="coerce")

# warn missing rows
_missing_rows = camp_df[camp_df[criteria].isna().any(axis=1)][[first_col] + criteria]
if not _missing_rows.empty:
    st.warning("Một số camp thiếu dữ liệu ở 1+ metric → Total Score có thể NaN/không xếp hạng.")
    st.dataframe(_missing_rows, use_container_width=True)

# maps
best_map = {}; worst_map = {}
if edited is not None and not edited.empty:
    for _, row in edited.iterrows():
        best_map[row["Metric"]]  = row["Benchmark tốt (100%) (0–1)"]
        worst_map[row["Metric"]] = row["Benchmark xấu (0%) (0–1)"]

# validate benches
missing_bench = [m for m in criteria if (pd.isna(best_map.get(m)) or pd.isna(worst_map.get(m)))]
equal_bench = [m for m in criteria if (
    not pd.isna(best_map.get(m)) and not pd.isna(worst_map.get(m)) and float(best_map[m]) == float(worst_map[m])
)]
if missing_bench:
    st.warning(f"Thiếu benchmark cho: {', '.join(missing_bench)} → hãy nhập Best/Worst.")
    st.stop()
if equal_bench:
    st.error(f"Best == Worst ở: {', '.join(equal_bench)} → đặt Best khác Worst.")
    st.stop()

# scores per metric (0–100)
scored = camp_df[[first_col]].copy()
for m in criteria:
    best = float(best_map[m]); worst = float(worst_map[m])
    vals = camp_df[m].astype(float)
    if best > worst:  # higher is better
        s = (vals - worst) / (best - worst) * 100.0
    else:             # lower is better
        s = (worst - vals) / (worst - best) * 100.0
    scored[f"Score_{m}"] = s.clip(0.0, 100.0)

# contributions & total
contrib = camp_df[[first_col]].copy()
total = 0
for m in criteria:
    w = float(w_map.get(m, 0.0))
    contrib[f"Contrib_{m}"] = scored[f"Score_{m}"] * w
    total = total + contrib[f"Contrib_{m}"]
contrib["Total Score"] = total
contrib["Rank"] = contrib["Total Score"].rank(ascending=False, method="dense").astype("Int64")

# final table
final = (
    camp_df[[first_col] + criteria]
    .merge(scored, on=first_col, how="left")
    .merge(contrib[[first_col, "Total Score", "Rank"]], on=first_col, how="left")
    .sort_values(["Total Score", first_col], ascending=[False, True])
    .reset_index(drop=True)
)

# Add Recommendation column
def _recommend(total):
    if pd.isna(total): return "Review"
    if total > 85: return "Scale"
    if total >= 70: return "Optimize"
    if total < 20: return "Stop"
    return "Review"
final["Recommendation"] = final["Total Score"].apply(_recommend)

st.markdown("**Bảng Score chi tiết** (giá trị metric ở thang 0–1; Score ở thang 0–100)")
st.dataframe(final, use_container_width=True)

st.download_button(
    "⬇️ Tải camp_scores.csv",
    data=final.to_csv(index=False).encode("utf-8"),
    file_name="camp_scores.csv", mime="text/csv"
)

# Explanation for top camp
if not final.empty:
    top_id = final.loc[0, first_col]
    st.markdown(f"### 🔎 Giải thích cách tính điểm (Top camp: **{top_id}**)")
    top_vals = camp_df.loc[camp_df[first_col] == top_id, criteria].iloc[0]
    rows = []
    for m in criteria:
        best = float(best_map[m]); worst = float(worst_map[m])
        val = float(top_vals[m])
        if best > worst:
            raw_score = (val - worst) / (best - worst) * 100.0
            orientation = "cao là tốt (Best > Worst)"
            formula = f"({val:.4f} - {worst:.4f}) / ({best:.4f} - {worst:.4f}) × 100"
        else:
            raw_score = (worst - val) / (worst - best) * 100.0
            orientation = "thấp là tốt (Best < Worst)"
            formula = f"({worst:.4f} - {val:.4f}) / ({worst:.4f} - {best:.4f}) × 100"
        clipped = max(0.0, min(100.0, raw_score))
        w = float(w_map.get(m, 0.0))
        rows.append({
            "Metric": m,
            "Best": best,
            "Worst": worst,
            "Value (top)": val,
            "Hướng": orientation,
            "Score (0–100)": clipped,
            "Weight": w,
            "Contribution": clipped * w,
            "Công thức": formula,
        })
    explain_df = pd.DataFrame(rows)
    st.dataframe(explain_df, use_container_width=True)
    st.caption("Score cắt ngưỡng về [0,100], nhân trọng số để ra Contribution; tổng các Contribution là Total Score.")

st.divider()
st.caption("Nếu CR ≥ 0.10 ⇒ cân nhắc điều chỉnh so sánh cặp để giảm mâu thuẫn.")

# ==============
# 7) Summary & Slack
# ==============
st.subheader("7) Tạo tóm tắt & gửi Slack")

# Project name (dùng trong header tóm tắt)
project_name = st.text_input("Tên dự án", value="iOS_Heart_Rate")

# Lấy hôm nay (theo hệ thống chạy app)
today_str = datetime.date.today().strftime("%d-%m-%Y")

# Tạo list camp theo khuyến nghị
scale_ids = final.loc[final["Recommendation"].isin(["Scale","Scaled"]) , first_col].astype(str).tolist()
stop_ids  = final.loc[final["Recommendation"].str.lower() == "stop", first_col].astype(str).tolist()

def _format_list(ids):
    if not ids:
        return "- (không có)"
    return "\n".join([f"- {cid}" for cid in ids])

summary_text = (
    f"📅 Khuyến nghị ngày hôm nay ({today_str}) cho dự án *{project_name}*\n\n"
    f"🔼 *Campaign nên Scale:*\n"
    f"{_format_list(scale_ids)}\n\n"
    f"⛔ *Campaign nên Stop:*\n"
    f"{_format_list(stop_ids)}"
)

st.markdown("**Nội dung tóm tắt sẽ gửi:**")
st.code(summary_text, language="markdown")

with st.expander("Tùy chọn gửi Slack (DM theo email)", expanded=True):
    st.caption("Yêu cầu cài thư viện: `pip install slack_sdk`. Token là Bot token dạng `xoxb-...`.")
    slack_token = st.text_input("Slack Bot Token", type="password", help="Không lưu trữ. Dùng tạm thời để gửi tin.")
    email_target = st.text_input("Email người nhận", value="giangbh@ikameglobal.com")

    send_btn = st.button("📨 Gửi tóm tắt lên Slack")
    if send_btn:
        if not slack_token or not email_target:
            st.error("Vui lòng nhập đầy đủ Token và Email người nhận.")
        else:
            try:
                try:
                    from slack_sdk import WebClient
                    from slack_sdk.errors import SlackApiError
                except ImportError:
                    import subprocess, sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "slack_sdk"])
                    from slack_sdk import WebClient
                    from slack_sdk.errors import SlackApiError

                client = WebClient(token=slack_token)

                # 1) Tra user theo email
                user = client.users_lookupByEmail(email=email_target)["user"]
                user_id = user["id"]

                # 2) Mở DM (nếu chưa có)
                ch = client.conversations_open(users=[user_id])["channel"]["id"]

                # 3) Gửi tin nhắn
                resp = client.chat_postMessage(channel=ch, text=summary_text)
                st.success(f"Đã gửi thành công! ts={resp['ts']}")
            except ImportError:
                st.error("Chưa cài `slack_sdk`. Hãy chạy: pip install slack_sdk")
            except SlackApiError as e:
                try:
                    err = e.response.get("error")
                except Exception:
                    err = str(e)
                st.error(f"Lỗi Slack: {err}")
            except Exception as e:
                st.error(f"Lỗi không xác định: {e}")
