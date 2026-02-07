import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import urllib.parse
import numpy as np
from datetime import timedelta
from deep_translator import GoogleTranslator
import io

# --- 0. グラフ表示の安定化設定 ---
import matplotlib
matplotlib.use('Agg')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide", page_icon="🤖")

# カスタムCSS（見やすさ重視）
st.markdown("""
    <style>
    .main-step { color: #007bff; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 15px; border: 1px solid #ddd; }
    .news-box { background-color: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #eee; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .advice-box { padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; border: 2px solid #ddd; }
    .stButton > button { width: 100%; border-radius: 30px; height: 3.5em; background: linear-gradient(45deg, #007bff, #00c6ff); color: white; font-weight: bold; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AIモデルの準備 ---
@st.cache_resource
def load_ai():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
analyzer = load_ai()

# --- 3. 指標データの取得 ---
@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                current = float(info['Close'].iloc[-1])
                prev = float(info['Close'].iloc[-2])
                data[name] = (current, current - prev)
            else: data[name] = (None, None)
        except: data[name] = (None, None)
    return data

indices_data = get_market_indices()

# --- 4. メイン画面：ヘッダー ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIがニュースと価格トレンドから、明日の市場を予測します。")

# 市場概況（直感的な色分け）
m_col1, m_col2, m_col3 = st.columns(3)
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None: col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else: col.metric(label, "取得中...", "市場休止中")

display_metric(m_col1, "💴 ドル/円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

st.markdown("---")

# --- 5. 操作ステップ案内 ---
st.markdown("<div class='main-step'>STEP 1: 診断したい銘柄を選ぼう</div>", unsafe_allow_html=True)

stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
    "⚡ その他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
}
all_stocks = {}
for cat, items in stock_presets.items(): all_stocks.update(items)

selected_names = st.multiselect("気になる銘柄をタップ（複数可）", list(all_stocks.keys()), default=["エヌビディア"])

with st.expander("➕ 自分で銘柄コードを入力する"):
    custom_symbol = st.text_input("例: NFLX (Netflix) や 6752.T (パナソニック)", "")
    if custom_symbol:
        custom_name = f"入力({custom_symbol})"
        all_stocks[custom_name] = custom_symbol
        if custom_name not in selected_names: selected_names.append(custom_name)

st.markdown("<div class='main-step'>STEP 2: 条件を決めよう</div>", unsafe_allow_html=True)
set1, set2 = st.columns(2)
with set1:
    future_investment = st.number_input("シミュレーション金額(円)", min_value=1000, value=100000, help="この金額を投資した場合、明日いくらになるか予測します")
with set2:
    time_span = st.select_slider("分析する期間", options=["1週間", "30日", "1年", "5年"], value="30日", help="過去のどの期間を元に分析するか選べます")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y"}

execute = st.button("🚀 AI診断スタート！")

# --- 6. 実行ロジック ---
if execute:
    results = []
    plot_data = {}
    
    with st.spinner('AIが世界中のニュースを読み込み中...'):
        for name in selected_names:
            try:
                symbol = all_stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                
                # 予測計算
                current_price = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model_lr = LinearRegression().fit(X_reg, y_reg)
                pred_p = float(model_lr.predict([[len(y_reg)]])[0][0])
                
                # ニュース取得
                is_j = ".T" in symbol
                search_q = name.split("(")[-1].replace(")", "") if "入力" in name else (name if is_j else symbol)
                url_news = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url_news)
                
                news_details, stars_sum = [], 0
                if feed.entries:
                    for entry in feed.entries[:3]:
                        score = int(analyzer(entry.title)[0]['label'].split()[0])
                        stars_sum += score
                        title_jp = GoogleTranslator(source='en', target='ja').translate(entry.title) if not is_j else entry.title
                        news_details.append({"title_jp": title_jp, "score": score, "link": entry.link})
                    avg_stars = stars_sum / len(news_details)
                else: avg_stars = 3
                
                # アドバイス
                trend_up = pred_p > current_price
                if avg_stars >= 3.5 and trend_up: advice, color = "🌟【超ポジティブ】ニュースも価格も上昇中！", "#e8f5e9"
                elif avg_stars <= 2.5 and not trend_up: advice, color = "⚠️【警戒が必要】ニュース・価格共に弱気です。", "#ffebee"
                else: advice, color = "😐【様子見】今ははっきりしたトレンドがありません。", "#f5f5f5"

                results.append({"銘柄": name, "将来価値": future_investment * (pred_p / current_price), "評価": avg_stars, "pred": pred_p, "news": news_details, "advice": advice, "color": color})
            except: continue

    if results:
        st.markdown("<div class='main-step'>STEP 3: 診断結果を確認しよう</div>", unsafe_allow_html=True)
        
        # グラフセクション
        with st.container():
            st.subheader("📈 トレンド予測グラフ")
            st.write("過去の動きから、明日の「★マーク」を予測しました。")
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, data in plot_data.items():
                base_p = data['Close'].iloc[0]
                norm_p = data['Close'] / base_p * 100
                line = ax.plot(data.index, norm_p, label=name, linewidth=2)
                res_item = next(r for r in results if r['銘柄'] == name)
                norm_pred = (res_item['pred'] / base_p) * 100
                ax.scatter(data.index[-1] + timedelta(days=1), norm_pred, color=line[0].get_color(), marker='*', s=200, zorder=5)
            ax.legend()
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            st.download_button("📸 グラフ画像を保存する", data=buf.getvalue(), file_name="ai_graph.png", mime="image/png")

        st.markdown("---")
        
        # 銘柄ごとの詳細
        for res in results:
            with st.container():
                st.markdown(f"### 🎯 {res['銘柄']} の診断結果")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    diff = res['将来価値'] - future_investment
                    st.metric("明日の予想資産額", f"{res['将来価値']:,.0f}円", f"{diff:+,.0f}円")
                
                with col_res2:
                    st.markdown(f"<div class='advice-box' style='background-color: {res['color']};'>{res['advice']}</div>", unsafe_allow_html=True)
                
                st.write("**AIが読んだ関連ニュース:**")
                for n in res['news']:
                    st.markdown(f"<div class='news-box'>{'⭐' * n['score']} <a href='{n['link']}' target='_blank'>{n['title_jp']}</a></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

        # シェア
        st.subheader("📢 友達に教える")
        share_text = urllib.parse.quote(f"AI診断結果：{results[0]['銘柄']}は{results[0]['advice']} 🤖📈")
        st.components.v1.html(f"""
            <a href="https://twitter.com/intent/tweet?text={share_text}" target="_blank">
                <button style="width:100%; padding:15px; background:#000; color:#fff; border-radius:30px; border:none; cursor:pointer;">𝕏 でシェアする</button>
            </a>
        """, height=70)
        st.info("💡 保存したグラフ画像を添付してポストするのがおすすめです！")

# --- 7. 免責事項 ---
st.markdown("---")
st.markdown("""
    <div style="font-size: 0.8em; color: #666; background-color: #f1f3f5; padding: 20px; border-radius: 10px;">
        <b>⚠️ 使う前に読んでね（免責事項）</b><br>
        このアプリはAIの予測を表示するもので、利益を保証するものではありません。実際の投資は自己責任でお願いします！
    </div>
    <p style='text-align: center; color: #999; font-size: 0.7em; margin-top:10px;'>© 2026 AI Market Diagnosis Pro</p>
""", unsafe_allow_html=True)
