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

# カスタムCSS（ダークモード対応版）
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }
    div[data-testid="stMetric"] {
        background-color: rgba(150, 150, 150, 0.1);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(150, 150, 150, 0.3);
    }
    .news-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(150, 150, 150, 0.5);
        margin-bottom: 10px;
    }
    .news-box a {
        text-decoration: none;
        color: #4dabf7 !important;
    }
    .advice-box {
        padding: 20px;
        border-radius: 15px;
        margin-top: 10px;
        font-size: 1.1em;
        text-align: center;
        border: 2px solid rgba(150, 150, 150, 0.3);
        color: #1a1a1a;
    }
    .ad-card {
        padding: 15px;
        border: 1px solid rgba(150, 150, 150, 0.3);
        border-radius: 10px;
        background-color: rgba(150, 150, 150, 0.05);
        text-align: center;
    }
    .span-hint {
        background-color: rgba(49, 130, 206, 0.1);
        padding: 12px;
        border-radius: 10px;
        font-size: 0.9em;
        border-left: 5px solid #3182ce;
        margin-bottom: 20px;
    }
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

# ⚠️ ここがエラーの原因になりやすいポイント！丁寧に記述しました
stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
    "⚡ その他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
}

all_stocks = {}
for cat, items in stock_presets.items():
    all_stocks.update(items)

selected_names = st.multiselect("気になる銘柄をタップ（複数可）", list(all_stocks.keys()), default=["エヌビディア"])

st.markdown("<div class='main-step'>STEP 2: 条件を決めよう</div>", unsafe_allow_html=True)
set1, set2 = st.columns(2)
with set1:
    future_investment = st.number_input("シミュレーション金額(円)", min_value=1000, value=100000)
with set2:
    time_span = st.select_slider("分析する期間", options=["1週間", "30日", "1年", "5年", "10年", "最大期間"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y", "10年": "10y", "最大期間": "max"}

span_hints = {
    "1週間": "🚀 **短期予測モード**: 直近の動きを重視します。",
    "30日": "📊 **中期予測モード**: 1ヶ月の流れを重視します。",
    "1年": "🐢 **長期予測モード**: 年間のトレンドを重視します。",
    "5年": "🏔️ **超長期予測モード**: 数年の大きなうねりを重視します。",
    "10年": "🏛️ **歴史的トレンドモード**: 10年間の成長性を分析します。",
    "最大期間": "♾️ **全歴史分析モード**: 上場来のすべてを考慮します。"
}
st.markdown(f"<div class='span-hint'>{span_hints[time_span]}<br>※期間を長くすると大きな流れが見えてきます。</div>", unsafe_allow_html=True)

execute = st.button("🚀 AI診断スタート！")

# --- 広告エリア ---
st.markdown("---")
st.write("### 💡 おすすめ投資サービス")
link_dmm = "https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY"
link_tossy = "https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y"

ad_col1, ad_col2 = st.columns(2)
with ad_col1:
    st.markdown(f'<div class="ad-card"><p style="font-weight: bold;">📊 証券口座なら</p><a href="{link_dmm}" target="_blank" rel="nofollow" style="text-decoration: none; color: #4dabf7; font-weight: bold;">DMM 株 で口座開設</a><p style="font-size: 0.7em; opacity: 0.7;">[広告：PR]</p></div>', unsafe_allow_html=True)
with ad_col2:
    st.markdown(f'<div class="ad-card"><p style="font-weight: bold;">📱 投資アプリなら</p><a href="{link_tossy}" target="_blank" rel="nofollow" style="text-decoration: none; color: #51cf66; font-weight: bold;">ウルトラ投資アプリ【TOSSY】</a><p style="font-size: 0.7em; opacity: 0.7;">[広告：PR]</p></div>', unsafe_allow_html=True)

st.markdown("---")

# --- 6. 実行ロジック ---
if execute:
    results = []
    plot_data = {}
    
    with st.spinner(f'過去 {time_span} のデータを分析中...'):
        for name in selected_names:
            try:
                symbol = all_stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                
                current_price = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model_lr = LinearRegression().fit(X_reg, y_reg)
                pred_p = float(model_lr.predict([[len(y_reg)]])[0][0])
                
                is_j = ".T" in symbol
                search_q = name if is_j else symbol
                url_news = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url_news)
                
                news_details = []
                stars_sum = 0
                if feed.entries:
                    for entry in feed.entries[:3]:
                        score = int(analyzer(entry.title)[0]['label'].split()[0])
                        stars_sum += score
                        title_jp = GoogleTranslator(source='en', target='ja').translate(entry.title) if not is_j else entry.title
                        news_details.append({"title_jp": title_jp, "score": score, "link": entry.link})
                    avg_stars = stars_sum / len(news_details)
                else: avg_stars = 3
                
                trend_up = pred_p > current_price
                if avg_stars >= 3.5 and trend_up: advice, color = f"🌟【{time_span}：強気】", "#d4edda"
                elif avg_stars <= 2.5 and not trend_up: advice, color = f"⚠️【{time_span}：警戒】", "#f8d7da"
                else: advice, color = f"😐【{time_span}：様子見】", "#e2e3e5"

                results.append({"銘柄": name, "将来価値": future_investment * (pred_p / current_price), "評価": avg_stars, "pred": pred_p, "news": news_details, "advice": advice, "color": color})
            except: continue

    if results:
        st.markdown("<div class='main-step'>STEP 3: 診断結果を確認しよう</div>", unsafe_allow_html=True)
        
        with st.container():
            st.subheader(f"📈 {time_span}間のトレンド予測グラフ")
            if st.get_option("theme.base") == "dark": plt.style.use('dark_background')
            else: plt.style.use('default')
            japanize_matplotlib.japanize()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, data in plot_data.items():
                base_p = data['Close'].iloc[0]
                norm_p = data['Close'] / base_p * 100
                line = ax.plot(data.index, norm_p, label=name, linewidth=2.5)
                res_item = next(r for r in results if r['銘柄'] == name)
                norm_pred = (res_item['pred'] / base_p) * 100
                ax.scatter(data.index[-1] + timedelta(days=1), norm_pred, color=line[0].get_color(), marker='*', s=250, edgecolors='white', zorder=10)
            ax.legend()
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            st.download_button(f"📸 予測グラフを画像保存", data=buf.getvalue(), file_name=f"ai_forecast_{time_span}.png", mime="image/png")

        st.markdown("---")
        for res in results:
            with st.container():
                st.markdown(f"### 🎯 {res['銘柄']} の診断詳細")
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    diff = res['将来価値'] - future_investment
                    st.metric(f"予想額({time_span})", f"{res['将来価値']:,.0f}円", f"{diff:+,.0f}円")
                with col_res2:
                    st.markdown(f"<div class='advice-box' style='background-color: {res['color']};'>{res['advice']}</div>", unsafe_allow_html=True)
                st.write("**AIが分析した最新ニュース:**")
                for n in res['news']:
                    st.markdown(f"<div class='news-box'>{'⭐' * n['score']} <a href='{n['link']}' target='_blank'><b>🔗 {n['title_jp']}</b></a></div>", unsafe_allow_html=True)

        st.subheader("📢 友達に教える")
        share_text = urllib.parse.quote(f"AI診断：{results[0]['銘柄']}は過去{time_span}の傾向から見ると「{results[0]['advice']}」🤖📈")
        st.components.v1.html(f'<a href="https://twitter.com/intent/tweet?text={share_text}" target="_blank"><button style="width:100%; padding:15px; background:#1DA1F2; color:#fff; border-radius:30px; border:none; cursor:pointer; font-weight:bold;">𝕏 でシェアして応援する</button></a>', height=70)

st.markdown("---")
st.markdown('<div style="font-size: 0.8em; opacity: 0.8; background-color: rgba(150, 150, 150, 0.1); padding: 20px; border-radius: 10px;"><b>⚠️ ご利用上の注意</b><br>分析期間により予測は大きく変動します。実際の投資は自己責任でお願いします。</div>', unsafe_allow_html=True)
