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

# --- 0. グラフ表示の安定化設定 ---
import matplotlib
matplotlib.use('Agg')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide")

# カスタムCSS
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .news-box { background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #007bff; margin-bottom: 12px; }
    .advice-box { padding: 15px; border-radius: 10px; margin-top: 10px; font-weight: bold; border: 1px solid #ddd; }
    /* ボタンを横幅いっぱいに（スマホ用） */
    .stButton > button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; }
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
st.title("🌍 AIマーケット総合診断 Pro")

m_col1, m_col2, m_col3 = st.columns(3)
def display_metric(col, label, data_tuple, unit=""):
    val, diff = data_tuple
    if val is not None: col.metric(label, f"{val:,.2f}{unit}", f"{diff:+,.2f}")
    else: col.metric(label, "取得中...", "市場休止中")

display_metric(m_col1, "💴 ドル円", indices_data['ドル円'], "円")
display_metric(m_col2, "🇯🇵 日経平均", indices_data['日経平均'], "円")
display_metric(m_col3, "🇺🇸 NYダウ", indices_data['NYダウ'], "ドル")

st.markdown("---")

# --- 5. ★引っ越し：検索・設定エリアをメイン画面上部に配置 ---
st.subheader("🔍 銘柄を選んで診断")

# 銘柄リスト
stock_presets = {
    "🇺🇸 米国人気株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
    "🇯🇵 日本人気株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
    "⚡ 暗号資産/他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
}
all_stocks = {}
for cat, items in stock_presets.items(): all_stocks.update(items)

# スマホでも見やすい入力フォーム
selected_names = st.multiselect("リストから選択（複数OK）", list(all_stocks.keys()), default=["エヌビディア", "トヨタ"])
custom_symbol = st.text_input("✍️ 自由に入力 (例: NFLX, 6752.T)", "")
if custom_symbol:
    custom_name = f"自由入力({custom_symbol})"
    all_stocks[custom_name] = custom_symbol
    if custom_name not in selected_names: selected_names.append(custom_name)

# 詳細設定を1列に並べる（スマホだと自動で縦に並ぶ）
set1, set2 = st.columns(2)
with set1:
    future_investment = st.number_input("投資金額(円)", min_value=1000, value=100000)
with set2:
    time_span = st.select_slider("グラフ期間", options=["1週間", "30日", "1年", "5年"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y"}

# 診断実行ボタン（CSSで大きく表示）
execute = st.button("🚀 総合診断を実行")

# --- 広告エリア (DMM株) ---
st.markdown("---")
# A8.netの素材から抽出したリンクと画像
affiliate_link = "https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY"
banner_url = "https://www27.a8.net/svt/bgt?aid=260207137351&wid=001&eno=01&mid=s00000008903007010000&mc=1" # DMM株の一般的なバナーサイズを想定

st.markdown(f"""
    <div style="text-align: center; padding: 10px; background-color: #fcfcfc; border: 1px solid #eee; border-radius: 10px;">
        <p style="font-size: 0.8em; color: #666; margin-bottom: 8px;">＼ 株の取引を始めるなら ／</p>
        <a href="{affiliate_link}" target="_blank" rel="nofollow">
            <img src="https://www27.a8.net/svt/bgt?aid=260207137351&wid=001&eno=01&mid=s00000008903007010000&mc=1" 
                 style="width: 100%; max-width: 468px; border-radius: 5px;">
        </a>
        <p style="font-size: 0.7em; color: #999; margin-top: 5px;">[広告：PR] 株初心者から上級者まで、幅広く選ばれているDMM 株</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- 6. 実行ロジック（グラフと結果） ---
if execute:
    results = []
    plot_data = {}
    
    with st.spinner('AIが分析中...'):
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
                model = LinearRegression().fit(X_reg, y_reg)
                pred_p = float(model.predict([[len(y_reg)]])[0][0])
                
                # ニュース取得
                is_j = ".T" in symbol
                search_q = name.split("(")[-1].replace(")", "") if "自由入力" in name else (name if is_j else symbol)
                url_news = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url_news)
                
                news_details, stars_sum = [], 0
                if feed.entries:
                    for entry in feed.entries[:3]:
                        score = int(analyzer(entry.title)[0]['label'].split()[0])
                        stars_sum += score
                        title_jp = GoogleTranslator(source='en', target='ja').translate(entry.title) if not is_j else entry.title
                        news_details.append({"title_jp": title_jp, "title_en": entry.title, "score": score, "link": entry.link})
                    avg_stars = stars_sum / len(news_details)
                else: avg_stars = 3
                
                # アドバイス判定
                trend_up = pred_p > current_price
                if avg_stars >= 3.5 and trend_up: advice, color = "🌟【絶好調】勢いに乗っています！", "#e8f5e9"
                elif avg_stars <= 2.5 and not trend_up: advice, color = "⚠️【警戒】今は静観が良さそうです。", "#ffebee"
                elif avg_stars <= 2.5 and trend_up: advice, color = "🤔【チグハグ】悪材料に負けない買いがあります。", "#fff3e0"
                elif avg_stars >= 3.5 and not trend_up: advice, color = "❓【チグハグ】いい材料が無視されています。", "#e1f5fe"
                else: advice, color = "😐【様子見】大きな動きを待っています。", "#f5f5f5"

                results.append({"銘銘": name, "将来価値": future_investment * (pred_p / current_price), "評価": avg_stars, "pred": pred_p, "news": news_details, "symbol": symbol, "advice": advice, "color": color, "current": current_price})
            except: continue

    if results:
        # グラフ表示
        st.subheader("📈 トレンド予測グラフ")
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, data in plot_data.items():
            base_p = data['Close'].iloc[0]
            norm_p = data['Close'] / base_p * 100
            line = ax.plot(data.index, norm_p, label=name, linewidth=2.5)
            res_item = next(r for r in results if r['銘銘'] == name)
            norm_pred = (res_item['pred'] / base_p) * 100
            future_date = data.index[-1] + timedelta(days=1)
            ax.plot([data.index[-1], future_date], [norm_p.iloc[-1], norm_pred], color=line[0].get_color(), linestyle='--', alpha=0.5)
            ax.scatter(future_date, norm_pred, color=line[0].get_color(), marker='*', s=300, edgecolors='black', zorder=10)
        
        # 凡例をスマホでも見やすく（上部に配置）
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

        # 診断詳細
        st.markdown("---")
        st.subheader("🏆 AI診断詳細")
        for res in results:
            with st.expander(f"📌 {res['銘銘']} の結果を見る", expanded=True):
                st.metric("明日への予測額", f"{res['将来価値']:,.0f}円", f"{res['将来価値']-future_investment:+,.0f}円")
                st.markdown(f"<div class='advice-box' style='background-color: {res['color']};'>{res['advice']}</div>", unsafe_allow_html=True)
                st.write("**最新ニュース:**")
                for n in res['news']:
                    st.markdown(f"<div class='news-box'>{'⭐' * n['score']}<br><a href='{n['link']}' target='_blank'><b>🔗 {n['title_jp']}</b></a><br><small>{n['title_en']}</small></div>", unsafe_allow_html=True)
    else:
        st.info("銘柄を選んでボタンを押してください。")
        # --- 7. 免責事項（アプリの最下部に配置） ---
st.markdown("---")

# スタイリッシュな免責事項エリア
st.markdown("""
    <style>
    .disclaimer {
        font-size: 0.8em;
        color: #666;
        background-color: #f1f3f5;
        padding: 20px;
        border-radius: 10px;
        line-height: 1.6;
        margin-top: 30px;
    }
    </style>
    <div class="disclaimer">
        <b>【免責事項】</b><br>
        ● 本アプリが提供するデータ、AIによる分析結果、株価予測は情報の提供のみを目的としており、投資の勧誘や助言を目的としたものではありません。<br>
        ● 予測結果は過去のデータに基づく計算値であり、将来の運用成果を保証するものではありません。実際の市場動向は経済情勢や予期せぬイベントにより大きく異なる場合があります。<br>
        ● 本アプリに掲載されているリンク先（証券会社等）を通じて行われるサービスの利用については、利用者ご自身の判断と責任において行ってください。<br>
        ● 本アプリを利用したことにより生じたいかなる損害・不利益についても、開発者は一切の責任を負いません。<br>
        ● 投資に関する最終決定は、ご自身の判断と責任で行っていただくようお願いいたします。
    </div>
    <br>
    <p style='text-align: center; color: #999; font-size: 0.7em;'>© 2024 AI Market Diagnosis Pro - All Rights Reserved.</p>
    """, unsafe_allow_html=True)
