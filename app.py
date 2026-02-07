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
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide")

# カスタムCSS
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .news-box { background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #007bff; margin-bottom: 12px; }
    .advice-box { padding: 15px; border-radius: 10px; margin-top: 10px; font-weight: bold; border: 1px solid #ddd; }
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

# --- 5. 検索・設定エリア ---
st.subheader("🔍 銘柄を選んで診断")

stock_presets = {
    "🇺🇸 米国人気株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
    "🇯🇵 日本人気株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
    "⚡ 暗号資産/他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
}
all_stocks = {}
for cat, items in stock_presets.items(): all_stocks.update(items)

selected_names = st.multiselect("リストから選択（複数OK）", list(all_stocks.keys()), default=["エヌビディア", "トヨタ"])
custom_symbol = st.text_input("✍️ 自由に入力 (例: NFLX, 6752.T)", "")
if custom_symbol:
    custom_name = f"自由入力({custom_symbol})"
    all_stocks[custom_name] = custom_symbol
    if custom_name not in selected_names: selected_names.append(custom_name)

set1, set2 = st.columns(2)
with set1:
    future_investment = st.number_input("投資金額(円)", min_value=1000, value=100000)
with set2:
    time_span = st.select_slider("グラフ期間", options=["1週間", "30日", "1年", "5年"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y"}

execute = st.button("🚀 総合診断を実行")

# --- 広告エリア ---
st.markdown("---")
st.write("### 💡 おすすめ投資サービス")
link_dmm = "https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY"
link_tossy = "https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y"

ad_col1, ad_col2 = st.columns(2)
with ad_col1:
    st.markdown(f'<div style="padding: 10px; border: 1px solid #eee; border-radius: 10px; background-color: #ffffff; text-align: center;"><p style="font-size: 0.8em; color: #666; font-weight: bold;">📊 証券口座なら</p><a href="{link_dmm}" target="_blank" rel="nofollow" style="text-decoration: none; color: #007bff; font-weight: bold;"><div style="padding: 10px; border: 1px dashed #007bff; border-radius: 5px; margin-bottom: 5px;">DMM 株 で口座開設</div></a><p style="font-size: 0.7em; color: #999;">[広告：PR]</p></div>', unsafe_allow_html=True)
with ad_col2:
    st.markdown(f'<div style="padding: 10px; border: 1px solid #eee; border-radius: 10px; background-color: #ffffff; text-align: center;"><p style="font-size: 0.8em; color: #666; font-weight: bold;">📱 投資アプリなら</p><a href="{link_tossy}" target="_blank" rel="nofollow" style="text-decoration: none; color: #28a745; font-weight: bold;"><div style="padding: 10px; border: 1px dashed #28a745; border-radius: 5px; margin-bottom: 5px;">ウルトラ投資アプリ【TOSSY】</div></a><p style="font-size: 0.7em; color: #999;">[広告：PR]</p></div>', unsafe_allow_html=True)

st.markdown("---")

# --- 6. 実行ロジック ---
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
                
                # 予測
                current_price = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model_lr = LinearRegression().fit(X_reg, y_reg)
                pred_p = float(model_lr.predict([[len(y_reg)]])[0][0])
                
                # ニュース
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
                
                # アドバイス
                trend_up = pred_p > current_price
                if avg_stars >= 3.5 and trend_up: advice, color = "🌟【絶好調】勢いに乗っています！", "#e8f5e9"
                elif avg_stars <= 2.5 and not trend_up: advice, color = "⚠️【警戒】今は静観が良さそうです。", "#ffebee"
                else: advice, color = "😐【様子見】大きな動きを待っています。", "#f5f5f5"

                results.append({"銘柄": name, "将来価値": future_investment * (pred_p / current_price), "評価": avg_stars, "pred": pred_p, "news": news_details, "symbol": symbol, "advice": advice, "color": color})
            except: continue

    if results:
        st.subheader("📈 トレンド予測グラフ")
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, data in plot_data.items():
            base_p = data['Close'].iloc[0]
            norm_p = data['Close'] / base_p * 100
            line = ax.plot(data.index, norm_p, label=name, linewidth=2.5)
            res_item = next(r for r in results if r['銘柄'] == name)
            norm_pred = (res_item['pred'] / base_p) * 100
            ax.scatter(data.index[-1] + timedelta(days=1), norm_pred, color=line[0].get_color(), marker='*', s=200)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        st.pyplot(fig)

        # 画像保存ボタン
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button(label="📈 予測グラフを画像として保存", data=buf.getvalue(), file_name="ai_prediction.png", mime="image/png")

        st.markdown("---")
        st.subheader("🏆 AI診断詳細")
        for res in results:
            with st.expander(f"📌 {res['銘柄']} の結果を見る", expanded=True):
                st.metric("明日への予測額", f"{res['将来価値']:,.0f}円")
                st.markdown(f"<div class='advice-box' style='background-color: {res['color']};'>{res['advice']}</div>", unsafe_allow_html=True)
                for n in res['news']:
                    st.markdown(f"<div class='news-box'>{'⭐' * n['score']}<br><a href='{n['link']}' target='_blank'><b>🔗 {n['title_jp']}</b></a></div>", unsafe_allow_html=True)

        # シェアボタン   
        st.markdown("---")
        share_stock = selected_names[0] if selected_names else "注目銘柄"
        share_text = urllib.parse.quote(f"AIが「{share_stock}」を診断しました！🤖📈 #米国株 #AI投資診断 #アイモン")
        app_url = "https://stock-app-azmusn5x6drgnr4pacvp8s.streamlit.app/#156eb3e3" # ★自分のURLに書き換え
        share_url = f"https://twitter.com/intent/tweet?text={share_text}&url={app_url}"
        st.components.v1.html(f'<a href="{share_url}" target="_blank"><button style="width: 100%; padding: 15px; background-color: #000000; color: white; border: none; border-radius: 30px; font-size: 18px; font-weight: bold; cursor: pointer;">𝕏 (Twitter) でシェアして応援する</button></a>', height=80)
    else:
        st.info("銘柄を選んでボタンを押してください。")
 st.info("""
        📢 **画像付きでシェアする方法：**
        1. 上の『予測グラフを画像として保存』ボタンで画像をダウンロード
        2. 『𝕏 でシェア』ボタンを押し、投稿画面でその画像を貼り付けてね！
        """)     
        
# --- 7. 免責事項 ---
st.markdown("---")
st.markdown("""
    <div style="font-size: 0.8em; color: #666; background-color: #f1f3f5; padding: 20px; border-radius: 10px;">
        <b>【免責事項】</b><br>投資の最終決定はご自身の判断で行ってください。本アプリの利用による損害について開発者は一切の責任を負いません。
    </div>
    <p style='text-align: center; color: #999; font-size: 0.7em; margin-top:10px;'>© 2026 AI Market Diagnosis Pro</p>
""", unsafe_allow_html=True)
