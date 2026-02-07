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
import google.generativeai as genai

# --- 0. AIチャットの設定 ---
# StreamlitのSecretsに保存したキーを読み込みます
try:
    GOOGLE_API_KEY = st.secrets["AIzaSyC4kqvsdMNVr1tIHFLIDSSZa4oudBtki5g"]
except:
    # Secrets未設定時の予備（ここに直接貼っても動きますが、公開時はSecrets推奨）
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

genai.configure(api_key=GOOGLE_API_KEY)
model_chat = genai.GenerativeModel('gemini-pro')

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット総合診断 Pro", layout="wide", page_icon="🤖")

# カスタムCSS（ダークモード対応・スマホ最適化・免責事項デザイン）
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }
    div[data-testid="stMetric"] { background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }
    .news-box { padding: 12px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.5); margin-bottom: 10px; }
    .news-box a { text-decoration: none; color: #4dabf7 !important; }
    .advice-box { padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; border: 2px solid rgba(150, 150, 150, 0.3); color: #1a1a1a; }
    
    /* 広告コンテナ（スマホで縦、PCで横） */
    .ad-container { display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin: 20px 0; }
    .ad-card { flex: 1; min-width: 280px; max-width: 500px; padding: 20px; border: 2px dashed rgba(150, 150, 150, 0.5); border-radius: 15px; background-color: rgba(150, 150, 150, 0.05); text-align: center; }
    
    .span-hint { background-color: rgba(49, 130, 206, 0.1); padding: 12px; border-radius: 10px; font-size: 0.9em; border-left: 5px solid #3182ce; margin-bottom: 20px; }
    
    /* 免責事項・アフィリエイト明記スタイル */
    .disclaimer-box { font-size: 0.8em; opacity: 0.8; background-color: rgba(150, 150, 150, 0.1); padding: 20px; border-radius: 10px; line-height: 1.6; margin-top: 50px; border: 1px solid rgba(150, 150, 150, 0.2); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. サイドバー：本格AI対話チャット ---
with st.sidebar:
    st.title("🗨️ アイモン投資相談室")
    st.write("株や経済の疑問に答えるよ！")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 履歴表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 入力欄
    if prompt := st.chat_input("例：利下げって株にどう影響する？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                full_prompt = f"あなたは親切な投資アドバイザーの『アイモン』です。投資初心者の質問に対して、専門用語を避け、友だちのように優しく解説して。質問：{prompt}"
                response = model_chat.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except:
                st.error("APIキーを確認してください。設定直後は反映に時間がかかる場合があります。")

    if st.button("チャット履歴を消去"):
        st.session_state.messages = []
        st.rerun()

# --- 3. 感情分析モデル ---
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
analyzer = load_sentiment()

# --- 4. 指標データ取得 ---
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

# --- 5. メイン画面：ヘッダー ---
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

# 操作ステップ
st.markdown("<div class='main-step'>STEP 1: 診断したい銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL", "パランティア": "PLTR"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T", "三菱UFJ": "8306.T"},
    "⚡ その他": {"ビットコイン": "BTC-USD", "金(Gold)": "GC=F"}
}
all_stocks = {}
for cat, items in stock_presets.items(): all_stocks.update(items)
selected_names = st.multiselect("気になる銘柄を選択", list(all_stocks.keys()), default=["エヌビディア"])

st.markdown("<div class='main-step'>STEP 2: 条件を決めよう</div>", unsafe_allow_html=True)
set1, set2 = st.columns(2)
with set1:
    future_investment = st.number_input("シミュレーション金額(円)", min_value=1000, value=100000)
with set2:
    time_span = st.select_slider("分析する期間", options=["1週間", "30日", "1年", "5年", "10年", "最大期間"], value="30日")
    span_map = {"1週間": "7d", "30日": "1mo", "1年": "1y", "5年": "5y", "10年": "10y", "最大期間": "max"}

st.markdown(f"<div class='span-hint'>過去 {time_span} のデータをAIが読み込みます。</div>", unsafe_allow_html=True)
execute = st.button("🚀 AI診断スタート！")

# --- おすすめ投資サービス（広告エリア） ---
st.markdown("---")
st.write("### 💡 おすすめ投資サービス")
link_dmm = "https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY"
link_tossy = "https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y"

st.markdown(f"""
<div class="ad-container">
    <div class="ad-card">
        <p style="font-weight: bold;">📊 証券口座なら</p>
        <a href="{link_dmm}" target="_blank" rel="nofollow" style="text-decoration: none;">
            <div style="padding: 15px; background: #4dabf7; color: white; border-radius: 10px; font-weight: bold;">DMM 株 で口座開設</div>
        </a>
        <p style="font-size: 0.8em; opacity: 0.7; margin-top: 10px;">スマホで最短当日取引可能！[PR]</p>
    </div>
    <div class="ad-card">
        <p style="font-weight: bold;">📱 投資アプリなら</p>
        <a href="{link_tossy}" target="_blank" rel="nofollow" style="text-decoration: none;">
            <div style="padding: 15px; background: #51cf66; color: white; border-radius: 10px; font-weight: bold;">投資アプリ TOSSY</div>
        </a>
        <p style="font-size: 0.8em; opacity: 0.7; margin-top: 10px;">資産管理をもっと身近に。[PR]</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. 実行ロジック ---
if execute:
    results, plot_data = [], {}
    with st.spinner('AIが市場を分析中...'):
        for name in selected_names:
            try:
                symbol = all_stocks[name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                
                # トレンド予測
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                pred = float(LinearRegression().fit(X_reg, y_reg).predict([[len(y_reg)]])[0][0])
                
                # ニュース取得
                is_j = ".T" in symbol
                q = name if is_j else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url)
                
                news_list, stars = [], 0
                if feed.entries:
                    for e in feed.entries[:3]:
                        s = int(analyzer(e.title)[0]['label'].split()[0])
                        stars += s
                        title = GoogleTranslator(source='en', target='ja').translate(e.title) if not is_j else e.title
                        news_list.append({"title": title, "score": s, "link": e.link})
                    avg = stars / len(news_list)
                else: avg = 3

                # アドバイス生成
                up = pred > curr
                if avg >= 3.5 and up: adv, col = f"🌟【{time_span}：強気】", "#d4edda"
                elif avg <= 2.5 and not up: adv, col = f"⚠️【{time_span}：警戒】", "#f8d7da"
                else: adv, col = f"😐【{time_span}：様子見】", "#e2e3e5"

                results.append({"銘柄": name, "将来": future_investment * (pred / curr), "星": avg, "pred": pred, "news": news_list, "adv": adv, "col": col})
            except: continue

    if results:
        st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
        # グラフ
        fig, ax = plt.subplots(figsize=(10, 5))
        if st.get_option("theme.base") == "dark": plt.style.use('dark_background')
        japanize_matplotlib.japanize()
        for name, data in plot_data.items():
            base = data['Close'].iloc[0]
            line = ax.plot(data.index, data['Close']/base*100, label=name, linewidth=2.5)
            r = next(item for item in results if item['銘柄'] == name)
            ax.scatter(data.index[-1] + timedelta(days=1), (r['pred']/base)*100, color=line[0].get_color(), marker='*', s=250, edgecolors='white', zorder=10)
        ax.legend()
        st.pyplot(fig)

        for res in results:
            st.markdown(f"### 🎯 {res['銘柄']}")
            c1, c2 = st.columns([1, 2])
            c1.metric(f"予想額({time_span})", f"{res['将来']:,.0f}円", f"{res['将来']-future_investment:+,.0f}円")
            c2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
            st.write("**AI分析ニュース（星が多いほどポジティブ）:**")
            for n in res['news']:
                st.markdown(f"<div class='news-box'>{'⭐' * n['score']} <a href='{n['link']}' target='_blank'><b>🔗 {n['title']}</b></a></div>", unsafe_allow_html=True)

# --- 7. 免責事項（アフィリエイト明記含む） ---
st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ 免責事項・ご利用上の注意</b><br>
        ● <b>情報の性質について</b>：本アプリは情報の提供を目的としており、投資勧誘を意図したものではありません。表示される予測は過去のデータに基づくAIシミュレーションであり、将来の運用成果を示唆・保証するものではありません。<br>
        ● <b>投資判断について</b>：投資の最終決定は、利用者ご自身の判断と責任で行ってください。本アプリの利用により生じた直接的・間接的な損害について、開発者は一切の責任を負いません。<br>
        ● <b>広告について</b>：本アプリにはアフィリエイトプログラムが含まれており、掲載された広告リンクを経由してサービスに申し込まれた場合、開発者に報酬が支払われることがあります。提供される情報は常に最新かつ正確であるよう努めておりますが、リンク先のサービス内容、料金等については各公式サイトで必ずご確認ください。<br>
        ● <b>システムについて</b>：市場データの遅延やAI解析の誤りが発生する可能性があります。あらかじめご了承ください。
    </div>
    <p style='text-align: center; opacity: 0.5; font-size: 0.7em; margin-top:10px;'>© 2026 AI Market Diagnosis Pro | アフィリエイト広告を含みます</p>
""", unsafe_allow_html=True)
