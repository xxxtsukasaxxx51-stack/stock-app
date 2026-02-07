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
import random

# --- 0. 基本設定 ---
# あなたのGitHub上の画像URL
CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"

# 投資の名言リスト
INVESTMENT_QUOTES = [
    "「ルール1：絶対にお金を損しないこと。ルール2：ルール1を絶対に忘れないこと」— ウォーレン・バフェット",
    "「あなたがパニックに陥って売っている時、誰かが笑って買っている」— 投資の格言",
    "「強気相場は、悲観の中に生まれ、懐疑の中に育ち、楽観とともに成熟し、幸福感の中で消えていく」— ジョン・テンプルトン",
    "「投資で一番大切なのは、頭脳ではなく、忍耐強さだ」— ピーター・リンチ",
    "「賢者は、愚者が最後にすること（売却）を最初にする（購入）」— ロスチャイルド",
    "「卵を一つのカゴに盛るな」— 投資の格言",
    "「分散投資は無知に対する防御だ」— ウォーレン・バフェット",
    "「暴落時は、最高の買い場である」— 投資の格言",
    "「市場が強欲な時に恐れ、市場が恐れている時に強欲になれ」— ウォーレン・バフェット",
    "「準備をしておかなかったチャンスは、ただのピンチである」— 投資の格言"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. CSS：キャラクター透過強化・デザイン ---
st.markdown(f"""
    <style>
    /* メインデザイン */
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
    div[data-testid="stMetric"] {{ background-color: rgba(150, 150, 150, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(150, 150, 150, 0.3); }}
    .news-box {{ padding: 12px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.5); margin-bottom: 10px; font-size: 0.9em; }}
    .advice-box {{ padding: 20px; border-radius: 15px; margin-top: 10px; font-size: 1.1em; text-align: center; border: 2px solid rgba(150, 150, 150, 0.3); }}
    
    /* ★キャラクターの白いフチを消すための特殊フィルタ★ */
    .floating-char-box {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
        display: flex;
        flex-direction: column;
        align-items: center;
        pointer-events: none;
    }}
    .char-img {{
        width: 140px;
        /* 背景色を透過させる乗算ブレンド */
        mix-blend-mode: multiply;
        /* 白い残骸を飛ばすためのコントラスト調整 */
        filter: contrast(125%) brightness(108%) drop-shadow(5px 5px 15px rgba(0,0,0,0.2));
        animation: float 3s ease-in-out infinite;
    }}
    .bubble {{
        background: white; border: 2px solid #3182ce; border-radius: 12px;
        padding: 5px 10px; margin-bottom: 5px; font-size: 0.8em; font-weight: bold; color: #1a202c;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }}

    /* 透明ボタンをキャラに被せる */
    div[data-testid="stPopover"] {{
        position: fixed; bottom: 20px; right: 20px; z-index: 1000;
    }}
    div[data-testid="stPopover"] > button {{
        width: 140px !important; height: 180px !important;
        background: transparent !important; color: transparent !important; border: none !important;
        box-shadow: none !important;
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-12px); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. キャラクターと名言ポップオーバー ---
st.markdown(f"""
    <div class="floating-char-box">
        <div class="bubble">名言を聴く？</div>
        <img src="{CHARACTER_URL}" class="char-img">
    </div>
    """, unsafe_allow_html=True)

with st.popover(""):
    st.markdown("### 📜 今日の投資格言")
    st.info(random.choice(INVESTMENT_QUOTES))
    if st.button("別の名言に入れ替える"):
        st.rerun()

# --- 4. メイン画面：市場指標 ---
st.title("🤖 AIマーケット総合診断 Pro")
st.caption("最新AIが市場を予測。右下のキャラをタップして名言をチェック！")

@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"ドル円": "JPY=X", "日経平均": "^N225", "NYダウ": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            info = yf.download(ticker, period="1mo", progress=False)
            if not info.empty:
                curr = info['Close'].iloc[-1]
                prev = info['Close'].iloc[-2]
                data[name] = (float(curr), float(curr - prev))
            else: data[name] = (None, None)
        except: data[name] = (None, None)
    return data

idx_data = get_market_indices()
m1, m2, m3 = st.columns(3)
def disp_m(col, lab, d, u=""):
    if d[0]: col.metric(lab, f"{d[0]:,.2f}{u}", f"{d[1]:+,.2f}")
    else: col.metric(lab, "取得中...", "")

disp_m(m1, "💴 ドル/円", idx_data['ドル円'], "円")
disp_m(m2, "🇯🇵 日経平均", idx_data['日経平均'], "円")
disp_m(m3, "🇺🇸 NYダウ", idx_data['NYダウ'], "ドル")

st.markdown("---")

# --- 5. 銘柄入力 (フリー入力対応) ---
st.markdown("<div class='main-step'>STEP 1: 銘柄を選ぼう</div>", unsafe_allow_html=True)
stock_presets = {
    "🇺🇸 米国株": {"テスラ": "TSLA", "エヌビディア": "NVDA", "Apple": "AAPL"},
    "🇯🇵 日本株": {"トヨタ": "7203.T", "ソニー": "6758.T", "任天堂": "7974.T"}
}
all_stocks_preset = {}
for items in stock_presets.values(): all_stocks_preset.update(items)

col_in1, col_in2 = st.columns([2, 1])
with col_in1:
    selected_names = st.multiselect("リストから選択", list(all_stocks_preset.keys()), default=["エヌビディア"])
with col_in2:
    free_input = st.text_input("ティッカー直接入力 (例: 9984.T, MSFT)", "")

# 銘柄リストの統合
final_targets = {name: all_stocks_preset[name] for name in selected_names}
if free_input:
    # 直接入力されたものをリストに追加
    clean_input = free_input.strip().upper()
    final_targets[clean_input] = clean_input

st.markdown("<div class='main-step'>STEP 2: 条件設定</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: f_inv = st.number_input("投資金額(円)", min_value=1000, value=100000, step=10000)
with c2: 
    time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年"], value="30日")
    span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y"}

execute = st.button("🚀 AIマーケット診断スタート！")

# 広告
st.markdown("""<div class="ad-container">
    <div class="ad-card">📊 証券口座なら<br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株 口座開設</a></div>
    <div class="ad-card">📱 投資アプリなら<br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">投資アプリ TOSSY</a></div>
</div>""", unsafe_allow_html=True)

# --- 6. 診断ロジック ---
if "sentiment_analyzer" not in st.session_state:
    # 感情分析AIのロード
    st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

if execute and final_targets:
    results, plot_data = [], {}
    with st.spinner('AIが市場データとニュースを読み取っています...'):
        for name, symbol in final_targets.items():
            try:
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                plot_data[name] = df
                
                # 線形回帰による簡易予測
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred = float(model.predict([[len(y_reg) + 5]])[0][0]) # 5ステップ先を予測
                
                # ニュース取得と感情分析
                is_j = ".T" in symbol
                q = name if is_j else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl={'ja' if is_j else 'en'}&gl={'JP' if is_j else 'US'}"
                feed = feedparser.parse(url)
                news_list, stars = [], 0
                
                if feed.entries:
                    for e in feed.entries[:3]:
                        # 感情分析 (1-5 stars)
                        s = int(st.session_state.sentiment_analyzer(e.title)[0]['label'].split()[0])
                        stars += s
                        # 英語ニュースの場合は翻訳
                        title = GoogleTranslator(source='en', target='ja').translate(e.title) if not is_j else e.title
                        news_list.append({"title": title, "score": s, "link": e.link})
                    avg_score = stars / len(news_list)
                else:
                    avg_score = 3 # ニュースがない場合は中立

                # 診断アルゴリズム
                is_up = pred > curr
                if avg_score >= 3.5 and is_up:
                    adv, col = "🌟【強気】AIもポジティブです", "#d4edda"
                elif avg_score <= 2.4 and not is_up:
                    adv, col = "⚠️【警戒】リスクが高い局面です", "#f8d7da"
                else:
                    adv, col = "😐【様子見】今は静観が良さそうです", "#e2e3e5"
                
                results.append({
                    "銘柄": name, "将来": f_inv * (pred / curr), 
                    "星": avg_score, "pred": pred, "news": news_list, 
                    "adv": adv, "col": col
                })
            except Exception as e:
                st.warning(f"{name} の分析中にエラーが発生しました。")
                continue

    if results:
        st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
        
        # 比較チャート
        fig, ax = plt.subplots(figsize=(10, 4))
        japanize_matplotlib.japanize()
        for name, data in plot_data.items():
            # 開始価格を100として正規化
            base = data['Close'].iloc[0]
            line = ax.plot(data.index, data['Close']/base*100, label=name, linewidth=2)
            # 予測地点を星印で表示
            r = next(item for item in results if item['銘柄'] == name)
            ax.scatter(data.index[-1] + timedelta(days=2), (r['pred']/base)*100, 
                       color=line[0].get_color(), marker='*', s=150, zorder=10)
        
        ax.set_ylabel("成長率 (%)")
        ax.legend()
        st.pyplot(fig)

        # 各銘柄の詳細カード
        for res in results:
            st.markdown(f"### 🎯 {res['銘柄']}")
            cr1, cr2 = st.columns([1, 2])
            with cr1:
                st.metric("シミュレーション予想額", f"{res['将来']:,.0f}円", f"{res['将来']-f_inv:+,.0f}円")
            with cr2:
                st.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
            
            # ニュース表示
            st.write("▼ AIが分析した最新ニュース")
            for n in res['news']:
                st.markdown(f"""
                <div style='background:white; padding:10px; border-radius:8px; margin-bottom:8px; border:1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <span style='color:#f1c40f;'>{'★' * n['score']}</span> 
                    <a href='{n['link']}' target='_blank' style='text-decoration:none; color:#2c3e50; font-weight:bold;'>{n['title']}</a>
                </div>
                """, unsafe_allow_html=True)

# --- 7. 免責事項 ---
st.markdown("""
<div style="font-size: 0.8em; opacity: 0.7; padding: 20px; border-top: 1px solid #eee; margin-top: 50px;">
    <b>⚠️ 免責事項</b><br>
    本アプリは投資の助言を行うものではありません。予測はAIによる統計的な計算に基づいたものであり、将来の成果を保証するものではありません。
    実際の投資判断は自己責任で行ってください。また、本アプリにはアフィリエイトリンクが含まれています。[PR]
</div>
""", unsafe_allow_html=True)
