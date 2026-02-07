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
import re

# --- 0. 基本設定 ---
# 実際の運用時はここにご自身のデプロイしたURLを入力してください
APP_URL = "https://your-app-name.streamlit.app/" 

CHARACTER_URL = "https://github.com/xxxtsukasaxxx51-stack/stock-app/blob/main/Gemini_Generated_Image_j2mypyj2mypyj2my.png?raw=true"
INVESTMENT_QUOTES = [
    "「木を見て森を見ず」にならないように、期間を変えてチェックしよう！",
    "「短期は感情、長期は理屈」で動くのが相場の常だよ。",
    "「どの期間で戦うか」を決めることが、投資の第一歩だね。",
    "「分散投資」は、投資の世界で唯一のフリーランチ（タダ飯）だよ。"
]

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro (Max版)", layout="wide", page_icon="📈")

# --- 2. セッション管理 ---
if "char_msg" not in st.session_state:
    st.session_state.char_msg = random.choice(INVESTMENT_QUOTES)
if "results" not in st.session_state:
    st.session_state.results = []
if "plot_data" not in st.session_state:
    st.session_state.plot_data = {}

# --- 3. CSS (デザイン・広告・免責の最適化) ---
st.markdown(f"""
    <style>
    .welcome-box {{ background-color: #f0f7ff; padding: 20px; border-radius: 15px; border: 1px solid #3182ce; margin-bottom: 25px; }}
    .feature-tag {{ background: #3182ce; color: white; padding: 2px 10px; border-radius: 5px; font-size: 0.8em; margin-right: 5px; }}
    .main-step {{ color: #3182ce; font-weight: bold; font-size: 1.2em; margin-bottom: 15px; border-left: 5px solid #3182ce; padding-left: 10px; }}
    
    /* Xシェアボタンのデザイン */
    .x-share-button {{
        display: inline-block; background-color: #000000; color: white !important; 
        padding: 10px 24px; border-radius: 30px; text-decoration: none; 
        font-weight: bold; font-size: 0.9em; margin-top: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2); transition: 0.3s;
    }}
    .x-share-button:hover {{ background-color: #333333; transform: scale(1.02); opacity: 0.9; }}

    /* 広告カードのデザイン */
    .ad-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin: 40px 0; }}
    .ad-card {{ 
        flex: 1; min-width: 300px; max-width: 500px; padding: 25px; 
        border: 2px solid #e2e8f0; border-radius: 20px; text-align: center; 
        background: linear-gradient(145deg, #ffffff, #f7fafc);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); transition: 0.2s;
    }}
    .ad-card:hover {{ transform: translateY(-5px); border-color: #3182ce; }}
    .ad-badge {{ background: #ff4b4b; color: white; padding: 3px 10px; border-radius: 10px; font-size: 0.7em; font-weight: bold; margin-bottom: 10px; display: inline-block; }}
    .ad-card a {{ 
        display: block; background-color: #3182ce; color: white !important; 
        padding: 12px; border-radius: 10px; text-decoration: none; font-weight: bold; margin-top: 15px;
    }}
    
    /* ニュース・感情分析バッジ */
    .sentiment-badge {{ background: #3182ce; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; margin-bottom: 10px; }}
    .news-box {{ background: white; padding: 10px; border-radius: 8px; border-left: 5px solid #3182ce; margin-bottom: 8px; font-size: 0.9em; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .advice-box {{ padding: 20px; border-radius: 15px; text-align: center; font-weight: bold; border: 1px solid rgba(0,0,0,0.1); }}

    /* 免責事項 */
    .disclaimer-box {{ font-size: 0.85em; color: #4a5568; background: #ffffff; padding: 25px; border-radius: 15px; margin-top: 60px; border: 1px solid #cbd5e0; line-height: 1.6; }}

    /* キャラクター */
    .floating-char-box {{ position: fixed; bottom: 20px; right: 20px; z-index: 999; display: flex; flex-direction: column; align-items: center; pointer-events: none; }}
    .char-img {{ width: 140px; mix-blend-mode: multiply; filter: contrast(125%) brightness(108%); animation: float 3s ease-in-out infinite; }}
    .auto-quote-bubble {{ background: white; border: 2px solid #3182ce; border-radius: 15px; padding: 10px 15px; margin-bottom: 10px; font-size: 0.85em; font-weight: bold; width: 220px; text-align: center; position: relative; }}
    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-12px); }} }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. 銘柄リスト & 補助関数 ---
STOCK_PRESETS = {
    "🇺🇸 エヌビディア (AI半導体)": "NVDA", "🇺🇸 テスラ (電気自動車)": "TSLA", "🇺🇸 アップル (iPhone)": "AAPL",
    "🇺🇸 マイクロソフト (AI/OS)": "MSFT", "🇺🇸 アマゾン (EC)": "AMZN", "🇺🇸 アルファベット (Google)": "GOOGL",
    "🇯🇵 トヨタ自動車 (世界一)": "7203.T", "🇯🇵 ソニーG (エンタメ)": "6758.T", "🇯🇵 ソフトバンクG (投資)": "9984.T",
    "🇯🇵 任天堂 (ゲーム)": "7974.T", "🇯🇵 三菱UFJ銀 (金融)": "8306.T", "🇯🇵 キーエンス (高収益)": "6861.T"
}

def clean_stock_name(name):
    # 国旗やカッコを除去してグラフ・検索・Xシェア用に最適化
    name = re.sub(r'[^\w\s\.]', '', name)
    return name.strip().split(' ')[0]

# --- 5. メイン表示 ---
st.title("🤖 AIマーケット総合診断 Pro (Max)")

st.markdown("""
<div class="welcome-box">
    <h4 style="margin-top:0;">🌟 はじめての方へ：このアプリでできること</h4>
    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        <div><span class="feature-tag">予測</span> <b>1. 未来予測</b>：5日後の株価をAI算出。</div>
        <div><span class="feature-tag">分析</span> <b>2. 星判定</b>：最新ニュースを星5段階で判定。</div>
        <div><span class="feature-tag">共有</span> <b>3. Xでポスト</b>：診断結果をX(Twitter)に投稿可能！</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("💡 分析期間を変えると結果が変わるのはなぜ？"):
    st.write("""
    投資の目的（ゴール）によって、AIが見るべきデータが異なるからです。
    - **「1週間/30日」**: 短期的なトレンドや「勢い」を重視。デイトレ等の参考に。
    - **「5年/全期間」**: その企業が本来持っている「長期的な成長力」を重視。積立投資等の参考に。
    """)

with st.expander("⭐ 「星の指標（AI感情分析）」とは？"):
    st.write("最新ニュースをAIが読み取り、期待値を1.0〜5.0で数値化したものです。5に近いほどポジティブ、1に近いほど要警戒です。")

st.markdown(f"""<div class="floating-char-box"><div class="auto-quote-bubble">{st.session_state.char_msg}</div><img src="{CHARACTER_URL}" class="char-img"></div>""", unsafe_allow_html=True)

# STEP 1 & 2
st.markdown("<div class='main-step'>STEP 1 & 2: 診断したい銘柄と条件を選ぼう</div>", unsafe_allow_html=True)
c_in1, c_in2 = st.columns([2, 1])
selected_names = c_in1.multiselect("リストから選ぶ", list(STOCK_PRESETS.keys()), default=["🇺🇸 エヌビディア (AI半導体)"])
f_inv = c_in2.number_input("シミュレーション金額(円)", min_value=1000, value=100000, step=10000)

time_span = st.select_slider("分析する期間を選択（上の説明もチェック！）", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="全期間(Max)")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# 実行
if st.button("🚀 AI診断スタート"):
    results_temp, plot_data_temp = [], {}
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    with st.spinner('データを解析中...'):
        for full_name in selected_names:
            try:
                symbol = STOCK_PRESETS[full_name]
                df = yf.download(symbol, period=span_map[time_span], progress=False)
                if df.empty: continue
                
                curr = float(df['Close'].iloc[-1])
                y_reg = df['Close'].tail(20).values.reshape(-1, 1)
                X_reg = np.arange(len(y_reg)).reshape(-1, 1)
                model = LinearRegression().fit(X_reg, y_reg)
                pred_val = float(model.predict([[len(y_reg)+5]])[0][0])
                pred_date = (df.index[-1] + timedelta(days=5)).strftime('%m/%d')
                
                display_name = clean_stock_name(full_name)
                
                q = display_name if ".T" in symbol else symbol
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ja&gl=JP"
                feed = feedparser.parse(url)
                news_list, stars_sum = [], 0
                if feed.entries:
                    for e in feed.entries[:3]:
                        s = int(st.session_state.sentiment_analyzer(e.title[:128])[0]['label'].split()[0])
                        stars_sum += s
                        title = GoogleTranslator(source='en', target='ja').translate(e.title) if ".T" not in symbol else e.title
                        news_list.append({"title": title, "score": s, "link": e.link})
                    avg_score = stars_sum / len(news_list)
                else: avg_score = 3.0
                
                adv, col = ("🚀 強気", "#d4edda") if avg_score >= 3.5 and pred_val > curr else ("⚠️ 警戒", "#f8d7da") if avg_score <= 2.2 else ("☕ 様子見", "#e2e3e5")
                
                plot_data_temp[display_name] = df
                results_temp.append({
                    "銘柄": display_name, "将来": f_inv * (pred_val / curr), "adv": adv, "col": col, 
                    "news": news_list, "stars": avg_score, "gain": f_inv * (pred_val / curr) - f_inv, 
                    "pred_val": pred_val, "pred_date": pred_date, "period_label": time_span, "invest": f_inv
                })
            except: continue

    st.session_state.results = results_temp
    st.session_state.plot_data = plot_data_temp
    st.rerun()

# --- 7. 結果表示 ---
if st.session_state.results:
    st.markdown(f"<div class='main-step'>STEP 3: {st.session_state.results[0].get('period_label')}の診断結果</div>", unsafe_allow_html=True)
    
    # グラフ表示
    fig, ax = plt.subplots(figsize=(10, 4))
    japanize_matplotlib.japanize()
    for res in st.session_state.results:
        name = res['銘柄']
        if name in st.session_state.plot_data:
            df = st.session_state.plot_data[name]
            base = df['Close'].iloc[0]
            line = ax.plot(df.index, df['Close']/base*100, label=f"{name}")
            ax.scatter(df.index[-1] + timedelta(days=5), (res['pred_val']/base)*100, marker='*', s=200, color=line[0].get_color(), edgecolors='black', zorder=5)
    ax.set_ylabel("成長率 (%)")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    # 銘柄別詳細カード
    for res in st.session_state.results:
        st.markdown(f"### 🎯 {res['銘柄']}")
        c_res1, c_res2 = st.columns([1, 2])
        c_res1.metric(f"{res['pred_date']} 予想資産額", f"{res['将来']:,.0f}円", f"{res['gain']:+,.0f}円")
        c_res2.markdown(f"<div class='advice-box' style='background-color: {res['col']};'>{res['adv']}</div>", unsafe_allow_html=True)
        
        # X（Twitter）シェアテキストの作成
        share_text = (
            f"📈 【AIマーケット診断】\n"
            f"━━━━━━━━━━━━━━\n"
            f"🎯 銘柄：{res['銘柄']}\n"
            f"🔍 分析期間：{res['period_label']}\n"
            f"💰 投資額：{res['invest']:,.0f}円\n"
            f"📢 AI判定：{res['adv']}\n"
            f"🚀 5日後の予想：{res['将来']:,.0f}円\n"
            f"━━━━━━━━━━━━━━\n"
            f"AIが最新ニュースと相場を解析しました！\n"
            f"詳細をアプリでチェック 👇\n"
            f"{APP_URL}"
        )
        x_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
        st.markdown(f'<a href="{x_url}" target="_blank" class="x-share-button">𝕏 この結果をポストして保存</a>', unsafe_allow_html=True)

        st.markdown(f"<div class='sentiment-badge'>AI感情分析: {res['stars']:.1f} / 5.0 {'⭐' * int(res['stars'])}</div>", unsafe_allow_html=True)
        for n in res['news']:
            st.markdown(f"<div class='news-box'>{'★' * n['score']} <a href='{n['link']}' target='_blank'><b>{n['title']}</b></a></div>", unsafe_allow_html=True)

# --- 8. 広告セクション ---
st.markdown("""
<div class="ad-container">
    <div class="ad-card">
        <span class="ad-badge">初心者におすすめ</span>
        <p style="font-weight:bold; margin-bottom:5px;">スマホで始める最短の株式投資</p>
        <p style="font-size:0.85em; color:#718096;">AI診断で気になった銘柄、すぐチェックしませんか？1株から買える手軽さが人気です。</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">DMM 株で口座開設(無料) [PR]</a>
    </div>
    <div class="ad-card">
        <span class="ad-badge">資産運用の強い味方</span>
        <p style="font-weight:bold; margin-bottom:5px;">高機能チャートアプリ TOSSY</p>
        <p style="font-size:0.85em; color:#718096;">プロ級の分析をスマホで。AI予測と組み合わせて、より精度の高い投資判断をサポート。</p>
        <a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">今すぐアプリを体験する [PR]</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 9. 丁寧な免責事項 ---
st.markdown("""
<div class="disclaimer-box">
    <div style="font-weight:bold; color:#2d3748; margin-bottom:10px;">⚠️ ご利用にあたっての重要なご案内</div>
    <p>
        本アプリで提供される株価予測および「星の指標（感情分析）」は、過去の市場データと最新のニュース記事を独自のアルゴリズムおよびAI技術を用いて解析したものであり、<b>将来の運用成果を保証するものではありません。</b>
    </p>
    <ul style="padding-left: 20px;">
        <li>株価は経済情勢、政治、企業業績などにより変動し、投資元本を割り込むリスクがあります。</li>
        <li>AIによる予測はあくまで一つの判断材料であり、その正確性を保証するものではありません。</li>
        <li>本アプリの利用によって生じたいかなる損害についても、開発者は一切の責任を負いかねます。</li>
        <li>実際の取引にあたっては、各金融機関の最新情報をご確認の上、ご自身の責任で判断してください。</li>
    </ul>
    <p style="margin-top:10px; font-size:0.9em; border-top:1px solid #eee; pt:10px;">
        ※本サービスにはプロモーションが含まれています。これによる収益はAIモデルの維持および品質向上のために活用されます。
    </p>
</div>
""", unsafe_allow_html=True)
