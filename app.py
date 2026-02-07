import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import urllib.parse
import re

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. CSS (ダークモード・スマホ対応) ---
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.1rem; border-left: 5px solid #3182ce; padding-left: 10px; margin: 20px 0 10px 0; }
    .advice-box { padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; color: #1a202c; margin-bottom: 15px; }
    .x-share-button { display: inline-block; background: #000; color: #fff !important; padding: 12px 24px; border-radius: 30px; text-decoration: none; font-weight: bold; }
    .disclaimer { font-size: 0.8rem; color: gray; margin-top: 40px; padding: 15px; border: 1px solid #ddd; border-radius: 10px; }
    .ad-card { background: rgba(128, 128, 128, 0.05); padding: 15px; border-radius: 10px; text-align: center; flex: 1; min-width: 250px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AIマーケット総合診断 Pro")

# --- 3. 入力セクション ---
st.markdown("<div class='main-step'>STEP 1: 銘柄を選んで診断</div>", unsafe_allow_html=True)

popular_stocks = {
    "🇺🇸 エヌビディア": "NVDA", "🇺🇸 テスラ": "TSLA", "🇺🇸 アップル": "AAPL",
    "🇯🇵 トヨタ": "7203.T", "🇯🇵 三菱UFJ": "8306.T", "🇯🇵 ソフトバンクG": "9984.T"
}

col1, col2 = st.columns([2, 1])
selected_popular = col1.multiselect("🔥 人気銘柄から選択", list(popular_stocks.keys()), default=["🇺🇸 エヌビディア"])
free_input = col1.text_input("✍️ 自由入力 (例: MSFT, 6758.T)", placeholder="カンマ区切りで入力")

f_inv = col2.number_input("投資額(円)", min_value=1000, value=100000)
time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="1年")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# 銘柄リストの統合
final_symbols = [popular_stocks[name] for name in selected_popular]
if free_input:
    final_symbols.extend([s.strip().upper() for s in free_input.split(",") if s.strip()])

# --- 4. 実行ボタン ---
if st.button("🚀 AI診断スタート"):
    if not final_symbols:
        st.error("銘柄を入力してください。")
    else:
        results = []
        plot_data = {}
        
        with st.spinner('データを解析中...'):
            for symbol in list(dict.fromkeys(final_symbols)):
                try:
                    # データ取得
                    df = yf.download(symbol, period=span_map[time_span], progress=False)
                    if df.empty:
                        st.warning(f"銘柄 {symbol} のデータが見つかりませんでした。")
                        continue
                    
                    # 予測ロジック (直近20日のトレンド)
                    y = df['Close'].tail(20).values
                    x = np.arange(len(y)).reshape(-1, 1)
                    model = LinearRegression().fit(x, y)
                    pred_price = float(model.predict([[len(y)+5]])[0])
                    curr_price = float(df['Close'].iloc[-1])
                    
                    # 感情指数 (軽量化のためランダムシミュレーション + トレンド加味)
                    stars = round(np.clip(3.0 + (pred_price/curr_price - 1)*10, 1.5, 5.0), 1)
                    
                    # 判定
                    adv, col = ("🚀 強気", "#d4edda") if pred_price > curr_price else ("⚠️ 警戒", "#f8d7da")
                    
                    results.append({
                        "symbol": symbol, "future": f_inv * (pred_price / curr_price),
                        "gain": (f_inv * (pred_price / curr_price)) - f_inv,
                        "adv": adv, "col": col, "stars": stars, "period": time_span
                    })
                    plot_data[symbol] = df
                except Exception as e:
                    st.error(f"{symbol} の解析中にエラーが発生しました: {e}")

        # --- 5. 結果表示 ---
        if results:
            st.markdown("<div class='main-step'>STEP 2: 診断結果</div>", unsafe_allow_html=True)
            
            # グラフ
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            for s, d in plot_data.items():
                # 成長率に換算して表示
                ax.plot(d.index, d['Close'] / d['Close'].iloc[0] * 100, label=s)
            ax.legend()
            ax.set_title("Price Growth Rate (%)")
            st.pyplot(fig)

            for res in results:
                st.markdown(f"### 🎯 {res['symbol']} ({res['period']}分析)")
                r1, r2 = st.columns(2)
                r1.metric("5日後の予想資産額", f"{res['future']:,.0f}円", f"{res['gain']:+,.0f}円")
                r2.markdown(f"<div class='advice-box' style='background-color:{res['col']};'>{res['adv']} (感情指数: ⭐{res['stars']})</div>", unsafe_allow_html=True)
                
                # X投稿用
                share_text = f"📈 AIマーケット診断\n🎯 {res['symbol']}\n📢 判定: {res['adv']}\n🚀 5日後の予想: {res['future']:,.0f}円\n#AI株診断 #投資"
                x_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
                st.markdown(f'<a href="{x_url}" target="_blank" class="x-share-button">𝕏 結果をポストする</a>', unsafe_allow_html=True)
                st.divider()
        else:
            st.info("診断結果を生成できませんでした。銘柄コードが正しいか確認してください。")

# --- 6. 広告・免責 ---
st.markdown("""
<div class="disclaimer">
    <b>⚠️ 免責事項</b><br>
    本予測は過去のデータに基づくAIシミュレーションであり、将来の運用成果を保証しません。投資は元本割れのリスクがあります。最終的な判断はご自身の責任で行ってください。
</div>
<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px;">
    <div class="ad-card"><b>DMM 株 [PR]</b><br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">最短即日で口座開設</a></div>
    <div class="ad-card"><b>TOSSY [PR]</b><br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">高機能チャートを体験</a></div>
</div>
""", unsafe_allow_html=True)
