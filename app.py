import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import urllib.parse
import re

# --- 0. 基本設定 (エラーの原因だった定義を追加) ---
APP_URL = "https://your-app-name.streamlit.app/" 

# --- 1. ページ設定 ---
st.set_page_config(page_title="AIマーケット診断 Pro", layout="wide", page_icon="📈")

# --- 2. CSS (ダークモード・スマホ対応・Xボタン) ---
st.markdown("""
    <style>
    .main-step { color: #3182ce; font-weight: bold; font-size: 1.1rem; border-left: 5px solid #3182ce; padding-left: 10px; margin: 20px 0 10px 0; }
    .advice-box { padding: 15px; border-radius: 12px; text-align: center; font-weight: bold; color: #1a202c; margin-bottom: 15px; }
    .x-share-button { display: inline-block; background: #000; color: #fff !important; padding: 12px 24px; border-radius: 30px; text-decoration: none; font-weight: bold; margin: 10px 0; }
    .disclaimer-box { font-size: 0.8rem; padding: 20px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2); margin-top: 40px; line-height: 1.6; color: gray; }
    .ad-card { flex: 1; min-width: 280px; padding: 20px; border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 15px; background: rgba(128, 128, 128, 0.05); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AIマーケット総合診断 Pro")

# --- 3. 解説セクション ---
with st.expander("💡 感情指数と期間設定について"):
    st.markdown("""
    * **感情指数**: 市場の勢いをAIが解析。⭐4以上は上昇トレンド、⭐2以下は警戒。
    * **分析期間**: 短期（1週間〜30日）は値動きの速さ、長期（1年〜）は安定性を重視します。
    """)

st.markdown("<div class='main-step'>STEP 1 & 2: 銘柄選びと条件設定</div>", unsafe_allow_html=True)

# --- 4. 銘柄入力 (人気選択 + フリー入力) ---
popular_stocks = {
    "🇺🇸 エヌビディア": "NVDA", "🇺🇸 テスラ": "TSLA", "🇺🇸 アップル": "AAPL",
    "🇯🇵 トヨタ": "7203.T", "🇯🇵 三菱UFJ": "8306.T", "🇯🇵 ソフトバンクG": "9984.T"
}

c_sel, c_free = st.columns([1, 1])
selected_popular = c_sel.multiselect("🔥 人気の銘柄から選ぶ", list(popular_stocks.keys()))
free_input = c_free.text_input("✍️ 自由に入力 (例: MSFT, 6758.T)", placeholder="カンマ区切りで入力")

# 銘柄リストを結合
final_symbols = [popular_stocks[name] for name in selected_popular]
if free_input:
    final_symbols.extend([s.strip().upper() for s in free_input.split(",") if s.strip()])
final_symbols = list(dict.fromkeys(final_symbols)) # 重複削除

c_in1, c_in2 = st.columns([1, 1])
f_inv = c_in1.number_input("投資金額(円)", min_value=1000, value=100000)
time_span = st.select_slider("分析期間", options=["1週間", "30日", "1年", "5年", "全期間(Max)"], value="1年")
span_map = {"1週間":"7d","30日":"1mo","1年":"1y","5年":"5y","全期間(Max)":"max"}

# --- 5. 実行ロジック ---
if st.button("🚀 AI診断スタート"):
    if not final_symbols:
        st.error("銘柄を選択するか、入力してください。")
    else:
        results = []
        plot_data = {}
        
        with st.spinner('データを解析中...'):
            for symbol in final_symbols:
                try:
                    df = yf.download(symbol, period=span_map[time_span], progress=False)
                    if df.empty:
                        st.warning(f"{symbol} のデータが取得できませんでした。")
                        continue
                    
                    # 予測ロジック (エラー対策済み)
                    y = df['Close'].values.flatten()
                    if len(y) < 2: continue
                    
                    y_last = y[-20:] if len(y) >= 20 else y
                    x_last = np.arange(len(y_last)).reshape(-1, 1)
                    
                    model = LinearRegression().fit(x_last, y_last)
                    pred_price = float(model.predict(np.array([[len(y_last)+5]]))[0])
                    curr_price = float(y[-1])
                    
                    # 感情指数シミュレーション
                    stars = round(np.clip(3.0 + (pred_price/curr_price - 1)*10, 1.5, 5.0), 1)
                    adv, col = ("🚀 強気", "#d4edda") if pred_price > curr_price else ("⚠️ 警戒", "#f8d7da")
                    
                    results.append({
                        "symbol": symbol,
                        "future": f_inv * (pred_price / curr_price),
                        "gain": (f_inv * (pred_price / curr_price)) - f_inv,
                        "adv": adv, "col": col, "stars": stars, "period": time_span
                    })
                    plot_data[symbol] = df
                except Exception as e:
                    st.error(f"{symbol} の解析中にエラーが発生しました。")

        # --- 6. 結果表示 ---
        if results:
            st.markdown("<div class='main-step'>STEP 3: 診断結果</div>", unsafe_allow_html=True)
            
            # グラフ描画
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            for s, d in plot_data.items():
                ax.plot(d.index, d['Close'] / d['Close'].iloc[0] * 100, label=s)
            ax.set_ylabel("Growth Rate (%)")
            ax.legend()
            st.pyplot(fig)

            for res in results:
                st.markdown(f"### 🎯 {res['symbol']} ({res['period']}分析)")
                r1, r2 = st.columns(2)
                r1.metric("5日後の予想資産", f"{res['future']:,.0f}円", f"{res['gain']:+,.0f}円")
                r2.markdown(f"<div class='advice-box' style='background-color:{res['col']};'>{res['adv']} (期待値: ⭐{res['stars']})</div>", unsafe_allow_html=True)
                
                # X投稿 (ここがエラーの箇所でした)
                share_text = f"📈 AIマーケット診断\n🎯 {res['symbol']} ({res['period']})\n📢 判定: {res['adv']}\n🚀 予想: {res['future']:,.0f}円\n{APP_URL}"
                x_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
                st.markdown(f'<a href="{x_url}" target="_blank" class="x-share-button">𝕏 結果をポストする</a>', unsafe_allow_html=True)
                st.divider()

# --- 7. 免責事項 & 広告 ---
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ 免責事項</b><br>
    本アプリの診断結果は過去のデータに基づいたシミュレーションであり、将来の投資成果を保証するものではありません。売買の最終判断はご自身の責任で行ってください。
</div>
<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 20px;">
    <div class="ad-card"><b>DMM 株 [PR]</b><br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+7YDIR6+1WP2+15RRSY" target="_blank">口座開設はこちら</a></div>
    <div class="ad-card"><b>TOSSY [PR]</b><br><a href="https://px.a8.net/svt/ejp?a8mat=4AX5KE+8LLFCI+1WP2+1HM30Y" target="_blank">アプリを体験</a></div>
</div>
""", unsafe_allow_html=True))
