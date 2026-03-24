import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.set_page_config(page_title="QCPS Explainable AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open("results/simulation_data.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

data = load_data()
if data is None:
    st.error("Data not found. Please run `run_evaluation.py` first.")
    st.stop()

res_rl = data['rl']
res_static = data['static']
tickers = data.get('tickers', [f"ASSET_{i}" for i in range(10)])

steps_rl = res_rl.get('historical_trajectory', [])
steps_static = res_static.get('historical_trajectory', [])

if not steps_rl:
    st.error("Simulation trajectory is empty.")
    st.stop()

total_steps = len(steps_rl)

# --- STATE MANAGEMENT ---
if 'step_idx' not in st.session_state:
    st.session_state.step_idx = total_steps - 1
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

# --- SIDEBAR & PLAYBACK ---
st.sidebar.title("⏱️ Timeline Navigation")
st.sidebar.markdown("Press play to watch the AI adapt over time.")

col1, col2, col3 = st.sidebar.columns(3)
if col1.button("⏮ Prev"):
    st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
    
if col2.button("⏸ Pause" if st.session_state.is_playing else "▶️ Play"):
    st.session_state.is_playing = not st.session_state.is_playing

if col3.button("Next ⏭"):
    st.session_state.step_idx = min(total_steps - 1, st.session_state.step_idx + 1)

speed = st.sidebar.slider("Playback Speed (seconds)", 0.5, 3.0, 1.0)

selected_step = st.sidebar.slider("Select Month", 0, total_steps - 1, st.session_state.step_idx, key='slider_ui')
if selected_step != st.session_state.step_idx:
    st.session_state.step_idx = selected_step

# Auto-play loop hook
if st.session_state.is_playing:
    if st.session_state.step_idx < total_steps - 1:
        time.sleep(speed)
        st.session_state.step_idx += 1
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.rerun()

current_rl = steps_rl[st.session_state.step_idx]
current_static = steps_static[st.session_state.step_idx]

# --- HELPER STORY FUNCTIONS ---
def get_ai_narrator_message(s_rl, s_static):
    metrics = s_rl.get('market_metrics', {})
    vol = metrics.get('volatility', 0)
    dd = metrics.get('drawdown', 0)
    
    # 1. Input Context
    if vol > 0.18 or dd < -0.10:
        market_obs = "I noticed high turbulence in the stock market right now. Volatility is spiking."
        is_crash = True
    else:
        market_obs = "The market looks relatively calm and stable at the moment."
        is_crash = False
        
    # 2. AI Decision
    q = s_rl['policy_action']['risk_aversion_q']
    if is_crash and q > 1.5:
        decision = f"Because of the elevated risk, I decided to play it safe. I increased my **cautiousness** to protect our capital and told the quantum solver to seek safe-haven assets."
    elif not is_crash and q <= 1.5:
        decision = "Since things are stable, I kept my constraints loose. I commanded the quantum solver to aggressively hunt for growth assets."
    else:
        decision = "I adjusted the portfolio boundaries moderately to navigate the shifting math topography."
        
    # 3. Outcome
    diff = s_rl['realized_return'] - s_static['realized_return']
    if diff > 0.002:
        outcome = "This was a **great decision**! I successfully avoided losses that a rigid, fixed strategy would have suffered."
        verdict = "GOOD 🟢"
    elif diff < -0.002:
        outcome = "This was a **poor decision** short-term. My defensive stance caused us to slightly lag behind the fixed baseline strategy."
        verdict = "BAD 🔴"
    else:
        outcome = "The outcome was **neutral**. Both my adaptive strategy and the fixed baseline performed similarly."
        verdict = "NEUTRAL ⚪"
        
    # Build Top Banner
    if is_crash and diff > 0.002:
        banner = "📉 Market Crash ➔ 🛡️ AI Increased Safety ➔ 💰 Capital Protected"
    elif not is_crash and diff > 0.002:
        banner = "📈 Stable Market ➔ 🚀 AI Captured Growth ➔ 🏆 Outperformed Baseline"
    elif is_crash and diff <= 0.002:
        banner = "📉 Market Crash ➔ 🛡️ AI Played Safe ➔ ⚖️ Kept Pace with Baseline"
    else:
        banner = "⚙️ Market Shift ➔ 🧠 AI Adjusted Weights ➔ 📊 Outcome Logged"
        
    return banner, f"{market_obs}\n\n{decision}\n\n{outcome}", verdict, diff

# --- HEADER / MAIN UI ---
st.title("🧠 Quantum Cognitive AI System")
st.info(f"📅 **Date Range Evaluation: {current_rl['step_start'].strftime('%B %Y')} to {current_rl['step_end'].strftime('%B %Y')}**")

# 1. SUMMARY BANNER
banner, narrator_text, verdict, diff = get_ai_narrator_message(current_rl, current_static)
st.success(f"**TL;DR Insight:** {banner}")

# 2. AI NARRATOR MODE
with st.chat_message("ai"):
    st.markdown(f"### My Thought Process (Step {st.session_state.step_idx})")
    st.write(narrator_text)

# 3. GOOD/BAD INDICATOR & BEFORE/AFTER COMPARISON
c1, c2, c3 = st.columns(3)
c1.metric("AI Decision Verdict", verdict)

static_return = current_static['realized_return'] * 100
ai_return = current_rl['realized_return'] * 100
c2.metric("Adaptive AI Return", f"{ai_return:.2f}%", delta=f"{(ai_return - static_return):.2f}% vs Baseline")
c3.metric("Fixed Baseline Return", f"{static_return:.2f}%")

st.divider()

# 4. DECISION BREAKDOWN PIPELINE
st.markdown("### 🔍 Step-by-Step Breakdown")
cols = st.columns(3)

vol = current_rl.get('market_metrics', {}).get('volatility', 0)
risk_label = 'High 🔴' if vol > 0.18 else ('Medium 🟡' if vol > 0.12 else 'Low 🟢')
cols[0].info(f"**1️⃣ Input (Market sensors)**\n\nRisk Level Detected: **{risk_label}**")

q = current_rl['policy_action']['risk_aversion_q']
cautious = 'Highly Defensive 🛡️' if q > 1.5 else 'Aggressive Growth 🎯'
cols[1].warning(f"**2️⃣ Decision (AI Policy)**\n\nStrategic Stance: **{cautious}**")

stab = current_rl.get('stability_score', 0)
conf = "High 🟢" if stab > 0.7 else ("Medium 🟡" if stab > 0.4 else "Low 🔴")
cols[2].error(f"**3️⃣ Outcome (Quantum Solver)**\n\nMathematical Confidence: **{conf}**")

st.divider()

# --- EXPLAINABLE TABS ---
st.markdown("### Dig Deeper (Visual Views)")
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 AI Learning Chart", "⚖️ Uncertainty Meter", "⚛️ Inside the Quantum Brain", "📈 Money & Friction"
])

with tab1:
    st.header("How the AI Learns")
    st.write("Over time, the AI learns to tune its internal dials (like 'Risk Aversion') to maximize financial rewards while minimizing trading costs.")
    
    dates = [s['step_end'] for s in steps_rl]
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    q_vals = [s['policy_action']['risk_aversion_q'] for s in steps_rl]
    c_vals = [s['policy_action']['constraint_scaling'] for s in steps_rl]
    
    ax[0].plot(dates, q_vals, linewidth=2, label="Risk Aversion (Cautiousness)")
    ax[0].plot(dates, c_vals, label="Constraint Scaling (Flexibility)")
    ax[0].axvline(x=current_rl['step_end'], color='grey', linestyle='--', alpha=0.7)
    ax[0].set_title("AI Dial Adjustments Over Time")
    ax[0].legend()
    
    c_rl = np.cumprod([1 + s['realized_return'] for s in steps_rl])
    c_static = np.cumprod([1 + s['realized_return'] for s in steps_static])
    ax[1].plot(dates, c_static, color='red', linestyle='--', label="Fixed Baseline Strategy")
    ax[1].plot(dates, c_rl, color='green', linewidth=2.5, label="Adaptive AI Strategy")
    ax[1].axvline(x=current_rl['step_end'], color='grey', linestyle='--', alpha=0.7)
    ax[1].set_title("Cumulative Wealth Protection")
    ax[1].legend()
    
    st.pyplot(fig)

with tab2:
    st.header("⚖️ How Confident was the Quantum Solver?")
    st.write("Sometimes the quantum solver is incredibly sure about what to buy. Other times, the market math is chaotic, and the solver struggles to find consensus.")
    
    stab = current_rl.get('stability_score', 0)
    conf_label = "High Confidence 🟢" if stab > 0.7 else ("Medium Confidence 🟡" if stab > 0.4 else "Low Confidence 🔴")
    
    st.subheader(f"Current Status: **{conf_label}**")
    st.progress(min(1.0, stab))
    
    st.markdown("*(The chart below shows where the solver was confused. Tall red bars mean high disagreement!)*")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ensemble = current_rl.get('ensemble_results', [])
    if ensemble:
        df_weights = pd.DataFrame()
        for i, run in enumerate(ensemble):
            w = run.get('weights', np.zeros(len(tickers)))
            df_weights[f"R{i}"] = w
        
        means = df_weights.mean(axis=1)
        variances = df_weights.var(axis=1) * 500  # scaled
        
        x = np.arange(len(tickers))
        ax.bar(x - 0.2, means, 0.4, label='Final AI Allocation Decision', color='#2ca02c')
        ax.bar(x + 0.2, variances, 0.4, label='Quantum Disagreement (Uncertainty)', color='#d62728')
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45)
        ax.legend()
        st.pyplot(fig)

with tab3:
    st.header("Inside the Quantum Optimizer")
    st.write("The AI translates its financial commands into a **QUBO Matrix** (a topography map representing the investment problem).")
    st.write("**QAOA** (the Quantum algorithm) then searches this map for the deepest valleys, representing the optimal portfolio.")
    
    Q = np.array(current_rl.get('qubo_matrix', []))
    if Q.size > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(Q, cmap='vlag', center=0, ax=ax, xticklabels=False, yticklabels=False)
        ax.set_title("QUBO Matrix Heatmap (Blue = Favorable Asset Pairs)")
        st.pyplot(fig)
    else:
        st.warning("No topographical matrix recorded for this execution gap.")

with tab4:
    st.header("Financial Trade Friction")
    
    dates = [s['step_end'] for s in steps_rl]
    t_rl = [s['turnover'] for s in steps_rl]
    t_static = [s['turnover'] for s in steps_static]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, t_static, color='red', linestyle='--', alpha=0.5, label='Fixed Strategy Trading Friction')
    ax.plot(dates, t_rl, color='blue', linewidth=2, label='AI Adaptive Trading Friction')
    ax.axvline(x=current_rl['step_end'], color='grey', linestyle='--')
    ax.set_title("Portfolio Turnover (Lower is better to avoid trading fees)")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("### Lifetime Evaluation Summary")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.success("**Adaptive AI System**")
        st.write(f"Total Wealth Growth: `{res_rl['summary']['total_return']:.2%}`")
        st.write(f"Smoothed Sharpe Score: `{res_rl['summary']['realized_sharpe']:.2f}`")
    with cc2:
        st.error("**Fixed Baseline System**")
        st.write(f"Total Wealth Growth: `{res_static['summary']['total_return']:.2%}`")
        st.write(f"Smoothed Sharpe Score: `{res_static['summary']['realized_sharpe']:.2f}`")
