import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
import numpy as np

st.set_page_config(page_title="Chat Wrapped", layout="centered")

# --- Constants & Regex ---
EMOJI_REGEX = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
LINK_REGEX = re.compile(r'https?://|www\.')

# --- Style Helpers ---
def clean_plot(ax, fig, title, xlabel="", ylabel=""):
    """Keeps the original look but cleans up the spines as requested."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

def slide(title, subtitle, gradient=("purple", "pink")):

    st.markdown(
        f"""
            <div style="
                height: 50vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                color: white;
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background: linear-gradient(135deg, {gradient[0]}, {gradient[1]});
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            ">
                <h1 style="font-size: 42px; margin-bottom: 15px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{title}</h1>
                <p style="font-size: 22px; opacity: .9; font-weight: 300;">{subtitle}</p>
            </div>
        """,
        unsafe_allow_html=True
    )
    st.write("") # Essential gap

# --- Parser ---
def load_whatsapp_chat(file):
    lines = file.read().decode("utf-8").split("\n")
    data = []

    for line in lines:

        if " - " in line:
            timestamp_str, rest = line.split(" - ", 1)

            if ": " in rest:
                sender, message = rest.split(": ", 1)
            else:
                sender = None
                message = rest
            try:
                timestamp_str = timestamp_str.replace('\u202f', ' ').replace('\u00a0', ' ')
                timestamp = datetime.strptime(timestamp_str.strip(), "%d/%m/%Y, %I:%M %p")
            except:
                timestamp = None

            data.append({"timestamp": timestamp, "sender": sender, "message": message})

    return pd.DataFrame(data)

# --- PERFORMANCE ENGINE ---
@st.cache_data(show_spinner=False)
def compute_master_metrics(df):
    m = {}
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.day_name()
    total_counts = df['sender'].value_counts()
    
    # Volume
    df['date_only'] = df['timestamp'].dt.date
    m['total_messages'] = len(df)
    m['most_active_day'] = df.groupby('date_only').size().idxmax()
    m['most_active_day_count'] = df.groupby('date_only').size().max()
    
    # Monthly
    df['month_dt'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
    m['monthly_counts'] = df.groupby('month_dt').size()
    
    # Emoji Intelligence
    all_emojis = []
    sender_emoji_counts = Counter()
    sender_emoji_map = defaultdict(Counter)
    msg_counts = Counter()

    for msg, sender in zip(df["message"], df["sender"]):

        if not msg or not sender: continue
        msg_counts[sender] += 1
        chars = [c for c in msg if EMOJI_REGEX.match(c)]
        all_emojis.extend(chars)
        sender_emoji_counts[sender] += len(chars)

        for c in chars: sender_emoji_map[sender][c] += 1

    m['top_emojis'] = Counter(all_emojis)
    m['sender_emoji_counts'] = sender_emoji_counts
    m['sender_emoji_map'] = sender_emoji_map
    m['avg_emoji'] = {s: sender_emoji_counts[s]/msg_counts[s] for s in sender_emoji_counts if msg_counts[s] > 0}
    
    # Media & Links
    m['media_counts'] = df[df['message'].str.contains("<Media omitted", na=False)]['sender'].value_counts()
    m['link_counts'] = df[df['message'].str.contains(LINK_REGEX, na=False)]['sender'].value_counts()
    
    # Time Analysis
    m['morning_ratio'] = (df[(df['hour'] >= 4) & (df['hour'] < 12)]['sender'].value_counts() / total_counts).fillna(0)
    m['night_ratio'] = (df[(df['hour'] >= 21) | (df['hour'] < 4)]['sender'].value_counts() / total_counts).fillna(0)
    m['weekend_ratio'] = (df[df['weekday'].isin(["Saturday", "Sunday"])]['sender'].value_counts() / total_counts).fillna(0)
    m['weekday_ratio'] = (df[df['weekday'].isin(["Monday","Tuesday","Wednesday","Thursday","Friday"])]['sender'].value_counts() / total_counts).fillna(0)

    # Complexity
    df["msg_len"] = df["message"].apply(lambda x: len(x) if x else 0)
    m['sender_avg_len'] = df.groupby("sender")["msg_len"].mean()

    # Evidence Buster Logic
    deleted_df = df[df['message'].str.contains("This message was deleted|You deleted this message", na=False)]
    m['deleted_counts'] = deleted_df['sender'].value_counts()
    m['deleted_ratio'] = (m['deleted_counts'] / total_counts).fillna(0).sort_values(ascending=False)

    # Rapid Fire
    sender_gaps = {}
    df_sorted = df.sort_values('timestamp')

    for sender, group in df_sorted.groupby('sender'):
        gaps = group['timestamp'].diff().dt.total_seconds().dropna()
        active = gaps[gaps <= 300]

        if not active.empty: sender_gaps[sender] = active.mean()

    m['sender_gaps'] = sender_gaps

    # --- Connections (Tags) ---
    pair_counts = Counter()
    total_tags_sent = Counter()
    total_tags_received = Counter()
    START, END = "\u2068", "\u2069"
    real_members = set(df["sender"].unique()) - {"Meta AI"}
    
    for sender, msg in zip(df["sender"], df["message"].fillna("")):
        idx = 0

        while True:
            s = msg.find(START, idx)
            if s == -1: break
            e = msg.find(END, s+1)
            if e == -1: break
            tag = msg[s+1:e].strip()
            pair_counts[(sender, tag)] += 1
            total_tags_sent[sender] += 1
            total_tags_received[tag] += 1
            idx = e + 1
            
    m['pair_counts'] = pair_counts
    m['total_tags_sent'] = total_tags_sent
    m['total_tags_received'] = total_tags_received
    m['real_members'] = real_members

    # Social Rhythms
    df_sorted['gap_h'] = df_sorted['timestamp'].diff().dt.total_seconds() / 3600
    m['starters'] = df_sorted[(df_sorted['gap_h'] >= 3) | (df_sorted['gap_h'].isna())]['sender'].value_counts()
    df_sorted['next_gap'] = df_sorted['gap_h'].shift(-1)
    m['closers'] = df_sorted[df_sorted['next_gap'] >= 3]['sender'].value_counts()

    # --- The Big Summary ---
    m['hall_of_fame'] = {
        'boss': total_counts.idxmax(),
        'sniper': total_tags_sent.most_common(1)[0][0] if total_tags_sent else "None",
        'magnet': total_tags_received.most_common(1)[0][0] if total_tags_received else "None",
        'starter': m['starters'].idxmax() if not m['starters'].empty else "None",
        'closer': m['closers'].idxmax() if not m['closers'].empty else "None",
        'spam_lord': min(sender_gaps, key=sender_gaps.get) if sender_gaps else "None",
        'night_owl': m['night_ratio'].idxmax() if not m['night_ratio'].empty else "None",
        'media_king': m['media_counts'].idxmax() if not m['media_counts'].empty else "None",
        'philosopher': m['sender_avg_len'].idxmax() if not m['sender_avg_len'].empty else "None"
    }
    return m

# --- SLIDE DECK ---

def first_slide():
    slide("🎉 2025: A Year in Chat", "Your digital memories, decoded and delivered.")

    if "chat_df_2025" not in st.session_state:
        uploaded = st.file_uploader("Drop your WhatsApp export here (.txt)", type=["txt"])

        if uploaded:
            st.snow()
            st.toast("Chat uploaded, Loading...", icon="❤️‍🔥")
            raw_df = load_whatsapp_chat(uploaded)
            df = raw_df[raw_df['timestamp'].notnull()]
            df = df[(df['timestamp'] >= datetime(2025, 1, 1)) & (df['timestamp'] < datetime(2026, 1, 1))]
            df = df[df['sender'] != "Meta AI"]
            st.session_state.chat_df_2025 = df
            st.session_state.metrics = compute_master_metrics(df)
            st.session_state.slide = 1
            st.rerun()

def total_messages_slide():
    m = st.session_state.metrics

    slide(f"🔥 {m['total_messages']} Messages Later...", 
          f"We peaked on {m['most_active_day'].strftime('%B %d')} with a massive {m['most_active_day_count']} texts!")

def monthly_messages_slide():
    m = st.session_state.metrics
    counts = m['monthly_counts']
    slide("🗓️ The Calendar of Chaos", "From January's energy to December's vibe.")
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('#ca70ad')
    ax.bar(counts.index.strftime('%b'), counts.values, color='#8a1187')
    clean_plot(ax, fig, "Monthly Message Breakdown", "Month", "Messages")
    st.pyplot(fig)

def emoji_stats_slide():
    m = st.session_state.metrics
    top5 = m['top_emojis'].most_common(5)
    slide("🎭 The Emoji Awards", "These five carried the entire chat on their backs.")
    st.markdown("### ✨ The Group's Holy Grail")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {e} — Used {c} times' for i,(e,c) in enumerate(top5)])}</pre>", unsafe_allow_html=True)
    counts = pd.Series(m['sender_emoji_counts']).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(counts)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(counts.index, counts.values, color='#8a1187')
    clean_plot(ax, fig, "Emoji Addicts Leaderboard", "Count", "Member")
    st.pyplot(fig)

def favorite_emoji_per_user_slide():
    m = st.session_state.metrics
    favorites = []

    for s, counter in m['sender_emoji_map'].items():

        if counter:
            fav, count = counter.most_common(1)[0]
            favorites.append((s, fav, count))

    favorites.sort(key=lambda x: x[2], reverse=True)
    slide("💖 The Signature Moves", "Every member had that one emoji they just couldn't drop.")
    rows = [f"{i+1}. {n} — {e} ({c} times)" for i,(n,e,c) in enumerate(favorites)]
    st.markdown(f"<pre>{'<br>'.join(rows)}</pre>", unsafe_allow_html=True)

def media_link_senders_slide():
    m = st.session_state.metrics
    slide("📸 The Content Curators", "Links, memes, and photos—the lifeblood of our chat.")
    col1, col2 = st.columns(2)

    with col1:
        data = m['media_counts'].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, max(3, len(data)*0.35)))
        fig.patch.set_facecolor('#ca70ad')
        ax.barh(data.index, data.values, color='#8a1187')
        clean_plot(ax, fig, "Media Sent")
        st.pyplot(fig)

    with col2:
        data = m['link_counts'].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, max(3, len(data)*0.35)))
        fig.patch.set_facecolor('#ca70ad')
        ax.barh(data.index, data.values, color='#8a1187')
        clean_plot(ax, fig, "Links Shared")
        st.pyplot(fig)

def busiest_hour_slide():
    df = st.session_state.chat_df_2025
    hour_counts = df["timestamp"].dt.hour.value_counts().sort_index()
    slide(f"⏰ Prime Time: {hour_counts.idxmax():02d}:00", "The hour our phones never stopped vibrating.")
    fig, ax = plt.subplots(figsize=(8,3.5))
    fig.patch.set_facecolor('#ca70ad')
    ax.bar(hour_counts.index, hour_counts.values, color='#8a1187')
    ax.set_xticks(range(0,24))
    clean_plot(ax, fig, "Daily Activity Cycle", "Hour (24h)", "Messages")
    st.pyplot(fig)

def busiest_weekday_slide():
    df = st.session_state.chat_df_2025
    weekday_counts = df["timestamp"].dt.day_name().value_counts()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_counts = weekday_counts.reindex(order).fillna(0)
    slide(f"📅 {weekday_counts.idxmax()} Warrior", "The day we collectively decide to do no work.")
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor('#ca70ad')
    ax.bar(weekday_counts.index, weekday_counts.values, color='#8a1187')
    clean_plot(ax, fig, "The Weekly Grind", "Day", "Messages")
    st.pyplot(fig)

def chat_boss_award_slide():
    counts = st.session_state.chat_df_2025["sender"].value_counts()
    slide(f"👑 The Chat Boss: {counts.idxmax()}", "Absolute dominance. No one stood a chance.")
    st.markdown("### 🏆 Final Leaderboard")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {c} messages' for i,(n,c) in enumerate(counts.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(counts)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(counts.index[::-1], counts.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "The Hierarchy of Talk", "Messages", "Member")
    st.pyplot(fig)

def silent_observer_award_slide():
    counts = st.session_state.chat_df_2025["sender"].value_counts().sort_values()
    slide(f"🕶️ The Silent Observer: {counts.idxmin()}", "Watching everything, saying nothing. A true lurker.")
    st.markdown("### 📉 The Low-Key List")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {c} messages' for i,(n,c) in enumerate(counts.items())])}</pre>", unsafe_allow_html=True)

def early_bird_slide():
    m = st.session_state.metrics
    ratios = m['morning_ratio'].sort_values(ascending=False)
    slide(f"🌅 The Early Bird: {ratios.idxmax()}", "They speak while the rest of us are still dreaming.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%} of their texts' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(ratios)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(ratios.index[::-1], ratios.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "Sunrise Activity Ratio")
    st.pyplot(fig)

def night_owl_slide():
    m = st.session_state.metrics
    ratios = m['night_ratio'].sort_values(ascending=False)
    slide(f"🌙 The Night Owl: {ratios.idxmax()}", "The group chat doesn't sleep until they do.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%} of their texts' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(ratios)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(ratios.index[::-1], ratios.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "Midnight Activity Ratio")
    st.pyplot(fig)

def weekend_warrior_slide():
    m = st.session_state.metrics
    ratios = m['weekend_ratio'].sort_values(ascending=False)
    slide(f"🎉 The Weekend Warrior: {ratios.idxmax()}", "They wait all week just to drop the Saturday heat.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%}' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)

def weekday_distractor_slide():
    m = st.session_state.metrics
    ratios = m['weekday_ratio'].sort_values(ascending=False)
    slide(f"🧑‍💻 The Professional Distractor: {ratios.idxmax()}", "Keeping the group alive during work hours.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%}' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)

def message_length_awards_slide():
    m = st.session_state.metrics
    lengths = m['sender_avg_len'].sort_values(ascending=False)
    slide(f"📜 The Philosopher: {lengths.idxmax()}", "Why send one word when you can send a whole novel?")
    st.markdown(f"**Special Mention:** 🧩 {lengths.idxmin()} (The One-Word Warrior)")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {l:.1f} characters' for i,(n,l) in enumerate(lengths.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(lengths)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(lengths.index[::-1], lengths.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "Average Characters per Message")
    st.pyplot(fig)

def deleted_messages_slide():
    m = st.session_state.metrics
    ratios = m['deleted_ratio']
    
    if not ratios.empty and ratios.max() > 0:
        winner = ratios.idxmax()
        slide(f"🕵️ The Evidence Buster: {winner}", "What was so scandalous it had to be removed? We'll never know.")
    
        st.markdown("### 🔍 The 'Oops' Leaderboard")
        st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%} of their messages' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(8, max(3, len(ratios)*0.35)))
        fig.patch.set_facecolor('#ca70ad')
        ax.barh(ratios.index[::-1], ratios.values[::-1], color='#8a1187')
        clean_plot(ax, fig, "Percentage of Messages Deleted")
        st.pyplot(fig)
    else:
        slide("😇 The Honest Squad", "Zero deleted messages found. Nothing to hide here!")

def spam_lord_slide():
    m = st.session_state.metrics
    gaps = pd.Series(m['sender_gaps']).sort_values()
    slide(f"🧨 The Spam Lord: {gaps.idxmin()}", "Fastest fingers in the west. Good luck keeping up.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {v:.1f}s between texts' for i,(n,v) in enumerate(gaps.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(gaps)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(gaps.index[::-1], gaps.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "Rapid-Fire Gap (Lower = More Spam)")
    st.pyplot(fig)

def conversation_starter_slide():
    m = st.session_state.metrics
    counts = m['starters'].sort_values(ascending=False)
    slide(f"🌅 The Ice Breaker: {counts.idxmax()}", "The brave soul who revives the chat after every silence.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {c} revivals' for i,(n,c) in enumerate(counts.items())])}</pre>", unsafe_allow_html=True)

def chat_closer_slide():
    m = st.session_state.metrics
    counts = m['closers'].sort_values(ascending=False)
    slide(f"💤 The Chat Closer: {counts.idxmax()}", "Always gets the last word. Always.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {c} final words' for i,(n,c) in enumerate(counts.items())])}</pre>", unsafe_allow_html=True)

def emoji_awards_slide():
    m = st.session_state.metrics
    avg = pd.Series(m['avg_emoji']).sort_values(ascending=True)
    slide(f"👑 Emoji Emperor: {avg.idxmax()}", "They don't just speak English; they speak Emoji.")
    
    st.markdown("### 🏆 The Decoration Leaderboard")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {v:.2f} per msg' for i,(n,v) in enumerate(avg.iloc[::-1].items())])}</pre>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(avg)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(avg.index, avg.values, color='#8a1187')
    clean_plot(ax, fig, "Emoji Density (Average per Message)")
    st.pyplot(fig)
    
def media_per_message_slide():
    df = st.session_state.chat_df_2025
    ratio = (df[df['message'].str.contains("<Media omitted", na=False)]['sender'].value_counts() / df['sender'].value_counts()).fillna(0).sort_values(ascending=False)
    slide(f"📷 The Sensory Learner: {ratio.idxmax()}", "Voicenotes and pictures are worth 1,000 texts, and they know it.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%}' for i,(n,r) in enumerate(ratio.items())])}</pre>", unsafe_allow_html=True)

def links_per_message_slide():
    df = st.session_state.chat_df_2025
    ratio = (df[df['message'].str.contains(LINK_REGEX, na=False)]['sender'].value_counts() / df['sender'].value_counts()).fillna(0).sort_values(ascending=False)
    slide(f"🔗 The Link Librarian: {ratio.idxmax()}", "Our personal source for everything on the internet.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%}' for i,(n,r) in enumerate(ratio.items())])}</pre>", unsafe_allow_html=True)

def tag_sniper_slide():
    m = st.session_state.metrics
    by_sender = defaultdict(Counter)
    for (s, t), c in m['pair_counts'].items(): by_sender[s][t] += c
    sniper_favs = {s: tgts.most_common(1)[0] for s, tgts in by_sender.items()}
    
    sorted_snipers = sorted(m['total_tags_sent'].items(), key=lambda x: x[1], reverse=True)
    
    winner, total_shots = sorted_snipers[0]
    fav_target = sniper_favs[winner][0]
    
    slide(f"🏹 The Tag Sniper: {winner}", f"Fired off {total_shots} total tags! Favorite target: **{fav_target}**.")
    
    st.markdown("### 🎯 Sniper Leaderboard")
    rows = [f"{i+1}. {s}: {c} tags" for i, (s, c) in enumerate(sorted_snipers)]
    st.markdown(f"<pre>{'<br>'.join(rows)}</pre>", unsafe_allow_html=True)
    
    names = [x[0] for x in sorted_snipers[:10]][::-1]
    counts = [x[1] for x in sorted_snipers[:10]][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(names, counts, color='#8a1187')
    clean_plot(ax, fig, "Sniper Power (Total Tags Sent)")
    st.pyplot(fig)

def tag_magnet_slide():
    m = st.session_state.metrics
    incoming = {mem: Counter() for mem in m['real_members']}

    for (tagger, target), c in m['pair_counts'].items():
        if target in incoming: incoming[target][tagger] += c

    magnet_sources = {mem: srcs.most_common(1)[0] for mem, srcs in incoming.items() if srcs}
    valid_magnets = {name: count for name, count in m['total_tags_received'].items() if name in m['real_members']}
    sorted_magnets = sorted(valid_magnets.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_magnets:
        slide("🧲 The Tag Magnet", "No tags found between members!")
        return

    winner, total_pulls = sorted_magnets[0]
    
    slide(f"🧲 The Tag Magnet: {winner}", f"Everyone's calling... pulled into the chat {total_pulls} times!")
    
    st.markdown("### 📢 Most Wanted")
    rows = [f"{i+1}. {m_name}: {c} tags" for i, (m_name, c) in enumerate(sorted_magnets)]
    st.markdown(f"<pre>{'<br>'.join(rows)}</pre>", unsafe_allow_html=True)
    
    names = [x[0] for x in sorted_magnets[:10]][::-1]
    counts = [x[1] for x in sorted_magnets[:10]][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(names, counts, color='#8a1187')
    clean_plot(ax, fig, "Magnetism Level (Total Tags Received)")
    st.pyplot(fig)

def final_wrap_up_slide():
    m = st.session_state.metrics
    df = st.session_state.chat_df_2025
    hall = m['hall_of_fame']
    
    slide("🎬 The 2025 Grand Finale", "One legendary year. One legendary squad.")
    st.write("")
    
    def award_card(title, winner, emoji):
        return f"""
        <div style="background: linear-gradient(135deg, #8a1187, #8a1187); 
                    padding: 15px; border-radius: 12px; margin-bottom: 10px; 
                    border: 1px solid rgba(255,255,255,0.3); text-align: center;">
            <p style="margin:0; font-size:10px; font-weight:bold; color: rgba(255,255,255,0.7); text-transform: uppercase;">{emoji} {title}</p>
            <p style="margin:0; font-size:18px; font-weight:bold; color: white;">{winner}</p>
        </div>
        """

    col1, col2 = st.columns(2)
    
    awards = [
        ("The Chat Boss", f"{hall['boss']}", "👑"),
        ("The Spam Lord", hall['spam_lord'], "🧨"),
        ("Evidence Buster", f"{m['deleted_ratio'].idxmax() if not m['deleted_ratio'].empty else 'None'}", "🕵️"),
        ("Silent Observer", df["sender"].value_counts().idxmin(), "🕶️"),
        ("Ice Breaker", hall['starter'], "🌅"),
        ("The Closer", hall['closer'], "💤"),
        ("Night Owl", hall['night_owl'], "🌙"),
        ("The Philosopher", hall['philosopher'], "📜"),
        ("Emoji Emperor", f"{pd.Series(m['avg_emoji']).idxmax() if m['avg_emoji'] else 'None'}", "🎭"),
        ("Tag Sniper", f"{hall['sniper']}", "🏹"),
        ("Tag Magnet", f"{hall['magnet']}", "🧲"),
        ("Sensory Learner", f"{(df[df['message'].str.contains('<Media omitted', na=False)]['sender'].value_counts() / df['sender'].value_counts()).idxmax()}", "📸"),
        ("Link Librarian", f"{(df[df['message'].str.contains(LINK_REGEX, na=False)]['sender'].value_counts() / df['sender'].value_counts()).idxmax()}", "🔗"),
    ]

    for i, (title, winner, emoji) in enumerate(awards):

        if i % 2 == 0:
            with col1: st.markdown(award_card(title, winner, emoji), unsafe_allow_html=True)
        else:
            with col2: st.markdown(award_card(title, winner, emoji), unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: #8a1187; margin-top:30px; font-family: system-ui;'>Another year of us. See you in the next one.. 🥂</h2>", unsafe_allow_html=True)
    st.balloons()

# --- APP ROUTING ---
if "slide" not in st.session_state: st.session_state.slide = 0

slides, slide_names = [first_slide], ["Welcome"]

if "chat_df_2025" in st.session_state:

    active = [
        (total_messages_slide, "Total Messages"), (monthly_messages_slide, "Calendar"),
        (emoji_stats_slide, "Emoji Holy Grail"), (favorite_emoji_per_user_slide, "Signatures"),
        (media_link_senders_slide, "Curators"), (busiest_hour_slide, "Prime Time"),
        (busiest_weekday_slide, "Weekly Grind"), (chat_boss_award_slide, "Chat Boss"),
        (silent_observer_award_slide, "Silent Observer"), (early_bird_slide, "Early Bird"),
        (night_owl_slide, "Night Owl"), (weekend_warrior_slide, "Weekend Warrior"),
        (weekday_distractor_slide, "Distractor"), (message_length_awards_slide, "The Novelist"),
        (spam_lord_slide, "The Spammer"), (deleted_messages_slide, "Evidence Buster"), 
        (tag_sniper_slide, "Sniper"), (tag_magnet_slide, "Magnet"),
        (conversation_starter_slide, "Ice Breaker"), (chat_closer_slide, "The Closer"),
        (emoji_awards_slide, "Emoji Emperor"), (media_per_message_slide, "Visual Learner"),
        (links_per_message_slide, "Librarian"), (final_wrap_up_slide, "Finale")
    ]

    for fn, n in active:
        slides.append(fn); slide_names.append(n)

# Sidebar
st.sidebar.title("📑 The Slide Deck")
selected = st.sidebar.radio("Jump to", range(len(slides)), format_func=lambda i: slide_names[i], index=st.session_state.slide)

if selected != st.session_state.slide:
    st.session_state.slide = selected; st.rerun()

st.progress((st.session_state.slide + 1) / len(slides))
slides[st.session_state.slide]()

c1, c2 = st.columns([8,1])

with c1:

    if st.session_state.slide > 0 and st.button("← Previous"):
        st.session_state.slide -= 1; st.rerun()
with c2:
    
    if st.session_state.slide < len(slides)-1 and st.button("Next →"):
        st.session_state.slide += 1; st.rerun()