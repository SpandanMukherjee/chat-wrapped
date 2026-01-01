import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
from dateutil import parser as date_parser
 

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
    st.write("") 

# --- Safety Helpers ---
def safe_idxmax(s, default="None"):

    try:

        if getattr(s, "empty", False):
            return default
        return s.idxmax()
    
    except Exception:

        try:
            return max(s) if s else default
        except Exception:
            return default

def safe_idxmin(s, default="None"):

    try:

        if getattr(s, "empty", False):
            return default
        return s.idxmin()
    
    except Exception:

        try:
            return min(s) if s else default
        except Exception:
            return default

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
                timestamp_str = timestamp_str.replace('\u202f', ' ').replace('\u00a0', ' ').strip()
                # Use dateutil to handle all WhatsApp date formats (99% coverage)
                timestamp = date_parser.parse(timestamp_str, dayfirst=True)
            except:
                timestamp = None

            data.append({"timestamp": timestamp, "sender": sender, "message": message})

    return pd.DataFrame(data)

# --- Threshold Filter ---
def filter_df_by_threshold(df, min_messages):

    if min_messages <= 0:
        return df
    
    sender_counts = df['sender'].value_counts()
    valid_senders = sender_counts[sender_counts >= min_messages].index
    return df[df['sender'].isin(valid_senders)]

# --- PERFORMANCE ENGINE ---
@st.cache_data(show_spinner=False)
def compute_master_metrics(df, min_threshold=0):
    m = {}
    df_orig = df.copy()  # Keep original for raw counts
    df = df.copy().sort_values('timestamp')
    df_filtered = filter_df_by_threshold(df, min_threshold)  # Filtered for ratios
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.day_name()
    df_filtered['hour'] = df_filtered['timestamp'].dt.hour
    df_filtered['weekday'] = df_filtered['timestamp'].dt.day_name()
    total_counts = df['sender'].value_counts()  # Raw counts
    total_counts_filtered = df_filtered['sender'].value_counts()  # Filtered counts
    
    # Volume
    df['date_only'] = df['timestamp'].dt.date
    m['total_messages'] = len(df)
    m['most_active_day'] = df.groupby('date_only').size().idxmax()
    m['most_active_day_count'] = df.groupby('date_only').size().max()
    
    # Monthly
    df['month_dt'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
    m['monthly_counts'] = df.groupby('month_dt').size()
    
    # Emoji Intelligence (using filtered data for per-message averages)
    all_emojis = []
    sender_emoji_counts = Counter()
    sender_emoji_map = defaultdict(Counter)
    msg_counts = Counter()

    for msg, sender in zip(df_filtered["message"], df_filtered["sender"]):

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
    
    # Time Analysis (using filtered data for ratios)
    m['morning_ratio'] = (df_filtered[(df_filtered['hour'] >= 4) & (df_filtered['hour'] < 12)]['sender'].value_counts() / total_counts_filtered).fillna(0)
    m['night_ratio'] = (df_filtered[(df_filtered['hour'] >= 21) | (df_filtered['hour'] < 4)]['sender'].value_counts() / total_counts_filtered).fillna(0)
    m['weekend_ratio'] = (df_filtered[df_filtered['weekday'].isin(["Saturday", "Sunday"])]['sender'].value_counts() / total_counts_filtered).fillna(0)
    m['weekday_ratio'] = (df_filtered[df_filtered['weekday'].isin(["Monday","Tuesday","Wednesday","Thursday","Friday"])]['sender'].value_counts() / total_counts_filtered).fillna(0)

    # Complexity (using filtered data for averages)
    df["msg_len"] = df["message"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df_filtered["msg_len"] = df_filtered["message"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    m['sender_avg_len'] = df_filtered.groupby("sender")["msg_len"].mean()

    # Evidence Buster Logic
    deleted_df = df[df['message'].str.contains("This message was deleted|You deleted this message", na=False)]
    m['deleted_counts'] = deleted_df['sender'].value_counts()
    m['deleted_ratio'] = (m['deleted_counts'] / total_counts.replace(0, 1)).fillna(0).sort_values(ascending=False)

    # Rapid Fire (using filtered data)
    sender_gaps = {}
    df_sorted = df_filtered.sort_values('timestamp')

    for sender, group in df_sorted.groupby('sender'):
        gaps = group['timestamp'].diff().dt.total_seconds().dropna()
        active = gaps[gaps <= 300]

        if not active.empty: sender_gaps[sender] = active.mean()

    m['sender_gaps'] = sender_gaps

    # --- Connections (Tags) --- (using filtered data)
    pair_counts = Counter()
    total_tags_sent = Counter()
    total_tags_received = Counter()
    START, END = "\u2068", "\u2069"
    real_members = set(df_filtered["sender"].unique()) - {"Meta AI"}
    
    for sender, msg in zip(df_filtered["sender"], df_filtered["message"].fillna("")):
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

    # Social Rhythms (using filtered data)
    df_sorted['gap_h'] = df_sorted['timestamp'].diff().dt.total_seconds() / 3600
    m['starters'] = df_sorted[(df_sorted['gap_h'] >= 3) | (df_sorted['gap_h'].isna())]['sender'].value_counts()
    df_sorted['next_gap'] = df_sorted['gap_h'].shift(-1)
    m['closers'] = df_sorted[df_sorted['next_gap'] >= 3]['sender'].value_counts()

    # --- The Big Summary ---
    m['hall_of_fame'] = {
        'boss': total_counts_filtered.idxmax() if not total_counts_filtered.empty else "None",
        'sniper': total_tags_sent.most_common(1)[0][0] if total_tags_sent else "None",
        'magnet': total_tags_received.most_common(1)[0][0] if total_tags_received else "None",
        'starter': m['starters'].idxmax() if not m['starters'].empty else "None",
        'closer': m['closers'].idxmax() if not m['closers'].empty else "None",
        'spam_lord': min(sender_gaps, key=sender_gaps.get) if sender_gaps else "None",
        'night_owl': m['night_ratio'].idxmax() if not m['night_ratio'].empty else "None",
        'early_bird': m['morning_ratio'].idxmax() if not m['morning_ratio'].empty else "None",
        'media_king': m['media_counts'].idxmax() if not m['media_counts'].empty else "None",
        'philosopher': m['sender_avg_len'].idxmax() if not m['sender_avg_len'].empty else "None"
    }
    return m

# --- Sample File Generator ---
def generate_sample_chat():

    START, END = "\u2068", "\u2069"  # Invisible tag markers
    sample = f"""01/01/2025, 09:15 AM - Alice: Good morning everyone! ☀️
01/01/2025, 09:16 AM - Bob: Hey Alice! 👋
01/01/2025, 09:17 AM - Charlie: Happy New Year! 🎉
01/01/2025, 10:45 AM - Diana: Did you guys watch the fireworks? 🎆
01/02/2025, 02:30 PM - Alice: Check this out www.example.com
01/02/2025, 02:31 PM - Bob: <Media omitted>
01/03/2025, 08:20 AM - Charlie: That was insane 😂😂
01/03/2025, 08:21 AM - Diana: Literally can't stop laughing 💀
01/05/2025, 06:15 PM - Alice: Anyone down for pizza tonight?
01/05/2025, 06:16 PM - Bob: This message was deleted
01/05/2025, 06:17 PM - Charlie: Count me in! 🍕
01/07/2025, 11:30 PM - Diana: {START}Alice{END} you coming to the party? 🎊
01/08/2025, 12:45 AM - Alice: Yeah! Can't wait 🥳
01/15/2025, 03:45 PM - Bob: Just finished the project
01/15/2025, 03:46 PM - Charlie: Nice! Let me check it out
01/16/2025, 09:20 AM - Diana: So what's the plan for next weekend? 🤔
01/16/2025, 09:21 AM - Alice: {START}Charlie{END} you in? 🏔️
01/20/2025, 05:30 PM - Bob: Check this https://youtube.com/watch?v=example
01/20/2025, 05:31 PM - Charlie: OMG this is gold 😂
01/25/2025, 11:15 PM - Alice: Can't sleep 😅
01/25/2025, 11:16 PM - Diana: {START}Bob{END} same here, what's up? 🌙
01/26/2025, 07:30 AM - Charlie: Morning folks! ☕
01/28/2025, 04:20 PM - Alice: <Media omitted>
01/28/2025, 04:21 PM - Bob: That looks amazing! 🤩
02/01/2025, 08:45 AM - Diana: {START}Alice{END} New month, new goals! 💪
02/05/2025, 02:15 PM - Charlie: {START}Bob{END} {START}Diana{END} Anyone free tomorrow?
02/10/2025, 10:30 AM - Alice: This weather is insane ☀️☀️☀️
02/14/2025, 06:45 PM - Bob: Happy Valentine's Day everyone! 💕
02/20/2025, 11:20 PM - Diana: {START}Charlie{END} Movie night was so fun! 🎬
02/21/2025, 12:30 AM - Charlie: {START}Alice{END} {START}Bob{END} Best night ever! 😄"""
    return sample

# --- SLIDE DECK ---

def first_slide():
    slide("🎉 2025: A Year in Chat", "Your digital memories, decoded and delivered.")

    if "chat_df_2025" not in st.session_state:
        # Help/Instructions Section
        with st.expander("📖 How to Use This App", expanded=False):
            st.markdown("""
            ### Step 1: Export Your WhatsApp Chat
                        
            1. Open WhatsApp and go to the group chat you want to export
            2. Tap **Menu (⋮)** → **More** → **Export chat**
            3. Select **"Without Media"** (this is important!)
            4. Choose where to save the file
            
            ### Step 2: Upload Here
            - Download the `.txt` file to your computer
            - Click the upload box below and select your exported chat file
            - Make sure it contains messages from **2025**
            
            ### Step 3: Set Preferences
            - Use the slider to set a minimum message threshold
            - This filters out inactive members from awards and statistics
            - Raw message counts stay accurate regardless
            
            ### Step 4: Enjoy Your Wrapped!
            - Click "Confirm & Start Wrapped" and let the app analyze your chat
            - Navigate through all the fun stats and awards
            - Use the sidebar to jump between slides
            
            ### ⚠️ Important Notes
            - Only **.txt files** are supported
            - Export **without media** to reduce file size
            - The app only analyzes messages from **2025**
            - Your data is processed locally and not stored
            """)
        
        st.markdown("")
        uploaded = st.file_uploader("Drop your WhatsApp export here (.txt)", type=["txt"])
        
        # Show sample file download
        st.markdown("---")
        st.markdown("### Sample File")
        sample_content = generate_sample_chat()
        st.download_button(
            label="⬇️ Download Sample Chat Export",
            data=sample_content,
            file_name="sample_chat_export.txt",
            mime="text/plain",
            help="Download this sample file to see the expected WhatsApp export format"
        )

        try:

            if uploaded:
                st.toast("Chat uploaded, Loading...", icon="❤️‍🔥")
                raw_df = load_whatsapp_chat(uploaded)
                df = raw_df[raw_df['timestamp'].notnull()]
                df = df[(df['timestamp'] >= datetime(2025, 1, 1)) & (df['timestamp'] < datetime(2026, 1, 1))]
                df = df[df['sender'] != "Meta AI"]

                if df.empty:
                    st.error("No messages found for the year 2025. Please upload a valid WhatsApp chat export that includes messages from 2025.")
                else:
                    st.session_state.chat_df_2025 = df
                    if "min_threshold" not in st.session_state:
                        st.session_state.min_threshold = 0

        except Exception as e:
            st.error(f"Error processing chat file:")
            st.info("Make sure the file is a valid WhatsApp chat export (without media) in .txt format.")
            print(f"Dev Log: {e}")
    
    # Show slider and confirmation after file is loaded
    if "chat_df_2025" in st.session_state and "metrics" not in st.session_state:
        st.markdown("### ✨ Set Your Preferences")
        
        threshold = st.slider(
            "Minimum messages per member (for awards & ratios):",
            min_value=0,
            max_value=min(200, len(st.session_state.chat_df_2025) // 2),
            value=0,
            step=1,
            help="Members with fewer messages won't be considered for awards and ratio calculations. Raw message counts are unaffected."
        )
        
        # Disable button while processing
        button_disabled = "processing" in st.session_state
        
        if st.button("✅ Confirm & Start Wrapped", use_container_width=True, disabled=button_disabled):
            st.session_state.processing = True
            st.toast("✨ Processing your preferences... This may take a moment for large files.", icon="⏳")
            st.session_state.min_threshold = threshold
            st.session_state.metrics = compute_master_metrics(st.session_state.chat_df_2025, threshold)
            st.session_state.slide = 1
            del st.session_state.processing
            st.rerun()

def total_messages_slide():
    
    if "snow_played" not in st.session_state:
        st.snow()
        st.session_state.snow_played = True

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

def great_silence_slide():
    df = st.session_state.chat_df_2025.sort_values('timestamp').copy()
    df['gap'] = df['timestamp'].diff()
    if not df['gap'].dropna().empty:
        max_gap = df['gap'].max()
        idx = df['gap'].idxmax()
        end_date = df.loc[idx, 'timestamp']
        start_date = end_date - max_gap
        total_seconds = int(max_gap.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600

        slide("🏜️ The Great Silence", f"The chat went ghost for {days} days and {hours} hours.")
        st.markdown(f"""
            <div style='text-align:center; background:rgba(255,255,255,0.1); padding:20px; border-radius:15px; border: 1px solid rgba(255,255,255,0.2);'>
                <p style="margin:0; opacity:0.8;">From: <b>{start_date.strftime('%B %d, %H:%M')}</b></p>
                <p style="margin:5px 0;">⬇️</p>
                <p style="margin:0; opacity:0.8;">To: <b>{end_date.strftime('%B %d, %H:%M')}</b></p>
            </div>
        """, unsafe_allow_html=True)

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
    if getattr(ratios, 'empty', False):
        slide("🌅 The Early Bird: None", "No morning activity found.")
        return
    winner = safe_idxmax(ratios, "None")
    slide(f"🌅 The Early Bird: {winner}", "They speak while the rest of us are still dreaming.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%} of their texts' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(ratios)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(ratios.index[::-1], ratios.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "Sunrise Activity Ratio")
    st.pyplot(fig)

def night_owl_slide():
    m = st.session_state.metrics
    ratios = m['night_ratio'].sort_values(ascending=False)
    if getattr(ratios, 'empty', False):
        slide("🌙 The Night Owl: None", "No night activity found.")
        return
    winner = safe_idxmax(ratios, "None")
    slide(f"🌙 The Night Owl: {winner}", "The group chat doesn't sleep until they do.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%} of their texts' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(ratios)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(ratios.index[::-1], ratios.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "Midnight Activity Ratio")
    st.pyplot(fig)

def weekend_warrior_slide():
    m = st.session_state.metrics
    ratios = m['weekend_ratio'].sort_values(ascending=False)
    if getattr(ratios, 'empty', False):
        slide("🎉 The Weekend Warrior: None", "No weekend activity found.")
        return
    winner = safe_idxmax(ratios, "None")
    slide(f"🎉 The Weekend Warrior: {winner}", "They wait all week just to drop the Saturday heat.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%}' for i,(n,r) in enumerate(ratios.items())])}</pre>", unsafe_allow_html=True)

def weekday_distractor_slide():
    m = st.session_state.metrics
    ratios = m['weekday_ratio'].sort_values(ascending=False)
    if getattr(ratios, 'empty', False):
        slide("🧑‍💻 The Professional Distractor: None", "No weekday activity found.")
        return
    winner = safe_idxmax(ratios, "None")
    slide(f"🧑‍💻 The Professional Distractor: {winner}", "Keeping the group alive during work hours.")
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
    if gaps.empty:
        slide("🧨 The Spam Lord: None", "No rapid-fire users found.")
        return
    winner = safe_idxmin(gaps, "None")
    slide(f"🧨 The Spam Lord: {winner}", "Fastest fingers in the west. Good luck keeping up.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {v:.1f}s between texts' for i,(n,v) in enumerate(gaps.items())])}</pre>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(gaps)*0.35)))
    fig.patch.set_facecolor('#ca70ad')
    ax.barh(gaps.index[::-1], gaps.values[::-1], color='#8a1187')
    clean_plot(ax, fig, "Rapid-Fire Gap (Lower = More Spam)")
    st.pyplot(fig)

def conversation_starter_slide():
    m = st.session_state.metrics
    counts = m['starters'].sort_values(ascending=False)
    if getattr(counts, 'empty', False):
        slide("🌅 The Ice Breaker: None", "No conversation starters found.")
        return
    winner = safe_idxmax(counts, "None")
    slide(f"🌅 The Ice Breaker: {winner}", "The brave soul who revives the chat after every silence.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {c} revivals' for i,(n,c) in enumerate(counts.items())])}</pre>", unsafe_allow_html=True)

def chat_closer_slide():
    m = st.session_state.metrics
    counts = m['closers'].sort_values(ascending=False)
    if getattr(counts, 'empty', False):
        slide("💤 The Chat Closer: None", "No chat closers found.")
        return
    winner = safe_idxmax(counts, "None")
    slide(f"💤 The Chat Closer: {winner}", "Always gets the last word. Always.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {c} final words' for i,(n,c) in enumerate(counts.items())])}</pre>", unsafe_allow_html=True)

def emoji_awards_slide():
    m = st.session_state.metrics
    avg = pd.Series(m['avg_emoji']).sort_values(ascending=True)
    if getattr(avg, 'empty', False):
        slide("👑 Emoji Emperor: None", "No emoji usage data found.")
        return
    winner = safe_idxmax(avg, "None")
    slide(f"👑 Emoji Emperor: {winner}", "They don't just speak English; they speak Emoji.")
    
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
    if getattr(ratio, 'empty', False):
        slide("📷 The Sensory Learner: None", "No media detected in chat.")
        return
    winner = safe_idxmax(ratio, "None")
    slide(f"📷 The Sensory Learner: {winner}", "Voicenotes and pictures are worth 1,000 texts, and they know it.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%}' for i,(n,r) in enumerate(ratio.items())])}</pre>", unsafe_allow_html=True)

def links_per_message_slide():
    df = st.session_state.chat_df_2025
    ratio = (df[df['message'].str.contains(LINK_REGEX, na=False)]['sender'].value_counts() / df['sender'].value_counts()).fillna(0).sort_values(ascending=False)
    if getattr(ratio, 'empty', False):
        slide("🔗 The Link Librarian: None", "No links shared in chat.")
        return
    winner = safe_idxmax(ratio, "None")
    slide(f"🔗 The Link Librarian: {winner}", "Our personal source for everything on the internet.")
    st.markdown(f"<pre>{'<br>'.join([f'{i+1}. {n}: {r:.1%}' for i,(n,r) in enumerate(ratio.items())])}</pre>", unsafe_allow_html=True)

def tag_sniper_slide():
    m = st.session_state.metrics
    by_sender = defaultdict(Counter)
    for (s, t), c in m['pair_counts'].items(): by_sender[s][t] += c
    sniper_favs = {s: tgts.most_common(1)[0] for s, tgts in by_sender.items()}
    
    sorted_snipers = sorted(m['total_tags_sent'].items(), key=lambda x: x[1], reverse=True)
    if not sorted_snipers:
        slide("🏹 The Tag Sniper: None", "No tags sent among members.")
        return

    winner, total_shots = sorted_snipers[0]
    fav_target = sniper_favs.get(winner, ("None", 0))[0]

    slide(f"🏹 The Tag Sniper: {winner}", f"Fired off {total_shots} total tags! Favorite target: {fav_target}.")
    
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

def legend_search_slide():
    df = st.session_state.chat_df_2025.copy()
    slide("🔍 The Legend Search", "Settle the debates. How many times did we actually say it?")
    q = st.text_input("Search for an inside joke, a name, or a word:", "lol")
    
    if q:
        pattern = r'\b' + re.escape(q) + r'\b'
        df['temp_count'] = df['message'].str.count(pattern, flags=re.IGNORECASE).fillna(0).astype(int)
        res = df[df['temp_count'] > 0].copy()
        total_mentions = int(df['temp_count'].sum())
        
        st.markdown(f"<h1 style='text-align:center; color:#8a1187; font-size:60px;'>{total_mentions}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Total mentions of <b>'{q}'</b> in 2025</p>", unsafe_allow_html=True)
        
        if not res.empty:
            peak_hour = res['timestamp'].dt.hour.value_counts().idxmax()
            time_display = f"{peak_hour:02d}:00"
            user_counts = df.groupby('sender')['temp_count'].sum().sort_values(ascending=False)
            top_user = user_counts.index[0]
            top_val = int(user_counts.values[0])
            
            st.info(f"🏆 {top_user} is the biggest fan of this word, saying it {top_val} times.")
            st.write(f"⏰ This word is most commonly used around {time_display}.")
        else:
            st.warning(f"Nobody said '{q}' as a standalone word this year!")

def final_wrap_up_slide():
    m = st.session_state.metrics
    df = st.session_state.chat_df_2025
    hall = m['hall_of_fame']
    
    # compute safe winners for ratios that may be empty
    sensory_ratio = (df[df['message'].str.contains('<Media omitted', na=False)]['sender'].value_counts() / df['sender'].value_counts()).fillna(0)
    sensory_winner = safe_idxmax(sensory_ratio, 'None')
    link_ratio = (df[df['message'].str.contains(LINK_REGEX, na=False)]['sender'].value_counts() / df['sender'].value_counts()).fillna(0)
    link_winner = safe_idxmax(link_ratio, 'None')

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
        ("Early Bird", hall['early_bird'], "🌅"),
        ("The Philosopher", hall['philosopher'], "📜"),
        ("Emoji Emperor", f"{pd.Series(m['avg_emoji']).idxmax() if m['avg_emoji'] else 'None'}", "🎭"),
        ("Tag Sniper", f"{hall['sniper']}", "🏹"),
        ("Tag Magnet", f"{hall['magnet']}", "🧲"),
        ("Sensory Learner", f"{sensory_winner}", "📸"),
        ("Link Librarian", f"{link_winner}", "🔗"),
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
        (total_messages_slide, "Total Messages"), (monthly_messages_slide, "Calendar"), (great_silence_slide, "The Great Silence"),
        (emoji_stats_slide, "Emoji Holy Grail"), (favorite_emoji_per_user_slide, "Signatures"),
        (media_link_senders_slide, "Curators"), (busiest_hour_slide, "Prime Time"),
        (busiest_weekday_slide, "Weekly Grind"), (chat_boss_award_slide, "Chat Boss"),
        (silent_observer_award_slide, "Silent Observer"), (early_bird_slide, "Early Bird"),
        (night_owl_slide, "Night Owl"), (weekend_warrior_slide, "Weekend Warrior"),
        (weekday_distractor_slide, "Distractor"), (message_length_awards_slide, "The Novelist"),
        (spam_lord_slide, "The Spammer"), (deleted_messages_slide, "Evidence Buster"), 
        (tag_sniper_slide, "Sniper"), (tag_magnet_slide, "Magnet"),
        (conversation_starter_slide, "Ice Breaker"), (chat_closer_slide, "The Closer"),
        (emoji_awards_slide, "Emoji Emperor"), (media_per_message_slide, "Sensory Learner"),
        (links_per_message_slide, "Librarian"), (legend_search_slide, "Legend Search"), (final_wrap_up_slide, "Finale")
    ]

    for fn, n in active:
        slides.append(fn); slide_names.append(n)

# Always show the current slide
if st.session_state.slide < len(slides):
    slides[st.session_state.slide]()

if "metrics" in st.session_state:
    # Sidebar
    st.sidebar.title("📑 The Slide Deck")
    selected = st.sidebar.radio("Jump to", range(len(slides)), format_func=lambda i: slide_names[i], index=st.session_state.slide)

    if selected != st.session_state.slide:
        st.session_state.slide = selected; st.rerun()

    st.progress((st.session_state.slide + 1) / len(slides))

    c1, c2 = st.columns([8,1])

    with c1:
        st.write("")

        if st.session_state.slide > 0 and st.button("← Previous"):
            st.session_state.slide -= 1; st.rerun()
    with c2:
        st.write("")

        if st.session_state.slide < len(slides)-1 and st.button("Next →"):
            st.session_state.slide += 1; st.rerun()