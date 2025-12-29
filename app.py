import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
import numpy as np

st.set_page_config(page_title="Chat Wrapped", layout="centered")

# Slide function
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
                font-family: system-ui;
                background: linear-gradient(135deg, {gradient[0]}, {gradient[1]});
                border-radius: 12px;
                padding: 24px;
            ">
                <h1 style="font-size: 34px; margin-bottom: 10px;">{title}</h1>
                <p style="font-size: 20px; opacity: .95;">{subtitle}</p>
            </div>
        """,
        unsafe_allow_html=True
    )

# WhatsApp parser
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
                timestamp = datetime.strptime(timestamp_str.strip(), "%d/%m/%Y, %I:%M %p")
            except:
                timestamp = None
            data.append({"timestamp": timestamp, "sender": sender, "message": message})
    return pd.DataFrame(data)

# --- Initialize slide index ---
if "slide" not in st.session_state:
    st.session_state.slide = 0

# --- Build slides dynamically each run ---
slides = []
slide_names = []

def add_slide(fn, name):
    slides.append(fn)
    slide_names.append(name)

# First slide: welcome + uploader
def first_slide():
    slide("🎉 Welcome to Chat Wrapped", "Upload your chat to get started!")
    if "chat_df" not in st.session_state:
        uploaded = st.file_uploader("", type=["txt"])
        if uploaded:
            st.session_state.chat_df = load_whatsapp_chat(uploaded)
            df = st.session_state.chat_df
            df = df[df['timestamp'].notnull()]
            df = df[(df['timestamp'] >= datetime(2025, 1, 1)) & (df['timestamp'] < datetime(2026, 1, 1))]
            df = df[df['sender'] != "Meta AI"]
            st.session_state.chat_df_2025 = df
            st.success("Chat loaded! 🎉")
            st.session_state.slide = 1
            st.rerun()

add_slide(first_slide, "Welcome")

# Stats slide (total messages)

def total_messages_slide():
    df = st.session_state.chat_df_2025.copy()
    total_messages = len(df)

    # --- Find day with most messages ---
    df['date_only'] = df['timestamp'].dt.date
    most_active_day_count = df.groupby('date_only').size().max()
    most_active_day = df.groupby('date_only').size().idxmax()  # datetime.date object

    # Format day nicely
    day_str = most_active_day.strftime("%d %b %Y")

    slide(
        f"🔥 The group sent {total_messages} messages this year!",
        f"📅 Top day: {day_str} with {most_active_day_count} messages!"
    )

# Monthly messages graph slide
def monthly_messages_slide():
    df = st.session_state.chat_df_2025.copy()
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    monthly_counts.index = monthly_counts.index.to_timestamp()

    slide("📊 Messages Per Month", "See how your chat activity was distributed this year")
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("#ca70ad")
    ax.bar(monthly_counts.index.strftime('%b'), monthly_counts.values, color='#8a1187')
    ax.set_ylabel('Number of Messages')
    ax.set_xlabel('Month')
    ax.set_title('Messages per Month in 2025', fontsize=16)
    plt.xticks(rotation=45)
    st.pyplot(fig)

EMOJI_REGEX = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF]"
)

@st.cache_data(show_spinner=False)
def compute_emoji_stats(df):
    all_emojis = []
    sender_counts = Counter()

    for msg, sender in zip(df["message"], df["sender"]):
        if not msg or not sender:
            continue

        # Extract *characters*, not grouped runs
        chars = [c for c in msg if EMOJI_REGEX.match(c)]
        if not chars:
            continue

        all_emojis.extend(chars)
        sender_counts[sender] += len(chars)

    return Counter(all_emojis), sender_counts


def emoji_stats_slide():
    df = st.session_state.chat_df_2025.copy()

    top_emojis, sender_emoji_counts = compute_emoji_stats(df)

    # --- Top 5 Emojis ---
    top5 = top_emojis.most_common(5)
    emoji_text = "<br>".join(
        f"{i+1}. {e} — {c} times"
        for i, (e, c) in enumerate(top5)
    ) if top5 else "No emojis found 😔"

    slide("😎 Emoji Moments", "Most loved expressions in the group this year")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown("### ⭐ Top 5 Emojis")
    st.markdown(f"<pre>{emoji_text}</pre>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # --- Top Emoji Senders (list) ---
    top_senders = sender_emoji_counts.most_common()
    if not top_senders:
        st.write("No emoji senders found 😅")
        return

    st.markdown("### Emoji Count by Member")

    rank_text = "<br>".join(
        f"{i+1}. {n} — {c} emojis"
        for i, (n, c) in enumerate(top_senders)
    )
    st.markdown(f"<pre>{rank_text}</pre>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # --- Graph: ALL members' emoji usage ---
    st.markdown("### 📊 Emoji Usage by Members")

    all_counts = pd.Series(sender_emoji_counts).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(3, len(all_counts) * 0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(all_counts.index, all_counts.values, color='#8a1187')
    ax.set_xlabel("Number of Emojis Sent")
    ax.set_title("Emoji Usage Across Members")
    st.pyplot(fig)

def favorite_emoji_per_user_slide():
    df = st.session_state.chat_df_2025.copy()

    # sender → Counter of emoji usage
    sender_emoji_map = defaultdict(Counter)

    for msg, sender in zip(df["message"], df["sender"]):
        if not msg or not sender:
            continue

        # extract emojis quickly
        for ch in msg:
            if EMOJI_REGEX.match(ch):
                sender_emoji_map[sender][ch] += 1

    favorites = []

    for sender, counter in sender_emoji_map.items():
        if not counter:
            continue
        fav_emoji, count = counter.most_common(1)[0]
        favorites.append((sender, fav_emoji, count))

    # Sort by count (optional, looks nicer)
    favorites.sort(key=lambda x: x[2], reverse=True)

    slide("💖 Everyone’s Favorite Emoji",
          "The emoji each member used more than any other this year")

    if not favorites:
        st.write("No emojis detected in the chat 😅")
        return

    rows = [
        f"{i+1}. {name} — {emoji}  ({count} times)"
        for i, (name, emoji, count) in enumerate(favorites)
    ]
    
    st.write("")
    st.write("")
    st.markdown(f"<pre>{'<br>'.join(rows)}</pre>", unsafe_allow_html=True)


LINK_REGEX = re.compile(r'https?://|www\.')

def media_link_senders_slide():
    df = st.session_state.chat_df_2025.copy()

    media_counts = {}
    link_counts = {}

    for msg, sender in zip(df["message"], df["sender"]):
        if not msg or not sender:
            continue

        # Count <Media omitted>
        if "<Media omitted" in msg:
            media_counts[sender] = media_counts.get(sender, 0) + 1

        # Count links
        if LINK_REGEX.search(msg):
            link_counts[sender] = link_counts.get(sender, 0) + 1

    # Convert to pandas Series for consistent plotting
    media_series = pd.Series(media_counts).sort_values(ascending=False)
    link_series  = pd.Series(link_counts).sort_values(ascending=False)

    top_media = media_series
    top_links = link_series

    slide("📸 Media & 🔗 Link Legends",
          "Who dropped the most media and links this year?")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ---------- LEFT: MEDIA ----------
    with col1:
        st.markdown("### 📸 Top Media Senders")

        if media_series.empty:
            st.write("No media messages found 😅")
        else:
            # Full graph (all members)
            ordered = media_series.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, max(3, len(ordered)*0.35)))
            fig.patch.set_facecolor("#ca70ad")
            ax.barh(ordered.index, ordered.values, color='#8a1187')
            ax.set_xlabel("Media Messages")
            ax.set_title("Media Messages by Member")
            st.pyplot(fig); plt.close(fig)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Top-5 text list
            text = "<br>".join(
                f"{i+1}. {n} — {c} media"
                for i,(n,c) in enumerate(top_media.items())
            )
            st.markdown(f"<pre>{text}</pre>", unsafe_allow_html=True)

    # ---------- RIGHT: LINKS ----------
    with col2:
        st.markdown("### 🔗 Top Link Senders")

        if link_series.empty:
            st.write("No links shared 😅")
        else:
            # Full graph (all members)
            ordered = link_series.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, max(3, len(ordered)*0.35)))
            fig.patch.set_facecolor("#ca70ad")
            ax.barh(ordered.index, ordered.values, color='#8a1187')
            ax.set_xlabel("Links Shared")
            ax.set_title("Links Shared by Member")
            st.pyplot(fig); plt.close(fig)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Top-5 text list
            text = "<br>".join(
                f"{i+1}. {n} — {c} links"
                for i,(n,c) in enumerate(top_links.items())
            )
            st.markdown(f"<pre>{text}</pre>", unsafe_allow_html=True)

def busiest_hour_slide():
    df = st.session_state.chat_df_2025.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    # ---- Busiest hour ----
    hour_counts = df["hour"].value_counts().sort_index()
    peak_hour = hour_counts.idxmax()
    peak_count = hour_counts.max()

    slide(
        "⏰ Busiest Hour of the Day",
        f"The chat was most active at {peak_hour:02d}:00 with {peak_count} messages!"
    )

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # ---- Hourly bar chart ----
    fig, ax = plt.subplots(figsize=(8,3.5))
    fig.patch.set_facecolor("#ca70ad")
    ax.bar(hour_counts.index, hour_counts.values, color='#8a1187')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Messages")
    ax.set_title("Messages by Hour")
    ax.set_xticks(range(0,24))
    st.pyplot(fig)

    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

    # ---- Heatmap (weekday x hour) ----
    pivot = df.pivot_table(
        index="weekday",
        columns="hour",
        values="message",
        aggfunc="count",
        fill_value=0
    )

    # Order weekdays properly
    ordered_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(ordered_days)

    fig2, ax2 = plt.subplots(figsize=(9,4))
    im = ax2.imshow(pivot.values, aspect="auto")

    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels(pivot.index)

    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24), rotation=90)

    ax2.set_title("Activity Heatmap (Weekday vs Hour)")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Weekday")

    fig2.colorbar(im, ax=ax2)
    st.pyplot(fig2)

def busiest_weekday_slide():
    df = st.session_state.chat_df_2025.copy()
    df["weekday"] = df["timestamp"].dt.day_name()

    weekday_counts = df["weekday"].value_counts()
    busiest = weekday_counts.idxmax()
    busiest_count = weekday_counts.max()

    slide(
        "📅 Busiest Day of the Week",
        f"{busiest} was the most chatty day with {busiest_count} messages!"
    )

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # Order for nicer chart
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_counts = weekday_counts.reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#ca70ad")
    ax.bar(weekday_counts.index, weekday_counts.values)
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Messages")
    ax.set_title("Messages by Weekday")
    plt.xticks(rotation=20)
    st.pyplot(fig)

def chat_boss_award_slide():
    df = st.session_state.chat_df_2025.copy()

    counts = df["sender"].value_counts()

    # 🏆 Topper (Chat Boss)
    boss_name = counts.idxmax()
    boss_msgs = counts.max()

    top5 = counts

    slide(
        f"👑 Chat Boss Award — {boss_name}",
        f"{boss_name} ruled the chat with {boss_msgs} messages!"
    )

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # --- Graph: ALL members ---
    st.markdown("### 📊 Messages by All Members")

    fig, ax = plt.subplots(figsize=(8, max(3, len(counts) * 0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(counts.index[::-1], counts.values[::-1], color='#8a1187')
    ax.set_xlabel("Number of Messages")
    ax.set_title("Total Messages Sent")
    st.pyplot(fig)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Top 5 list ---
    st.markdown("### 🏆 Message Count by Members")

    top_list = "<br>".join(
        f"{i+1}. {name} — {count} msgs"
        for i, (name, count) in enumerate(top5.items())
    )
    st.markdown(f"<pre>{top_list}</pre>", unsafe_allow_html=True)

def silent_observer_award_slide():
    df = st.session_state.chat_df_2025.copy()

    counts = df["sender"].value_counts()

    # 🕶️ Least active = Silent Observer
    silent_name = counts.idxmin()
    silent_msgs = counts.min()

    bottom3 = counts.tail(5)[::-1]

    slide(
        f"🕶️ Silent Observer Award — {silent_name}",
        f"{silent_name} stayed low-key with just {silent_msgs} messages this year"
    )

    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

    st.markdown("### 🤫 The Quiet Squad")

    text = "<br>".join(
        f"{i+1}. {name} — {count} msgs"
        for i, (name, count) in enumerate(bottom3.items())
    )

    st.markdown(f"<pre>{text}</pre>", unsafe_allow_html=True)

def early_bird_slide():
    df = st.session_state.chat_df_2025.copy()

    # Extract hour and sender
    df['hour'] = df['timestamp'].dt.hour
    df['sender'] = df['sender']


    # Morning messages: 5am to 12pm
    morning_df = df[(df['hour'] >= 4) & (df['hour'] < 12)]

    # Count morning messages per sender
    morning_counts = morning_df['sender'].value_counts()

    # Total messages per sender
    total_counts = df['sender'].value_counts()

    # Compute ratio: morning / total
    morning_ratio = (morning_counts / total_counts).fillna(0)

    # --- Winner topper card ---
    if not morning_ratio.empty:
        winner_name = morning_ratio.idxmax()
        winner_ratio = morning_ratio.max()
        slide(
            f"🌅 Early Bird: {winner_name}",
            f"Fraction of messages sent in morning: {winner_ratio:.2f}"
        )
    else:
        slide("🌅 Early Birds", "No morning messages found this year!")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graph of all members ---
    sorted_ratio = morning_ratio.sort_values(ascending=True)
    st.markdown("### 📊 Morning Activity Ratio by Member")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_ratio)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(sorted_ratio.index, sorted_ratio.values, color='#8a1187')
    ax.set_xlabel("Fraction of Messages Sent in Morning (5am-12pm)")
    ax.set_title("Morning Message Ratio Across Members")
    st.pyplot(fig)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Top 5 Early Birds ---
    top5 = morning_ratio.sort_values(ascending=False).head(5)
    st.markdown("### 🏆 Top 5 Early Birds")
    top_list = "<br>".join(
        f"{i+1}. {name} — {ratio:.2f}" 
        for i, (name, ratio) in enumerate(top5.items())
    )
    st.markdown(f"<pre>{top_list}</pre>", unsafe_allow_html=True)

def night_owl_slide():
    df = st.session_state.chat_df_2025.copy()

    # Extract hour and sender
    df['hour'] = df['timestamp'].dt.hour
    df['sender'] = df['sender']

    # Night messages: 21:00–23:59 and 0:00–4:00
    night_df = df[(df['hour'] >= 21) | (df['hour'] < 4)]

    # Count night messages per sender
    night_counts = night_df['sender'].value_counts()

    # Total messages per sender
    total_counts = df['sender'].value_counts()

    # Compute ratio: night / total
    night_ratio = (night_counts / total_counts).fillna(0)

    # --- Winner topper card ---
    if not night_ratio.empty:
        winner_name = night_ratio.idxmax()
        winner_ratio = night_ratio.max()
        slide(
            f"🌙 Night Owl: {winner_name}",
            f"Fraction of messages sent at night: {winner_ratio:.2f}"
        )
    else:
        slide("🌙 Night Owls", "No night messages found this year!")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graph of all members ---
    sorted_ratio = night_ratio.sort_values(ascending=True)
    st.markdown("### 📊 Night Activity Ratio by Member")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_ratio)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(sorted_ratio.index, sorted_ratio.values, color='#8a1187')
    ax.set_xlabel("Fraction of Messages Sent at Night (9pm-4am)")
    ax.set_title("Night Message Ratio Across Members")
    st.pyplot(fig)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Top 5 Night Owls ---
    top5 = night_ratio.sort_values(ascending=False).head(5)
    st.markdown("### 🏆 Top 5 Night Owls")
    top_list = "<br>".join(
        f"{i+1}. {name} — {ratio:.2f}" 
        for i, (name, ratio) in enumerate(top5.items())
    )
    st.markdown(f"<pre>{top_list}</pre>", unsafe_allow_html=True)

def weekend_warrior_slide():
    df = st.session_state.chat_df_2025.copy()
    df['weekday'] = df['timestamp'].dt.day_name()
    df['sender'] = df['sender'].fillna("Unknown")

    # Weekend days
    weekend_df = df[df['weekday'].isin(["Saturday", "Sunday"])]

    # Count weekend messages per sender
    weekend_counts = weekend_df['sender'].value_counts()
    total_counts = df['sender'].value_counts()

    # Ratio: weekend / total
    weekend_ratio = (weekend_counts / total_counts).fillna(0)

    # --- Winner card ---
    if not weekend_ratio.empty:
        winner_name = weekend_ratio.idxmax()
        winner_ratio = weekend_ratio.max()
        slide(
            f"🎉 Weekend Warrior: {winner_name}",
            f"Fraction of messages sent on weekends: {winner_ratio:.2f}"
        )
    else:
        slide("🎉 Weekend Warriors", "No weekend messages found!")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graph for all members ---
    sorted_ratio = weekend_ratio.sort_values(ascending=True)
    st.markdown("### 📊 Weekend Message Ratio by Member")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_ratio)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(sorted_ratio.index, sorted_ratio.values, color='#8a1187')
    ax.set_xlabel("Fraction of Messages Sent on Weekends")
    ax.set_title("Weekend Activity Across Members")
    st.pyplot(fig)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Top 5 Weekend Warriors ---
    top5 = weekend_ratio.sort_values(ascending=False).head(5)
    st.markdown("### 🏆 Top 5 Weekend Warriors")
    top_list = "<br>".join(
        f"{i+1}. {name} — {ratio:.2f}" 
        for i, (name, ratio) in enumerate(top5.items())
    )
    st.markdown(f"<pre>{top_list}</pre>", unsafe_allow_html=True)


def weekday_distractor_slide():
    df = st.session_state.chat_df_2025.copy()
    df['weekday'] = df['timestamp'].dt.day_name()
    df['sender'] = df['sender'].fillna("Unknown")

    # Weekdays
    weekday_df = df[df['weekday'].isin(["Monday","Tuesday","Wednesday","Thursday","Friday"])]

    # Count weekday messages per sender
    weekday_counts = weekday_df['sender'].value_counts()
    total_counts = df['sender'].value_counts()

    # Ratio: weekday / total
    weekday_ratio = (weekday_counts / total_counts).fillna(0)

    # --- Winner card ---
    if not weekday_ratio.empty:
        winner_name = weekday_ratio.idxmax()
        winner_ratio = weekday_ratio.max()
        slide(
            f"🧑‍💻 Weekday Distractor: {winner_name}",
            f"Fraction of messages sent on weekdays: {winner_ratio:.2f}"
        )
    else:
        slide("🧑‍💻 Weekday Distractors", "No weekday messages found!")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graph for all members ---
    sorted_ratio = weekday_ratio.sort_values(ascending=True)
    st.markdown("### 📊 Weekday Message Ratio by Member")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_ratio)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(sorted_ratio.index, sorted_ratio.values, color='#8a1187')
    ax.set_xlabel("Fraction of Messages Sent on Weekdays")
    ax.set_title("Weekday Activity Across Members")
    st.pyplot(fig)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Top 5 Weekday Distractors ---
    top5 = weekday_ratio.sort_values(ascending=False).head(5)
    st.markdown("### 🏆 Top 5 Weekday Distractors")
    top_list = "<br>".join(
        f"{i+1}. {name} — {ratio:.2f}" 
        for i, (name, ratio) in enumerate(top5.items())
    )
    st.markdown(f"<pre>{top_list}</pre>", unsafe_allow_html=True)

def message_length_awards_slide():
    df = st.session_state.chat_df_2025.copy()
    
    # Compute message lengths per sender
    df["msg_len"] = df["message"].apply(lambda x: len(x) if x else 0)
    sender_avg_len = df.groupby("sender")["msg_len"].mean().sort_values()
    
    # One-Word Warrior (shortest)
    short_sender = sender_avg_len.idxmin()
    # Paragraph Philosopher (longest)
    long_sender = sender_avg_len.idxmax()
    
    slide("📝 Message Length Legends", f" 🧩One Word Warrior: {short_sender} <> 📜Paragraph Philosopher: {long_sender}")
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # --- Graph: all members ---
    st.markdown("### 📊 Average Message Length by Member")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sender_avg_len)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(sender_avg_len.index[::-1], sender_avg_len.values[::-1], color='#8a1187')
    ax.set_xlabel("Average Message Length (chars)")
    ax.set_title("Message Length Across Members")
    st.pyplot(fig)
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # --- Top 5 shortest and longest ---
    top_short = sender_avg_len.head(5)
    top_long = sender_avg_len.tail(5)[::-1]
    
    st.markdown("### 🏅 Top 5 Shortest Messagers (avg/message):")
    text_short = "<br>".join([f"{i+1}. {name} — {length:.1f} chars" for i, (name, length) in enumerate(top_short.items())])
    st.markdown(f"<pre>{text_short}</pre>", unsafe_allow_html=True)

    st.markdown("### 🏆 Top 5 Longest Messagers (avg/message):")
    text_long = "<br>".join([f"{i+1}. {name} — {length:.1f} chars" for i, (name, length) in enumerate(top_long.items(), start=0)])
    st.markdown(f"<pre>{text_long}</pre>", unsafe_allow_html=True)

def spam_lord_slide():
    df = st.session_state.chat_df_2025.copy()
    df = df[df['sender'].notna()]
    df = df.sort_values('timestamp')

    sender_gaps = {}

    for sender, group in df.groupby('sender'):
        times = group['timestamp'].sort_values().reset_index(drop=True)
        if len(times) < 2:
            continue

        # Correct gap calculation
        gaps = times.diff().dt.total_seconds().dropna()
        # Only consider "spamming bursts" gaps ≤ 1 hour
        active_gaps = gaps[gaps <= 300]
        if len(active_gaps) > 0:
            sender_gaps[sender] = active_gaps.mean()  # avg gap in seconds

    if not sender_gaps:
        slide("🧨 Spam Lord", "No spamming activity detected 😅")
        return

    # Rank senders by smallest average gap
    ranked = sorted(sender_gaps.items(), key=lambda x: x[1])
    top5 = ranked[:5]

    slide("🧨 Spam Lord", f"{top5[0][0]} bombarded the chat with rapid messages!")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graph: all members ---
    all_senders = pd.Series(sender_gaps).sort_values()[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(all_senders) * 0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(all_senders.index, all_senders.values, color='#8a1187')
    ax.set_xlabel("Average Gap Between Burst Messages (s)")
    ax.set_title("Spam Activity Across Members")
    st.pyplot(fig)

    # --- List ---
    st.markdown("### 🏅 Top 5 Spam Lords - avg interval in 5 minute bursts")
    text = "<br>".join(
        f"{i+1}. {n} — : {int(c)}s"
        for i, (n, c) in enumerate(top5)
    )
    st.markdown(f"<pre>{text}</pre>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
def emoji_awards_slide():
    df = st.session_state.chat_df_2025.copy()

    emoji_counts = Counter()
    msg_counts = Counter()

    # --- Single pass through messages ---
    for msg, sender in zip(df["message"], df["sender"]):
        if not msg or not sender:
            continue
        chars = [c for c in msg if EMOJI_REGEX.match(c)]
        emoji_counts[sender] += len(chars)
        msg_counts[sender] += 1

    # --- Compute average emojis per message ---
    avg_emoji = {s: emoji_counts[s]/msg_counts[s] for s in emoji_counts if msg_counts[s] > 0}

    if not avg_emoji:
        slide("😅 Emoji Awards", "No emojis found in the chat this year!")
        return

    # --- Identify winners ---
    emperor_name, emperor_avg = max(avg_emoji.items(), key=lambda x: x[1])
    minimalist_name, minimalist_avg = min(avg_emoji.items(), key=lambda x: x[1])

    # --- Slide header with toppers ---
    slide("🎭 Emoji Awards", f"Emoji Emperor: {emperor_name} 👑 | Emotional Minimalist: {minimalist_name} 🕊️")

    # --- Top 5 Emoji Emperor ---
    top_emperors = sorted(avg_emoji.items(), key=lambda x: x[1], reverse=True)[:5]
    list_text_emperor = "<br>".join(
        f"{i+1}. {name} — {avg:.2f} emojis/message"
        for i, (name, avg) in enumerate(top_emperors)
    )
    st.markdown("### 👑 Top 5 Emoji Emperors")
    st.markdown(f"<pre>{list_text_emperor}</pre>", unsafe_allow_html=True)

    # --- Top 5 Emotional Minimalist ---
    top_minimalists = sorted(avg_emoji.items(), key=lambda x: x[1])[:5]
    list_text_minimalist = "<br>".join(
        f"{i+1}. {name} — {avg:.2f} emojis/message"
        for i, (name, avg) in enumerate(top_minimalists)
    )
    st.markdown("### 🕊️ Top 5 Emotional Minimalists")
    st.markdown(f"<pre>{list_text_minimalist}</pre>", unsafe_allow_html=True)

    # --- Graph for all members ---
    all_series = pd.Series(avg_emoji).sort_values()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Average Emojis per Message Across Members")

    fig, ax = plt.subplots(figsize=(8, max(3, len(all_series)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(all_series.index, all_series.values, color='#8a1187')
    ax.set_xlabel("Average Emojis per Message")
    ax.set_title("Emoji Emperor & Emotional Minimalist Rankings")
    st.pyplot(fig)

def media_per_message_slide():
    df = st.session_state.chat_df_2025.copy()
    df = df[df['sender'].notna() & (df['sender'] != "Meta AI")]

    # Total messages per sender
    total_counts = df['sender'].value_counts()

    # Media messages per sender
    media_counts = df[df['message'].str.contains("<Media omitted", na=False)].groupby('sender').size()

    # Ratio: media / total messages
    media_ratio = (media_counts / total_counts).fillna(0)

    if not media_ratio.empty:
        topper = media_ratio.idxmax()
        ratio = media_ratio.max()
        slide(
            f"📸 Media Maven: {topper}",
            f"{topper} knows how to keep the chat 📷-tastic with {ratio:.2f} media per message!"
        )
    else:
        slide("📸 Media Maven", "No media messages found this year!")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graph all members ---
    sorted_ratio = media_ratio.sort_values(ascending=True)
    st.markdown("### 📊 Media Messages Per Message Ratio")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_ratio)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(sorted_ratio.index, sorted_ratio.values, color='#8a1187')
    ax.set_xlabel("Media Messages per Message")
    ax.set_title("Media Maven Rankings")
    st.pyplot(fig)

    # --- Top 5 list ---
    top5 = media_ratio.sort_values(ascending=False).head(5)
    st.markdown("### 🏆 Top 5 Media Mavens - Media Items/Message")
    text = "<br>".join(f"{i+1}. {name} — {ratio:.2f}" for i, (name, ratio) in enumerate(top5.items()))
    st.markdown(f"<pre>{text}</pre>", unsafe_allow_html=True)


def links_per_message_slide():
    df = st.session_state.chat_df_2025.copy()
    df = df[df['sender'].notna() & (df['sender'] != "Meta AI")]

    # Total messages per sender
    total_counts = df['sender'].value_counts()

    # Link messages per sender
    LINK_REGEX = re.compile(r'https?://|www\.')
    link_counts = df[df['message'].str.contains(LINK_REGEX, na=False)].groupby('sender').size()

    # Ratio: links / total messages
    link_ratio = (link_counts / total_counts).fillna(0)

    if not link_ratio.empty:
        topper = link_ratio.idxmax()
        ratio = link_ratio.max()
        slide(
            f"🔗 Link Legend: {topper}",
            f"{topper} keeps everyone connected with {ratio:.2f} links per message!"
        )
    else:
        slide("🔗 Link Legend", "No links shared this year!")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graph all members ---
    sorted_ratio = link_ratio.sort_values(ascending=True)
    st.markdown("### 📊 Links Per Message Ratio")
    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_ratio)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(sorted_ratio.index, sorted_ratio.values, color='#8a1187')
    ax.set_xlabel("Links per Message")
    ax.set_title("Link Legend Rankings")
    st.pyplot(fig)

    # --- Top 5 list ---
    top5 = link_ratio.sort_values(ascending=False).head(5)
    st.markdown("### 🏆 Top 5 Link Legends - Links/Message")
    text = "<br>".join(f"{i+1}. {name} — {ratio:.2f}" for i, (name, ratio) in enumerate(top5.items()))
    st.markdown(f"<pre>{text}</pre>", unsafe_allow_html=True)

def tag_sniper_slide():
    df = st.session_state.chat_df_2025.copy()
    df["message"] = df["message"].fillna("")
    df = df[df["sender"].notna()]

    START = "\u2068"
    END = "\u2069"

    pair_counts = Counter()      # (tagger → tagged)
    sniper_targets = {}          # best target per sender

    for sender, msg in zip(df["sender"], df["message"]):
        start = 0
        while True:
            s = msg.find(START, start)
            if s == -1:
                break
            e = msg.find(END, s+1)
            if e == -1:
                break

            mentioned = msg[s+1:e].strip()
            pair_counts[(sender, mentioned)] += 1
            start = e + 1

    if not pair_counts:
        slide("🏹 Tag Sniper", "No tag activity detected in the chat 😅")
        return

    # --- compute top target per sender ---
    by_sender = defaultdict(Counter)
    for (a, b), c in pair_counts.items():
            by_sender[a][b] += c

    for sender, targets in by_sender.items():
        tgt, cnt = targets.most_common(1)[0]
        sniper_targets[sender] = (tgt, cnt)

    # --- topper ---
    topper = max(sniper_targets.items(), key=lambda x: x[1][1])
    topper_name, (top_target, top_count) = topper

    slide(
        f"🏹 Tag Sniper — {topper_name}",
        f"{topper_name} tags {top_target} the most ({top_count} tags)"
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- full list ---
    st.markdown("### 🏹 Who Everyone Tags the Most")

    list_text = "<br>".join(
        f"{i+1}. {s} ➜ {t} — {c} tags"
        for i, (s, (t, c)) in enumerate(
            sorted(sniper_targets.items(), key=lambda x: x[1][1], reverse=True)
        )
    )
    st.markdown(f"<pre>{list_text}</pre>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- graph ---
    names = [f"{s} → {t}" for s, (t, _) in sniper_targets.items()]
    values = [c for _, (_, c) in sniper_targets.items()]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(names[::-1], values[::-1], color='#8a1187')
    ax.set_xlabel("Number of Tags")
    ax.set_title("Top Tag Targets per Member")
    st.pyplot(fig)

def tag_magnet_slide():
    df = st.session_state.chat_df_2025.copy()
    df["message"] = df["message"].fillna("")
    df = df[df["sender"].notna()]

    real_members = set(df["sender"].unique()) - {"Meta AI"}

    START = "\u2068"
    END = "\u2069"

    pair_counts = Counter()   # (tagger -> target)

    # ---- extract tag pairs ----
    for sender, msg in zip(df["sender"], df["message"]):
        start = 0
        while True:
            s = msg.find(START, start)
            if s == -1:
                break
            e = msg.find(END, s+1)
            if e == -1:
                break

            mentioned = msg[s+1:e].strip()

            if mentioned in real_members:
                pair_counts[(sender, mentioned)] += 1

            start = e + 1

    if not pair_counts:
        slide("🧲 Tag Magnet", "No tag activity detected 😅")
        return

    # ---- build incoming tag counts per TARGET ----
    incoming = {m: Counter() for m in real_members}

    for (tagger, target), c in pair_counts.items():
        if target in incoming:
            incoming[target][tagger] += c

    # ---- for each member, pick who tags them the most ----
    magnet_pairs = {}
    for member, srcs in incoming.items():
        if srcs:
            top_tagger, count = srcs.most_common(1)[0]
            magnet_pairs[member] = (top_tagger, count)

    # ---- topper ----
    topper_member, (top_src, top_cnt) = max(
        magnet_pairs.items(), key=lambda x: x[1][1]
    )

    slide(
        f"🧲 Tag Magnet — {topper_member}",
        f"{topper_member} is tagged most by {top_src}"
        f" ({top_cnt} tags)"
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ---- clean list (ONE ROW PER MEMBER) ----
    st.markdown("### 🧲 Who Tags Each Member the Most")

    list_text = "<br>".join(
        f"{i+1}. {member} ⟵ {tagger} — {count} tags"
        for i, (member, (tagger, count)) in enumerate(
            sorted(magnet_pairs.items(), key=lambda x: x[1][1], reverse=True)
        )
    )
    st.markdown(f"<pre>{list_text}</pre>", unsafe_allow_html=True)

    # ---- graph ----
    names = [f"{tagger} → {member}" for member, (tagger, _) in magnet_pairs.items()]
    values = [count for _, (_, count) in magnet_pairs.items()]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names)*0.35)))
    fig.patch.set_facecolor("#ca70ad")
    ax.barh(names[::-1], values[::-1], color='#8a1187')
    ax.set_xlabel("Number of Tags")
    ax.set_title("Top Incoming Tagger per Member")
    st.pyplot(fig)

if "chat_df_2025" in st.session_state:
    add_slide(total_messages_slide, "Total Messages")
    add_slide(monthly_messages_slide, "Monthly Messages")
    add_slide(emoji_stats_slide, "Emoji Stats")
    add_slide(favorite_emoji_per_user_slide, "Favorite Emoji Per User")
    add_slide(media_link_senders_slide, "Media & Link Senders")
    add_slide(busiest_hour_slide, "Busiest Hour")
    add_slide(busiest_weekday_slide, "Busiest Weekday")
    add_slide(chat_boss_award_slide, "Chat Boss Award")
    add_slide(silent_observer_award_slide, "Silent Observer Award")
    add_slide(early_bird_slide, "Early Bird")
    add_slide(night_owl_slide, "Night Owl")
    add_slide(weekend_warrior_slide, "Weekend Warrior")
    add_slide(weekday_distractor_slide, "Weekday Distractor")
    add_slide(message_length_awards_slide, "Message Length Awards")
    add_slide(spam_lord_slide, "Spam Lord")
    add_slide(emoji_awards_slide, "Emoji Awards")
    add_slide(media_per_message_slide, "Media per Message")
    add_slide(links_per_message_slide, "Links per Message")
    add_slide(tag_sniper_slide, "Tag Sniper")
    add_slide(tag_magnet_slide, "Tag Magnet")

# Clamp slide index
st.session_state.slide = min(st.session_state.slide, len(slides)-1)

# Render current slide

progress = (st.session_state.slide + 1) / len(slides)  # 1-indexed for nicer feel
st.progress(progress)
slides[st.session_state.slide]()

# --- Sidebar Navigation ---
# 1️⃣ Sidebar navigator
st.sidebar.title("📑 Slide Navigator")

selected = st.sidebar.radio(
    "Jump to slide",
    range(len(slides)),
    format_func=lambda i: slide_names[i],
    index=st.session_state.slide  # show current slide
)

# Only update session state if changed
if selected != st.session_state.slide:
    st.session_state.slide = selected
    st.rerun()  # optional, ensures page reloads immediately

# 2️⃣ Back / Next buttons
col1, col2 = st.columns([8,1])
with col1:
    st.write("")
    if st.session_state.slide > 0 and st.button("← Back"):
        st.session_state.slide -= 1
        st.experimental_rerun()
with col2:
    st.write("")
    if st.session_state.slide < len(slides)-1 and st.button("Next →"):
        st.session_state.slide += 1
        st.rerun()
