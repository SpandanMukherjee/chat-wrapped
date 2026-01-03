### Chat-Wrapped

# WhatsApp Data Analytics Engine (2025)
A Python-based web application for localized analysis of WhatsApp chat exports. This tool parses unstructured text data to generate behavioral insights and group-specific communication metrics.

# 🛠 Core Technologies
1. Pandas: High-performance data manipulation and filtering.
2. Streamlit: Reactive web interface and state management.
3. Python-Datetime: Time-series normalization for period-specific activity tracking.

# 🚀 Technical Features
1. Regex Parsing: Custom regular expression engine built to handle diverse OS-level timestamp variations (Android formats).
2. Stateless Architecture: Engineered for absolute privacy; data is processed entirely in-memory and purged upon session termination.
3. Error Handling: Global exception handling to manage malformed text inputs and unsupported file encodings (UTF-8/BOM).
4. KPI Generation: Computation of 25+ metrics, including peak activity hours, response latency, and emoji distribution.

# 🧠 Development Challenges

Date Format Fragmentation: Android and iOS export dates differently (e.g., 25/12/24 vs [25/12/24]). I developed a logic-gate system to automatically detect the source OS and adjust the parser on-the-fly.

Regex Complexity: Dealing with messages that span multiple lines required complex multi-line regex patterns to ensure "split" messages weren't counted as new entries.

# 🔒 Privacy & Compliance
To ensure total data isolation, this application:

Does not utilize persistent storage (No databases, no logs).

Performs all computations on a temporary, local Streamlit instance.

Requires no authentication, ensuring no link exists between users and their uploaded data.

# 💻 Installation & Usage
1. Clone the repository.
2. Install dependencies: pip install pandas streamlit
3. Run the app: streamlit run app.py
4. Upload: Use a .txt export from WhatsApp (without media).

Live Demo: https://chat-wrapped.streamlit.app
