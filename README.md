# chat-wrapped

# WhatsApp Data Analytics Engine (2025)

A Python-based web application for the localized analysis of WhatsApp chat exports. This tool parses unstructured text data to generate behavioral insights and communication metrics.

## Core Technologies
- **Pandas**: High-performance data manipulation and filtering.
- **Streamlit**: Web interface and reactive state management.
- **Python-Datetime**: Time-series normalization and period-specific filtering.

## Technical Features
- **Regex Parsing**: Custom-built regular expression engine to handle diverse OS-level timestamp variations (iOS/Android).
- **Stateless Architecture**: Engineered for privacy; data is processed in-memory and cleared upon session termination.
- **Error Handling**: Implemented global exception handling to manage malformed text inputs and unsupported file encodings.
- **KPI Generation**: Computation of 25+ distinct communication metrics, including engagement frequency and sentiment indicators (via emoji distribution).

## Privacy & Compliance
To ensure total data isolation, this application:
- Does not utilize persistent storage (Databases/Logs).
- Performs all computations on a temporary local instance.
- Requires no authentication, preventing any link between users and their data.

## Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install pandas streamlit`.
3. Run the app: `streamlit run app.py`.
4. Upload a `.txt` export from WhatsApp (without media).
   OR
Visit https://chat-wrapped.streamlit.app
