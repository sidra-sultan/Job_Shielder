import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))
from data_preprocessing import ALL_FEATURES
from feature_engineering import add_features

# --- Page Configuration ---
st.set_page_config(
    page_title="Job Shielder | Enterprise Recruitment Verification",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Enterprise-Grade CSS Styling (Clean, Modern, No Emojis) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
    }
    .metric-card .title {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card .subtitle {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 2px;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        color: #0f172a;
        padding: 10px 14px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }
    .badge-fraud {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    .badge-legit {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    .badge-uncertain {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

NONE = "(not specified)"


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "job_svm.joblib")


def scrape_job_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Failed to load page (Status code: {response.status_code})"
        
        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text().strip() if title_tag else "Extracted Job Posting"
        
        paragraphs = soup.find_all("p")
        description = "\n".join([p.get_text() for p in paragraphs]) if paragraphs else soup.get_text()
        
        return {
            "title": title,
            "company_profile": NONE,
            "description": description,
            "requirements": NONE,
            "benefits": NONE,
            "salary_range": NONE,
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": NONE,
            "required_experience": NONE,
            "required_education": NONE,
            "industry": NONE,
            "function": NONE,
        }, None
    except Exception as e:
        return None, str(e)


def analyse(row, bundle):
    pipe = bundle["pipeline"]
    X = add_features(row)[ALL_FEATURES]
    score = float(pipe.decision_function(X)[0])
    if score >= bundle["hi"]:
        tier = "Potentially Fraudulent"
    elif score <= bundle["lo"]:
        tier = "Likely Legitimate"
    else:
        tier = "Uncertain / Requires Manual Verification"
    return score, tier


if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

bundle = load_model()

# --- Top Navigation / Header Bar ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("## Job Shielder")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.95rem; margin-top: -8px;'>Enterprise Recruitment Verification & Fraud Analytics Platform</p>",
        unsafe_allow_html=True,
    )
with col_head2:
    st.markdown(
        "<div style='text-align: right; padding-top: 5px;'><span style='background-color: #e2e8f0; color: #334155; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;'>System Active</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- Enterprise Metrics Dashboard Row ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="title">Dataset Corpus</div>
            <div class="value">17,880</div>
            <div class="subtitle">EMSCAD benchmark records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="title">Legitimate Baseline</div>
            <div class="value">17,014</div>
            <div class="subtitle">95.16% distribution ratio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="title">Fraud Baseline</div>
            <div class="value">866</div>
            <div class="subtitle">4.84% anomaly markers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="title">Classification Engine</div>
            <div class="value">Linear SVM</div>
            <div class="subtitle">TF-IDF metadata pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

# --- Professional Tabs Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "URL Assessment", 
    "Manual Text Entry", 
    "Scan History Audit", 
    "Threat Reference"
])

with tab1:
    st.markdown("#### URL Content Extraction & Evaluation")
    st.markdown("Provide a live posting URL to automatically parse text features and evaluate security risk.")
    
    url_input = st.text_input(
        "Target Job URL",
        placeholder="https://example.com/job-posting",
        key="url_field"
    )
    
    col_btn1, col_btn2 = st.columns([2, 3])
    with col_btn1:
        run_url_analysis = st.button("Execute Verification Scan")

    if run_url_analysis:
        if not url_input.strip():
            st.warning("Please provide a valid URL endpoint.")
        else:
            with st.spinner("Extracting content payload and evaluating vector attributes..."):
                scraped_data, err = scrape_job_content(url_input.strip())
                
                if err or not scraped_data:
                    st.error(f"Extraction failed: {err}")
                    st.info(
                        "Note: Platforms such as Indeed and LinkedIn enforce automated bot restrictions. "
                        "If a URL is blocked, please utilize the 'Manual Text Entry' tab."
                    )
                else:
                    row = pd.DataFrame([scraped_data])
                    score, tier = analyse(row, bundle)

                    st.session_state.scan_history.insert(0, {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Channel": "URL Extraction",
                        "Target": scraped_data["title"][:35],
                        "Outcome": tier,
                        "Decision Score": round(score, 3)
                    })

                    st.markdown("---")
                    st.markdown("#### Assessment Outcome Summary")

                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        if tier == "Potentially Fraudulent":
                            st.markdown(f"<div class='badge-fraud'>Status: {tier}</div>", unsafe_allow_html=True)
                        elif tier == "Likely Legitimate":
                            st.markdown(f"<div class='badge-legit'>Status: {tier}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='badge-uncertain'>Status: {tier}</div>", unsafe_allow_html=True)

                        st.markdown(f"<p style='margin-top: 15px; font-size: 0.9rem; color: #334155;'>Decision Boundary Confidence Score: <b>{score:.3f}</b></p>", unsafe_allow_html=True)

                    with res_col2:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence Metric", 'font': {'size': 13}},
                            gauge = {
                                'axis': {'range': [bundle['lo'] - 2, bundle['hi'] + 2]},
                                'bar': {'color': "#2563eb"},
                                'steps': [
                                    {'range': [bundle['lo'] - 2, bundle['lo']], 'color': "#ecfdf5"},
                                    {'range': [bundle['lo'], bundle['hi']], 'color': "#fffbeb"},
                                    {'range': [bundle['hi'], bundle['hi'] + 2], 'color': "#fef2f2"}
                                ],
                                'threshold': {
                                    'line': {'color': "#b91c1c", 'width': 3},
                                    'thickness': 0.75,
                                    'value': bundle['hi']
                                }
                            }
                        ))
                        fig.update_layout(height=200, margin=dict(l=15, r=15, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Inspect Parsed Payload Data"):
                        st.write(f"**Identified Title:** {scraped_data['title']}")
                        st.text_area("Extracted Body Content", scraped_data["description"], height=160)

with tab2:
    st.markdown("#### Direct Text Description Evaluation")
    st.markdown("Analyze raw job description text directly when platform scraping filters apply.")
    
    manual_title = st.text_input("Position Designation", placeholder="e.g. Senior Data Analyst", key="man_title")
    manual_desc = st.text_area(
        "Job Description Corpus",
        placeholder="Paste full employment description content here...",
        height=180,
        key="man_desc"
    )
    
    col_mbtn1, col_mbtn2 = st.columns([2, 3])
    with col_mbtn1:
        run_manual_analysis = st.button("Execute Text Audit")

    if run_manual_analysis:
        if not manual_desc.strip():
            st.warning("Please supply valid description text.")
        else:
            manual_data = {
                "title": manual_title if manual_title.strip() else "Direct Audit Position",
                "company_profile": NONE,
                "description": manual_desc,
                "requirements": NONE,
                "benefits": NONE,
                "salary_range": NONE,
                "telecommuting": 0,
                "has_company_logo": 1,
                "has_questions": 1,
                "employment_type": NONE,
                "required_experience": NONE,
                "required_education": NONE,
                "industry": NONE,
                "function": NONE,
            }
            row = pd.DataFrame([manual_data])
            score, tier = analyse(row, bundle)

            st.session_state.scan_history.insert(0, {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Channel": "Manual Audit",
                "Target": manual_data["title"][:35],
                "Outcome": tier,
                "Decision Score": round(score, 3)
            })

            st.markdown("---")
            st.markdown("#### Assessment Outcome Summary")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if tier == "Potentially Fraudulent":
                    st.markdown(f"<div class='badge-fraud'>Status: {tier}</div>", unsafe_allow_html=True)
                elif tier == "Likely Legitimate":
                    st.markdown(f"<div class='badge-legit'>Status: {tier}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='badge-uncertain'>Status: {tier}</div>", unsafe_allow_html=True)

                st.markdown(f"<p style='margin-top: 15px; font-size: 0.9rem; color: #334155;'>Decision Boundary Confidence Score: <b>{score:.3f}</b></p>", unsafe_allow_html=True)

            with res_col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Metric", 'font': {'size': 13}},
                    gauge = {
                        'axis': {'range': [bundle['lo'] - 2, bundle['hi'] + 2]},
                        'bar': {'color': "#2563eb"},
                        'steps': [
                            {'range': [bundle['lo'] - 2, bundle['lo']], 'color': "#ecfdf5"},
                            {'range': [bundle['lo'], bundle['hi']], 'color': "#fffbeb"},
                            {'range': [bundle['hi'], bundle['hi'] + 2], 'color': "#fef2f2"}
                        ],
                        'threshold': {
                            'line': {'color': "#b91c1c", 'width': 3},
                            'thickness': 0.75,
                            'value': bundle['hi']
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=15, r=15, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("#### Scan History Audit Log")
    st.markdown("Active session ledger tracking all evaluated employment postings.")
    
    if st.session_state.scan_history:
        history_df = pd.DataFrame(st.session_state.scan_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        if st.button("Purge Session Audit Log"):
            st.session_state.scan_history = []
            st.rerun()
    else:
        st.info("No prior verification sessions recorded during the active run.")

with tab4:
    st.markdown("#### Enterprise Threat Intelligence & Red Flags")
    st.markdown("Core behavioural indicators correlated with recruitment fraud based on historical benchmark modeling.")
    
    tf_col1, tf_col2 = st.columns(2)
    with tf_col1:
        st.markdown("**Structural & Metadata Indicators**")
        st.markdown("- **Omission of Corporate Branding:** Absence of verified logos or organizational markers.")
        st.markdown("- **Lack of Qualification Filters:** Elimination of standard multi-stage screening protocols.")
        st.markdown("- **Broad Geographic Scope:** Undefined remote parameters targeted at scaling victim reach.")
    with tf_col2:
        st.markdown("**Lexical & Financial Indicators**")
        st.markdown("- **Unverified Compensation Structures:** Outlier salary ranges promised for low-barrier tasks.")
        st.markdown("- **Early Transactional Demands:** References regarding equipment procurement fees or payment clearing.")
        st.markdown("- **Ambiguous Entity Profiles:** Generic descriptive terminology lacking formal registration details.")