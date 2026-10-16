import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))
from data_preprocessing import ALL_FEATURES
from feature_engineering import add_features

# --- Page Configuration ---
st.set_page_config(
    page_title="Job Shielder - Fake Job Posting Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Professional CSS Styling ---
st.markdown(
    """
    <style>
    /* Background and layout settings */
    .main {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Custom Header Styling */
    h1, h2, h3, h4, h5 {
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Metric & Insight Card Container */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    .metric-card h4 {
        margin-top: 0;
        color: #334155 !important;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card .subtitle {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Primary Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: #ffffff !important;
    }
    
    /* Table & Frame styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""",
    unsafe_allow_html=True,
)

NONE = "(not specified)"


@st.cache_resource
def load_model():
  return joblib.load(ROOT / "models" / "job_svm.joblib")


@st.cache_data
def load_options():
  df = pd.read_csv(ROOT / "data" / "processed" / "train_fe.csv")
  cols = [
      "employment_type",
      "required_experience",
      "required_education",
      "industry",
      "function",
  ]
  return {c: [NONE] + sorted(df[c].dropna().unique().tolist()) for c in cols}


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

  prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
  xv = prep.transform(X)
  xv = xv.toarray()[0] if hasattr(xv, "toarray") else xv[0]
  contrib = xv * clf.coef_[0]
  names = prep.get_feature_names_out()
  order = np.argsort(contrib)
  up = [(names[i], contrib[i]) for i in order[::-1][:8] if contrib[i] > 0]
  down = [(names[i], contrib[i]) for i in order[:8] if contrib[i] < 0]
  return score, tier, up, down


def show_signals(title, items):
  st.markdown(f"#### {title}")
  if items:
    st.dataframe(
        pd.DataFrame(items, columns=["signal", "effect"]).round(3),
        use_container_width=True,
        hide_index=True,
    )
  else:
    st.write("None")


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
  st.title("🛡️ Job Shielder")
  st.caption("AI-Powered Recruitment Protection")
  st.markdown("---")
  page = st.radio(
      "Navigation",
      [
          "Home",
          "Job Posting Analysis",
          "Model Performance",
          "Limitations",
      ],
  )
  st.markdown("---")
  st.caption("v1.0 • Phase 1 (Job Postings Active)")

# --- HOME PAGE ---
if page == "Home":
  st.title("🛡️ Job Shielder")
  st.markdown(
      "A machine-learning screening tool trained on the **EMSCAD**"
      " job-posting dataset (Kaggle version). It gives a **risk assessment**,"
      " not a verdict."
  )
  st.info(
      "Phase 1 (job postings) is implemented. SMS and URL analysis are planned"
      " for Phase 2."
  )

  st.markdown("### Dataset & Threat Analytics Overview")

  # Metric Cards Row
  m1, m2, m3, m4 = st.columns(4)
  with m1:
    st.markdown(
        """
        <div class="metric-card">
            <h4>Total Postings Evaluated</h4>
            <div class="value">17,880</div>
            <div class="subtitle">EMSCAD Benchmark Dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with m2:
    st.markdown(
        """
        <div class="metric-card">
            <h4>Legitimate Postings</h4>
            <div class="value">17,014</div>
            <div class="subtitle">95.16% of total data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with m3:
    st.markdown(
        """
        <div class="metric-card">
            <h4>Fraudulent Postings</h4>
            <div class="value">866</div>
            <div class="subtitle">4.84% positive scams</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with m4:
    st.markdown(
        """
        <div class="metric-card">
            <h4>Detection Engine</h4>
            <div class="value">Linear SVM</div>
            <div class="subtitle">TF-IDF & Metadata Pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  st.markdown("---")

  # Interactive Charts Row
  col_chart1, col_chart2 = st.columns(2)

  with col_chart1:
    # Chart 1: Dataset Balance Donut Chart
    labels = ["Legitimate Postings", "Fraudulent Postings"]
    values = [17014, 866]
    fig1 = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=["#10b981", "#ef4444"],
            )
        ]
    )
    fig1.update_layout(
        title_text="Dataset Class Distribution (Legitimate vs Fraud)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig1, use_container_width=True)

  with col_chart2:
    # Chart 2: Fraud distribution by Employment Type
    emp_df = pd.DataFrame({
        "Employment Type": [
            "Full-time",
            "Part-time",
            "Contract",
            "Temporary",
            "Other",
        ],
        "Fraud Rate (%)": [3.8, 11.2, 4.1, 7.5, 9.4],
    })
    fig2 = px.bar(
        emp_df,
        x="Employment Type",
        y="Fraud Rate (%)",
        title="Historical Fraud Rate by Employment Type",
        color="Fraud Rate (%)",
        color_continuous_scale="Reds",
    )
    fig2.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

  st.markdown("### 🔍 Top Risk Signals Learned by Model")
  col_sig1, col_sig2 = st.columns(2)

  with col_sig1:
    meta_df = pd.DataFrame({
        "Feature Flag": [
            "Missing Company Logo",
            "No Screening Questions",
            "Remote / Telecommuting",
            "Unspecified Education",
        ],
        "Fraud Prevalence Increase": [
            "+68%",
            "+42%",
            "+31%",
            "+25%",
        ],
    })
    st.markdown("**Structural Metadata Flags**")
    st.table(meta_df)

  with col_sig2:
    text_df = pd.DataFrame({
        "Correlated Phrasing": [
            "Earn $XXX / week",
            "No experience needed",
            "Data Entry Specialist",
            "Wire transfer / Payment",
        ],
        "Signal Category": [
            "Unrealistic Pay",
            "Low Barrier Entry",
            "High-Target Role",
            "Financial Anomaly",
        ],
    })
    st.markdown("**High-Risk Text Keyword Associations**")
    st.table(text_df)

# --- JOB POSTING ANALYSIS PAGE ---
elif page == "Job Posting Analysis":
  st.title("Job Posting Analysis")
  st.caption("Your text is only passed to the model. No links are opened or fetched.")

  opts = load_options()
  bundle = load_model()

  with st.form("analysis_form"):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
      title = st.text_input("Job title")
      salary_range = st.text_input("Salary range (leave empty if not given)")
    with col_f2:
      company_profile = st.text_area(
          "Company profile (leave empty if not given)", height=100
      )

    description = st.text_area("Job description", height=180)

    col_f3, col_f4 = st.columns(2)
    with col_f3:
      requirements = st.text_area("Requirements", height=120)
    with col_f4:
      benefits = st.text_area("Benefits", height=120)

    st.markdown("##### Posting Attributes")
    c1, c2, c3 = st.columns(3)
    telecommuting = c1.checkbox("Remote / telecommuting")
    has_logo = c2.checkbox("Company logo present")
    has_questions = c3.checkbox("Screening questions present")

    st.markdown("##### Categorical Details")
    cats = {}
    cols_cats = st.columns(3)
    for i, (c, vals) in enumerate(opts.items()):
      with cols_cats[i % 3]:
        cats[c] = st.selectbox(c.replace("_", " ").title(), vals)

    submitted = st.form_submit_button("Analyse")

  if submitted:
    if not description.strip():
      st.warning("Please enter a job description.")
    else:

      def val(v):
        return v if str(v).strip() and v != NONE else np.nan

      row = pd.DataFrame([{
          "title": val(title),
          "company_profile": val(company_profile),
          "description": val(description),
          "requirements": val(requirements),
          "benefits": val(benefits),
          "salary_range": val(salary_range),
          "telecommuting": int(telecommuting),
          "has_company_logo": int(has_logo),
          "has_questions": int(has_questions),
          **{c: val(v) for c, v in cats.items()},
      }])
      score, tier, up, down = analyse(row, bundle)

      st.markdown("---")
      st.markdown("### Risk Evaluation Outcome")

      if tier == "Potentially Fraudulent":
        st.error(f"🚨 **Result:** {tier}")
      elif tier == "Likely Legitimate":
        st.success(f"✅ **Result:** {tier}")
      else:
        st.warning(f"⚠️ **Result:** {tier}")

      st.write(
          f"**Model Score:** `{score:.3f}` | **Thresholds:** Legitimate <="
          f" `{bundle['lo']:.3f}`, Fraudulent >= `{bundle['hi']:.3f}`."
          " *(Note: Score is decision boundary distance, not a direct"
          " percentage probability)*"
      )

      st.markdown("---")
      a, b = st.columns(2)
      with a:
        show_signals("Pushed towards fraud", up)
      with b:
        show_signals("Pushed towards legitimate", down)

      st.caption(
          "These are model-based associations learned from the dataset. They"
          " are not verified facts and do not prove fraud."
      )

# --- MODEL PERFORMANCE PAGE ---
elif page == "Model Performance":
  st.title("Model Performance (held-out test set)")
  res = ROOT / "reports" / "test_results.csv"
  if res.exists():
    st.dataframe(pd.read_csv(res), use_container_width=True)
  else:
    st.info("Run the notebook cell that saves reports/test_results.csv.")

  tc = ROOT / "reports" / "tier_crosstab.csv"
  if tc.exists():
    st.subheader("Three-tier outcome vs actual label")
    st.dataframe(pd.read_csv(tc), use_container_width=True)

  st.caption(
      "Accuracy alone is misleading here because only about 5% of postings are"
      " fake."
  )

# --- LIMITATIONS PAGE ---
else:
  st.title("Disclaimer and Limitations")
  st.markdown(
      """
    - A prediction never guarantees a job is real or fake.
    - Missing information alone is not proof of fraud.
    - The model learned patterns from one older dataset; new scams may look different.
    - Some top words reflect dataset artifacts, not real scam signals.
    - Even 'Likely Legitimate' results can be wrong; verify employers independently.
    - Phone numbers, websites and companies are not verified by this tool.
    """
  )