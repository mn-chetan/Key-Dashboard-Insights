import streamlit as st
import pandas as pd
import plotly.express as px
from google.cloud import storage
import gcsfs
from datetime import datetime
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
    FinishReason,
)
import google.oauth2.service_account

# --- CONFIGURATION & INITIALIZATION ---

st.set_page_config(
    layout="wide",
    page_title="Quality Insights Dashboard",
    initial_sidebar_state="expanded",
)

try:
    PROJECT_ID = st.secrets.gcp_service_account.project_id
    BUCKET_NAME = st.secrets.gcp_config.BUCKET_NAME
    REGION = "us-central1"

    # Define credentials using service account info from Streamlit secrets
    credentials = google.oauth2.service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )

    # Initialize Vertex AI with the explicit credentials
    vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

    # Initialize GCSFileSystem with the same explicit credentials
    fs = gcsfs.GCSFileSystem(project=PROJECT_ID, token=credentials)

except Exception as e:
    st.error(
        f"Failed to initialize GCP services. Check your Streamlit Secrets structure: {e}"
    )
    st.stop()

# --- STYLING ---
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: 1px solid #0069d9;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric * {
        color: #343a40 !important;
    }
    h1, h2, h3 {color: #343a40;}

    /* Proper radio button styling for tab-like appearance */
    .stRadio > div {
        flex-direction: row;
        gap: 0.25rem;
        align-items: center;
    }

    .stRadio > div > label {
        background-color: #e9ecef !important;
        padding: 0.75rem 1.25rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #dee2e6 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        color: #495057 !important;
        font-weight: 500 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        min-height: 2.5rem !important;
    }

    .stRadio > div > label:hover {
        background-color: #dee2e6 !important;
        color: #212529 !important;
    }

    .stRadio > div > label > div {
        color: inherit !important;
    }

    /* Style for the selected radio button */
    .stRadio > div > label[data-baseweb="radio"] input:checked + div {
        background-color: #007bff !important;
        color: white !important;
        border-color: #0056b3 !important;
        font-weight: 600 !important;
    }

    /* Hide the actual radio button circles */
    .stRadio input[type="radio"] {
        display: none !important;
    }

    /* Ensure text visibility in all states */
    .stRadio label div[data-testid="stMarkdownContainer"] p {
        color: inherit !important;
        margin: 0 !important;
    }

    /* Force text color inheritance */
    .stRadio * {
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- DATA LOADING (CACHED FOR PERFORMANCE) ---


@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_data(_fs, _bucket_name):
    """Loads and preprocesses data directly from GCS."""
    try:
        with _fs.open(f"{_bucket_name}/quality_data.csv") as f:
            df = pd.read_csv(f)
        df["Production_Date"] = pd.to_datetime(df["Production_Date"])
        df["Month"] = df["Production_Date"].dt.to_period("M").astype(str)
        return df
    except Exception as e:
        st.error(f"Failed to load data from GCS bucket '{_bucket_name}': {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache guidelines for 1 hour
def load_guidelines(_fs, _bucket_name):
    """Loads guidelines from GCS."""
    try:
        with _fs.open(f"{_bucket_name}/auto_trim_guidelines.md") as f:
            return f.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to load guidelines: {e}")
        return "Guidelines could not be loaded."


df_full = load_data(fs, BUCKET_NAME)
guidelines_content = load_guidelines(fs, BUCKET_NAME)
guidelines_excerpt = guidelines_content[:2000]

if df_full.empty:
    st.error("Dashboard cannot be displayed because the source data is empty.")
    st.stop()

# --- HELPER FUNCTIONS ---


def display_fallback_recommendations(error_rate, error_per_flag):
    """Simple fallback if the AI model fails."""
    top_error_flags = error_per_flag.head(2)["Flag_Type"].tolist()
    recommendations = f"""
    **Fallback Recommendations:**

    * **Focus Areas:** Prioritize training and calibration on **{", ".join(top_error_flags)}**, as these are the top error-prone areas.
    * **Performance Gap:** Address the performance gap between your highest and lowest-performing raters. Consider peer mentoring or targeted support for those with lower accuracy.
    * **Error Rate Check:** The current overall error rate is **{error_rate:.1f}%**. Review if this meets your team's goal (e.g., under 15%) and take action if it's too high.
    * **Process Review:** Schedule calibration sessions focused on the most common and impactful flags to ensure consistent understanding of the guidelines.
    """
    st.markdown(recommendations)


def get_ai_recommendations_with_retry(prompt, debug_mode=False):
    """Generates AI recommendations with a model rotation and retry mechanism."""
    # --- FIX APPLIED HERE: Reverted to your initial model list ---
    models = ["gemini-2.5-pro", "gemini-2.5-flash"]

    for i, model_name in enumerate(models):
        try:
            if debug_mode:
                st.info(
                    f"Attempt {i + 1}: Using model `{model_name}` with the following prompt:"
                )
                st.text_area("Prompt",
                             prompt,
                             height=250,
                             key=f"prompt_debug_{model_name}")

            model = GenerativeModel(model_name)
            config = GenerationConfig(temperature=0.2,
                                      top_p=0.9,
                                      max_output_tokens=8192)
            safety = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            response = model.generate_content(prompt,
                                              generation_config=config,
                                              safety_settings=safety)

            if response and hasattr(response,
                                    "candidates") and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason in [
                        FinishReason.STOP, FinishReason.MAX_TOKENS
                ]:
                    content = "".join(
                        part.text
                        for part in getattr(candidate.content, "parts", []))
                    if content and len(content.strip()) > 20:
                        return content.strip()

            if debug_mode:
                st.warning(
                    f"Model `{model_name}` returned an empty or invalid response. Finish Reason: {finish_reason}."
                )

        except Exception as e:
            if debug_mode:
                st.error(
                    f"An exception occurred with model `{model_name}`: {e}")
            continue

    return None


# --- SIDEBAR & GLOBAL FILTERS ---

st.sidebar.header("Dashboard Filters")

min_date = df_full["Production_Date"].min().date()
max_date = df_full["Production_Date"].max().date()

date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
)

all_raters = sorted(df_full["Rater_ID"].unique())
selected_raters = st.sidebar.multiselect("Select Rater(s)",
                                         options=all_raters,
                                         default=all_raters)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(
    date_range[1])
df = df_full[(df_full["Production_Date"] >= start_date)
             & (df_full["Production_Date"] <= end_date)
             & (df_full["Rater_ID"].isin(selected_raters))]

if df.empty:
    st.warning(
        "No data available for the selected filters. Please adjust the date range or rater selection."
    )
    st.stop()

# --- MAIN DASHBOARD ---

st.title("Auto Trim Workflow Insights Dashboard")

# Replaced st.tabs with st.radio to maintain state on rerun
tab_list = [
    "Overview",
    "Error Analysis",
    "Rater Performance",
    "Appeal Dynamics",
    "Trends",
    "Guidelines",
    "🤖 AI Recommendations",
]
active_tab = st.radio("Navigation",
                      tab_list,
                      horizontal=True,
                      label_visibility="collapsed")

# Shared calculations
total_records = len(df)
error_count = len(df[df["Marked_Error_By_QA"] == "Yes"])
error_rate = (error_count / total_records) * 100 if total_records > 0 else 0

# --- TAB-BASED CONTENT DISPLAY ---

if active_tab == "Overview":
    st.header("Performance Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Audits", f"{total_records:,}")
    col2.metric("Overall Error Rate", f"{error_rate:.2f}%")
    col3.metric(
        "Appeals Filed",
        len(df[(df["Marked_Error_By_QA"] == "Yes")
               & (df["Appeal_Status"] != "Not Appealed")]),
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            df,
            names="Flag_Type",
            hole=0.4,
            title="Flag Distribution by Volume",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        audits_over_time = (
            df.set_index("Production_Date").resample("D").size().reset_index(
                name="Count"))
        fig_line = px.line(audits_over_time,
                           x="Production_Date",
                           y="Count",
                           title="Audits Over Time")
        st.plotly_chart(fig_line, use_container_width=True)

elif active_tab == "Error Analysis":
    st.header("Error Analysis by Flag Type")
    error_per_flag = (df.groupby("Flag_Type")["Marked_Error_By_QA"].apply(
        lambda x: (x == "Yes").sum() / len(x) * 100).sort_values(
            ascending=False).round(2).reset_index())
    error_per_flag.columns = ["Flag_Type", "Error Rate (%)"]

    fig = px.bar(
        error_per_flag,
        x="Error Rate (%)",
        y="Flag_Type",
        orientation="h",
        color="Error Rate (%)",
        title="Error Rate by Flag Type",
        text_auto=".2f",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(error_per_flag, use_container_width=True)

elif active_tab == "Rater Performance":
    st.header("Rater Performance Analysis")
    rater_accuracy = (df.groupby("Rater_ID")["Marked_Error_By_QA"].apply(
        lambda x: (x == "No").sum() / len(x) * 100).sort_values(
            ascending=False).round(2).reset_index())
    rater_accuracy.columns = ["Rater_ID", "Accuracy (%)"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            rater_accuracy,
            x="Accuracy (%)",
            y="Rater_ID",
            orientation="h",
            title="Rater Accuracy",
            text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_rater_errors = (df[df["Marked_Error_By_QA"] == "Yes"].groupby(
            "Rater_ID").size().sort_values(ascending=True).reset_index())
        top_rater_errors.columns = ["Rater_ID", "Error Count"]
        fig2 = px.bar(
            top_rater_errors,
            x="Error Count",
            y="Rater_ID",
            orientation="h",
            title="Total Errors by Rater",
        )
        st.plotly_chart(fig2, use_container_width=True)

elif active_tab == "Appeal Dynamics":
    st.header("Appeal & Calibration Dynamics")
    appealed_df = df[df["Marked_Error_By_QA"] == "Yes"]

    col1, col2 = st.columns(2)
    with col1:
        appeal_dist = (
            (appealed_df["Appeal_Status"].value_counts(normalize=True) *
             100).round(2).reset_index())
        appeal_dist.columns = ["Appeal Status", "Percentage (%)"]
        fig = px.pie(
            appeal_dist,
            names="Appeal Status",
            values="Percentage (%)",
            hole=0.4,
            title="Appeal Status Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        calib = ((df["Calibration_Outcome"].value_counts(normalize=True) *
                  100).round(2).reset_index())
        calib.columns = ["Calibration Outcome", "Percentage (%)"]
        fig2 = px.pie(
            calib,
            names="Calibration Outcome",
            values="Percentage (%)",
            hole=0.4,
            title="Calibration Outcomes",
        )
        st.plotly_chart(fig2, use_container_width=True)

elif active_tab == "Trends":
    st.header("Monthly Trends")
    monthly_error = (df.groupby("Month")["Marked_Error_By_QA"].apply(
        lambda x: (x == "Yes").sum() / len(x) * 100).round(2).reset_index())
    monthly_error.columns = ["Month", "Error Rate (%)"]
    fig = px.line(
        monthly_error,
        x="Month",
        y="Error Rate (%)",
        markers=True,
        title="Monthly Error Rate Trend",
        text="Error Rate (%)",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

elif active_tab == "Guidelines":
    st.header("Auto Trim Guidelines")
    st.markdown(guidelines_content)

elif active_tab == "🤖 AI Recommendations":
    st.header("🤖 AI-Generated Recommendations")
    debug_mode = st.checkbox("Enable Debug Mode for AI Prompt")

    # These calculations need to be available for this tab
    error_per_flag = (df.groupby("Flag_Type")["Marked_Error_By_QA"].apply(
        lambda x: (x == "Yes").sum() / len(x) * 100).sort_values(
            ascending=False).round(2).reset_index())
    error_per_flag.columns = ["Flag_Type", "Error Rate (%)"]

    rater_accuracy = (df.groupby("Rater_ID")["Marked_Error_By_QA"].apply(
        lambda x: (x == "No").sum() / len(x) * 100).sort_values(
            ascending=False).round(2).reset_index())
    rater_accuracy.columns = ["Rater_ID", "Accuracy (%)"]

    monthly_error = (df.groupby("Month")["Marked_Error_By_QA"].apply(
        lambda x: (x == "Yes").sum() / len(x) * 100).round(2).reset_index())
    monthly_error.columns = ["Month", "Error Rate (%)"]

    top_error_flags_str = ", ".join([
        f"{row.Flag_Type} ({row['Error Rate (%)']:.1f}%)"
        for _, row in error_per_flag.head(3).iterrows()
    ])

    if not rater_accuracy.empty:
        rater_summary = (
            f"Accuracy ranges from {rater_accuracy.iloc[-1]['Accuracy (%)']:.1f}% "
            f"({rater_accuracy.iloc[-1]['Rater_ID']}) to {rater_accuracy.iloc[0]['Accuracy (%)']:.1f}% "
            f"({rater_accuracy.iloc[0]['Rater_ID']}).")
    else:
        rater_summary = "N/A"

    monthly_trend_desc = "stable"
    if len(monthly_error) > 1:
        recent_rate = monthly_error.iloc[-1]["Error Rate (%)"]
        try:
            previous_rate = monthly_error.iloc[-2]["Error Rate (%)"]
            if recent_rate > previous_rate * 1.1:
                monthly_trend_desc = (
                    f"trending up (from {previous_rate:.1f}% to {recent_rate:.1f}%)"
                )
            elif recent_rate < previous_rate * 0.9:
                monthly_trend_desc = (
                    f"trending down (from {previous_rate:.1f}% to {recent_rate:.1f}%)"
                )
        except IndexError:
            pass

    # Updated AI prompt with improved structure and formatting
    ai_prompt = f"""
<persona>
You are an expert Principal Program Manager specializing in operational excellence and quality assurance. Your objective is to generate a concise, data-driven, and highly actionable weekly quality report for team leads.
</persona>

<context>
  <data_summary>
    - **Time Period:** {date_range[0].strftime("%Y-%m-%d")} to {date_range[1].strftime("%Y-%m-%d")}
    - **Total Audits:** {total_records}
    - **Overall Error Rate:** {error_rate:.1f}%
    - **Top 3 Error Flags:** {top_error_flags_str}
    - **Rater Performance Summary:** {rater_summary}
    - **Monthly Error Trend:** The error rate is {monthly_trend_desc}
  </data_summary>
  <guidelines_excerpt>
    {guidelines_excerpt}
  </guidelines_excerpt>
</context>

<instructions>
Generate 4-5 actionable recommendations based on the data in the <context>.

For each recommendation, you MUST use the following Markdown structure exactly:

**Action:** Start with a strong action verb (e.g., Host, Implement, Analyze). Describe a single, clear action the team lead can take.
* **Rationale:** Explain *why* this action is necessary, directly referencing at least one specific metric from the <data_summary>.
* **Success Metric:** Define a tangible KPI to measure the action's success and a timeframe for evaluation.
* **Impact/Effort:** Classify the potential impact on quality (High, Medium, Low) and the effort required to implement (High, Medium, Low).

Additional Rules:
1.  Ensure one recommendation is for **Positive Reinforcement**, focusing on learning from top performers.
2.  Ensure one recommendation is a **Proactive Watch-out**, identifying a potential future risk.
3.  The tone must be professional, encouraging, and collaborative.
4.  Do not include any introductory or concluding paragraphs. Begin directly with the first recommendation.
</instructions>

<example>
Here is an example of a perfect response:

**Action:** Host a 45-minute calibration session focused specifically on the "Disrupted Narrative" flag.
* **Rationale:** This is our top error driver at 61.3%, indicating significant confusion around the "Incoherent Story" policy mentioned in the guidelines.
* **Success Metric:** Reduce the error rate for the "Disrupted Narrative" flag by 20% over the next 3 weeks.
* **Impact/Effort:** Impact: High / Effort: Medium
</example>
"""

    if st.button("Generate Recommendations"):
        with st.spinner("🤖 The AI is analyzing the data... Please wait."):
            ai_recs = get_ai_recommendations_with_retry(ai_prompt,
                                                        debug_mode=debug_mode)
            if "ai_recommendations" not in st.session_state:
                st.session_state.ai_recommendations = {}

            if ai_recs:
                st.session_state.ai_recommendations[active_tab] = ai_recs
            else:
                # Generate fallback recommendations text but don't display it here
                # Instead, store it in session state to be displayed outside the button-if block
                top_error_flags = error_per_flag.head(2)["Flag_Type"].tolist()
                fallback_text = f"""
                **Fallback Recommendations:**

                * **Focus Areas:** Prioritize training and calibration on **{", ".join(top_error_flags)}**, as these are the top error-prone areas.
                * **Performance Gap:** Address the performance gap between your highest and lowest-performing raters. Consider peer mentoring or targeted support for those with lower accuracy.
                * **Error Rate Check:** The current overall error rate is **{error_rate:.1f}%**. Review if this meets your team's goal (e.g., under 15%) and take action if it's too high.
                * **Process Review:** Schedule calibration sessions focused on the most common and impactical flags to ensure consistent understanding of the guidelines.
                """
                st.session_state.ai_recommendations[active_tab] = fallback_text
                st.warning(
                    "The AI model could not generate recommendations at this time. Showing fallback suggestions."
                )

    # Always check session state to display the recommendation (or fallback)
    if ("ai_recommendations" in st.session_state
            and st.session_state.ai_recommendations.get(active_tab)):
        st.markdown(st.session_state.ai_recommendations[active_tab])
