import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUESTS_LOG_FILE = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "requests.jsonl"
)

FEEDBACK_LOG_FILE = (
    PROJECT_ROOT
    / "monitoring"
    / "logs"
    / "feedback.jsonl"
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical Knowledge AI - Monitoring",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Medical Knowledge AI - Monitoring Dashboard"
)

st.markdown(
    """
    This dashboard monitors application performance,
    request success rates, latency, retrieval quality,
    and user feedback.
    """
)


# ============================================================
# LOAD JSONL FILE
# ============================================================

def load_jsonl(file_path):

    records = []

    if not file_path.exists():

        return pd.DataFrame()

    with open(

        file_path,

        "r",

        encoding="utf-8"

    ) as file:

        for line in file:

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:

                continue

    if not records:

        return pd.DataFrame()

    return pd.json_normalize(records)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=10)
def load_all_data():

    requests_df = load_jsonl(
        REQUESTS_LOG_FILE
    )

    feedback_df = load_jsonl(
        FEEDBACK_LOG_FILE
    )

    return requests_df, feedback_df


df, feedback_df = load_all_data()


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if df.empty:

    st.warning(
        "No monitoring data available yet."
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(

        df["timestamp"],

        errors="coerce"

    )


# ============================================================
# KPI METRICS
# ============================================================

total_requests = len(df)


if "success" in df.columns:

    successful_requests = int(

        df["success"]

        .fillna(False)

        .astype(bool)

        .sum()

    )

else:

    successful_requests = total_requests


failed_requests = (

    total_requests

    - successful_requests

)


success_rate = (

    successful_requests

    / total_requests

    * 100

)


if "latency_seconds" in df.columns:

    average_latency = (

        df["latency_seconds"]

        .mean()

    )

else:

    average_latency = 0


if "documents_retrieved" in df.columns:

    average_documents = (

        df["documents_retrieved"]

        .mean()

    )

else:

    average_documents = 0


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(

        "Total Requests",

        total_requests

    )


with col2:

    st.metric(

        "Success Rate",

        f"{success_rate:.2f}%"

    )


with col3:

    st.metric(

        "Failed Requests",

        failed_requests

    )


with col4:

    st.metric(

        "Average Latency",

        f"{average_latency:.2f}s"

    )


with col5:

    st.metric(

        "Avg Documents Retrieved",

        f"{average_documents:.2f}"

    )


st.divider()


# ============================================================
# CHART 1 - REQUESTS OVER TIME
# ============================================================

st.subheader(
    "📈 Requests Over Time"
)


if "timestamp" in df.columns:

    requests_over_time = (

        df.dropna(

            subset=[

                "timestamp"

            ]

        )

        .set_index(

            "timestamp"

        )

        .resample(

            "h"

        )

        .size()

        .reset_index(

            name="requests"

        )

    )

    fig_requests = px.line(

        requests_over_time,

        x="timestamp",

        y="requests",

        markers=True,

        title="Number of Requests Over Time"

    )

    st.plotly_chart(

        fig_requests,

        use_container_width=True

    )

else:

    st.info(

        "Timestamp data is not available."

    )


# ============================================================
# CHART 2 - LATENCY DISTRIBUTION
# ============================================================

st.subheader(

    "⏱️ Latency Distribution"

)


if "latency_seconds" in df.columns:

    fig_latency = px.histogram(

        df,

        x="latency_seconds",

        nbins=30,

        title="Request Latency Distribution",

        labels={

            "latency_seconds":

            "Latency (seconds)"

        }

    )

    st.plotly_chart(

        fig_latency,

        use_container_width=True

    )

else:

    st.info(

        "Latency data is not available."

    )


# ============================================================
# CHART 3 - SUCCESS VS FAILURE
# ============================================================

st.subheader(

    "✅ Request Success vs Failure"

)


status_data = pd.DataFrame(

    {

        "Status": [

            "Successful",

            "Failed"

        ],

        "Count": [

            successful_requests,

            failed_requests

        ]

    }

)


fig_status = px.pie(

    status_data,

    names="Status",

    values="Count",

    title="Request Success Rate",

    hole=0.4

)


st.plotly_chart(

    fig_status,

    use_container_width=True

)


# ============================================================
# CHART 4 - DOCUMENTS RETRIEVED
# ============================================================

st.subheader(

    "📚 Retrieved Documents"

)


if "documents_retrieved" in df.columns:

    fig_documents = px.histogram(

        df,

        x="documents_retrieved",

        nbins=20,

        title="Distribution of Retrieved Documents",

        labels={

            "documents_retrieved":

            "Documents Retrieved"

        }

    )

    st.plotly_chart(

        fig_documents,

        use_container_width=True

    )

else:

    st.info(

        "Documents retrieved data is not available."

    )


# ============================================================
# CHART 5 - USER FEEDBACK
# ============================================================

st.subheader(

    "👍 User Feedback"

)


if not feedback_df.empty:

    feedback_data = (

        feedback_df[

            "feedback"

        ]

        .value_counts()

        .reset_index()

    )

    feedback_data.columns = [

        "Feedback",

        "Count"

    ]

    fig_feedback = px.bar(

        feedback_data,

        x="Feedback",

        y="Count",

        title="Positive vs Negative User Feedback",

        text="Count"

    )

    st.plotly_chart(

        fig_feedback,

        use_container_width=True

    )

    # --------------------------------------------------------
    # FEEDBACK METRICS
    # --------------------------------------------------------

    positive_feedback = int(

        (

            feedback_df["feedback"]

            == "positive"

        )

        .sum()

    )

    negative_feedback = int(

        (

            feedback_df["feedback"]

            == "negative"

        )

        .sum()

    )

    total_feedback = (

        positive_feedback

        + negative_feedback

    )

    if total_feedback > 0:

        feedback_rate = (

            total_feedback

            / total_requests

            * 100

        )

    else:

        feedback_rate = 0

    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

    with feedback_col1:

        st.metric(

            "👍 Positive Feedback",

            positive_feedback

        )

    with feedback_col2:

        st.metric(

            "👎 Negative Feedback",

            negative_feedback

        )

    with feedback_col3:

        st.metric(

            "💬 Feedback Rate",

            f"{feedback_rate:.2f}%"

        )

else:

    st.info(

        "No feedback data available yet."

    )


# ============================================================
# RAW REQUEST DATA
# ============================================================

with st.expander(

    "🔍 View Raw Monitoring Data"

):

    st.dataframe(

        df,

        use_container_width=True

    )


# ============================================================
# RAW FEEDBACK DATA
# ============================================================

with st.expander(

    "👍 View Raw Feedback Data"

):

    if not feedback_df.empty:

        st.dataframe(

            feedback_df,

            use_container_width=True

        )

    else:

        st.info(

            "No feedback records available."

        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(

    "Medical Knowledge AI Monitoring System"

)