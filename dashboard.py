import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(
    page_title="Emotion Dashboard",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Emotion Detection Dashboard")

conn = sqlite3.connect("emotion_data.db")

df = pd.read_sql_query(
    "SELECT * FROM emotions",
    conn
)

# ================= TABLE =================

st.subheader("Stored Emotion Records")

st.dataframe(df)

# ================= METRICS =================

if not df.empty:

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))

    col2.metric(
        "Unique Persons",
        df['name'].nunique()
    )

    col3.metric(
        "Most Common Emotion",
        df['emotion'].mode()[0]
    )

    # ================= CHART =================

    st.subheader("Emotion Statistics")

    emotion_counts = df['emotion'].value_counts()

    st.bar_chart(emotion_counts)

else:
    st.warning("No data found.")

conn.close()