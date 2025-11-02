import streamlit as st
import pandas as pd
from io import BytesIO
from pyscript import web_app


st.set_page_config(page_title="Stacked_Model", layout="centered")

def stand(df1, df2):
    avg = df1.mean()
    stdev = df1.std()
    std_df1 = (df1 - avg) / stdev
    std_df2 = (df2 - avg) / stdev
    return std_df1, std_df2




user_input = st.text_input("Enter calculation starting point:", "60")



# --- Two columns for train & test side by side ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 Training Set")
    train_file = st.file_uploader("Upload Training Set (.xlsx)", type=["xlsx"], key="train")

with col2:
    st.subheader("📗 Test Set")
    test_file = st.file_uploader("Upload Test Set (.xlsx)", type=["xlsx"], key="test")


# --- Function to read Excel ---
def load_excel(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file, index_col=0)
    return None


train_df = load_excel(train_file)
test_df = load_excel(test_file)


# --- Show previews side by side ---
if train_df is not None and test_df is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Uploaded Training Set")
        st.dataframe(train_df)

    with col2:
        st.markdown("#### Uploaded Test Set")
        st.dataframe(test_df)

run_button = st.button("🚀 Run Calculation")

# --- Run calculation when button clicked ---
if run_button:
    if train_df is not None and test_df is not None:
        with st.spinner("🧮 Calculation going on... Please wait."):
           pls_met = web_app(df1=train_df, df2=test_df, leng=int(user_input))

        st.success("✅ Calculation Completed Successfully!")
        
        # --- Download button 2: the PLS metrics DataFrame ---
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pls_met.to_excel(writer, sheet_name="total_pls")
        excel_data2 = output.getvalue()

        st.download_button(
            label="📊 Download PLS Metrics Excel File",
            data=excel_data2,
            file_name="PLS_metrics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:

        st.warning("⚠️ Please upload both Training and Test files before running the calculation.")


