from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditLens | Credit Risk Intelligence", page_icon="💠", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ML" / "models" / "random_forest.joblib"
IMPORTANCE_PATH = ROOT / "ML" / "outputs" / "random_forest_feature_importance.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{background:radial-gradient(circle at 90% 0%,#e8edff 0,transparent 28rem),#f5f7fb;font-family:'DM Sans',sans-serif}
.block-container{max-width:1440px;padding-top:1.5rem;padding-bottom:3rem}
h1,h2,h3{font-family:Manrope,sans-serif!important;letter-spacing:-.035em;color:#182842}
[data-testid="stSidebar"]{background:#111f35}
[data-testid="stSidebar"] *{color:#eff4ff}
.hero{padding:2rem 2.2rem;border-radius:24px;color:white;background:linear-gradient(115deg,#14233d,#294a78 60%,#4668f5);box-shadow:0 18px 45px #1b30561c;margin-bottom:1.2rem}
.kicker{color:#a9c9ff;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700}
.hero-title{font:800 clamp(1.8rem,3vw,2.7rem) Manrope,sans-serif;color:white;line-height:1.15;margin:.5rem 0}
.hero-sub{color:#d8e3f5;max-width:760px;line-height:1.65}
.card{background:#fff;border:1px solid #e5ebf4;border-radius:18px;padding:1.1rem 1.25rem;box-shadow:0 7px 24px #19305209;min-height:108px}
.label{color:#758299;font-size:.76rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.value{font:800 1.5rem Manrope,sans-serif;color:#1b2b45;margin:.4rem 0}
.foot{font-size:.78rem;color:#8995a8}
.panel{background:white;border:1px solid #e5ebf4;border-radius:20px;padding:1.2rem 1.35rem;box-shadow:0 8px 28px #19305209}
.result{background:white;border:1px solid #e5ebf4;border-radius:22px;padding:1.5rem;box-shadow:0 10px 28px #182d4d0d}
.prob{font:800 3.1rem Manrope,sans-serif;letter-spacing:-.055em;line-height:1.2;margin:.3rem 0}
.muted{color:#718097;line-height:1.6;font-size:.9rem}
.pill{display:inline-block;padding:.35rem .75rem;border-radius:999px;font-size:.8rem;font-weight:700}
.green{background:#e3f8f0;color:#087b60}.red{background:#fff0ee;color:#b9382c}
div[data-testid="stForm"]{background:white;border:1px solid #e5ebf4;border-radius:20px;padding:1.3rem;box-shadow:0 8px 28px #19305209}
.stFormSubmitButton button{background:linear-gradient(100deg,#0f9f91,#4268ed)!important;color:white!important;border:0!important;border-radius:12px!important;font-weight:700!important;min-height:2.8rem}
.stButton button,.stDownloadButton button{border-radius:12px!important;font-weight:700!important}
[data-testid="stMetric"]{background:white;border:1px solid #e5ebf4;border-radius:15px;padding:.8rem 1rem}
.footer{text-align:center;color:#8995a7;font-size:.76rem;padding:1.5rem 0 .4rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

@st.cache_data
def load_importance(path):
    return pd.read_csv(path)

st.sidebar.markdown("## 💠 CreditLens")
st.sidebar.caption("Credit risk intelligence workspace")
st.sidebar.divider()

if not MODEL_PATH.exists():
    st.error(f"Saved model not found: {MODEL_PATH}. Run `python .\\ML\\credit_risk_model.py` from the project root first.")
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
    model = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    features = bundle["features"]
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

st.sidebar.success("Random Forest model ready")
st.sidebar.caption(f"Classification threshold: {threshold:.2f}")
st.sidebar.caption(f"Input features: {len(features)}")
st.sidebar.divider()
st.sidebar.markdown("**Navigation**")
st.sidebar.caption("Use the tabs on the main page to explore the app.")

st.markdown("""
<div class="hero">
  <div class="kicker">RISK ANALYTICS · MODEL WORKSPACE</div>
  <div class="hero-title">Understand risk before the decision.</div>
  <div class="hero-sub">Explore an applicant profile, estimate default probability, and review model insights in one clear workspace.</div>
</div>
""", unsafe_allow_html=True)

a,b,c = st.columns(3)
for col, label, value, foot in [
    (a,"MODEL","Random Forest","Trained and saved pipeline"),
    (b,"CLASSIFICATION THRESHOLD",f"{threshold:.0%}","Selected using validation data"),
    (c,"ASSESSMENT MODE","Single applicant","Interactive model estimate"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="foot">{foot}</div></div>',unsafe_allow_html=True)

st.write("")
tab_predict, tab_insights, tab_about = st.tabs(["✦  Risk assessment", "◈  Model insights", "ⓘ  About this tool"])

with tab_predict:
    left,right = st.columns([1.12,.88],gap="large")
    with left:
        st.markdown("### Applicant profile")
        st.markdown('<p class="muted">Complete the fields below. Loan-to-income is calculated automatically.</p>',unsafe_allow_html=True)
        with st.form("applicant_form"):
            x,y = st.columns(2)
            with x:
                age = st.number_input("Age (years)",18,100,25,1)
                income = st.number_input("Annual income ($)",1,100_000_000,60000,1000)
                employment = st.number_input("Employment length (years)",0.0,80.0,3.0,0.5)
                home = st.selectbox("Home ownership",["RENT","MORTGAGE","OWN","OTHER"])
            with y:
                loan = st.number_input("Loan amount ($)",500,100_000,10000,500)
                intent = st.selectbox("Loan purpose",["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
                prior_default = st.selectbox("Previous default on file?",["N","Y"],format_func=lambda v:"No" if v=="N" else "Yes")
                history = st.number_input("Credit history length (years)",0,80,4,1)
            ratio = loan / income
            st.info(f"Calculated loan-to-income ratio: **{ratio:.1%}**")
            submitted = st.form_submit_button("Estimate default risk  →",type="primary",use_container_width=True)

    with right:
        st.markdown("### Your risk summary")
        st.markdown('<p class="muted">Submit the applicant profile to see the estimated probability and classification.</p>',unsafe_allow_html=True)
        if submitted:
            # Clear the previous result immediately so invalid/new submissions
            # never leave an old applicant's prediction on screen.
            st.session_state.pop("prediction", None)

            if employment > age - 18:
                st.error("Employment length looks inconsistent with age. Please review the values.")
            elif history > age - 18:
                st.error("Credit history length looks inconsistent with age. Please review the values.")
            else:
                row = {
                    "person_age":age,
                    "person_income":income,
                    "person_home_ownership":home,
                    "person_emp_length":employment,
                    "loan_intent":intent,
                    "loan_amnt":loan,
                    "loan_percent_income":ratio,
                    "cb_person_default_on_file":prior_default,
                    "cb_person_cred_hist_length":history,
                }
                try:
                    frame = pd.DataFrame([row],columns=features)
                    probs = model.predict_proba(frame)[0]
                    class_list = list(model.classes_)
                    probability = float(probs[class_list.index(1)])
                    outcome = int(probability >= threshold)
                    result_record = {
                        "row": row,
                        "probability": probability,
                        "outcome": outcome,
                        "threshold": threshold,
                    }
                    st.session_state["prediction"] = result_record

                    # Keep a session-level history of each successful assessment.
                    if "applicant_history" not in st.session_state:
                        st.session_state["applicant_history"] = []
                    st.session_state["applicant_history"].append({
                        "assessment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        **row,
                        "estimated_default_probability": probability,
                        "predicted_class": "Default" if outcome == 1 else "No Default",
                        "classification_threshold": threshold,
                    })
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        result = st.session_state.get("prediction")
        if result:
            prob = result["probability"]
            is_default = result["outcome"] == 1
            color = "#bd4035" if is_default else "#07876b"
            pill = "red" if is_default else "green"
            label = "Default" if is_default else "No Default"
            note = ("The model score is at or above the selected threshold. This is a model classification, not a lending decision."
                    if is_default else
                    "The model score is below the selected threshold. This does not guarantee future repayment.")
            st.markdown(f"""
            <div class="result">
              <div class="label">ESTIMATED PROBABILITY OF DEFAULT</div>
              <div class="prob" style="color:{color}">{prob:.1%}</div>
              <div style="height:9px;background:#edf1f7;border-radius:99px;overflow:hidden;margin:.8rem 0 1rem">
                <div style="height:9px;width:{prob*100:.1f}%;background:linear-gradient(90deg,#13b99b,#4668f5);border-radius:99px"></div>
              </div>
              <span class="pill {pill}">Model prediction: {label}</span>
              <p class="muted" style="margin-top:1rem">{note}</p>
            </div>""",unsafe_allow_html=True)
            q,r = st.columns(2)
            q.metric("Threshold",f"{result['threshold']:.0%}")
            r.metric("Loan-to-income",f"{result['row']['loan_percent_income']:.1%}")
            with st.expander("Review submitted applicant details"):
                st.dataframe(pd.DataFrame([result["row"]]),hide_index=True,use_container_width=True)
            export = {**result["row"],"estimated_default_probability":prob,"predicted_class":label,"classification_threshold":result["threshold"]}
            st.download_button("Download prediction summary (CSV)",pd.DataFrame([export]).to_csv(index=False).encode("utf-8"),"credit_risk_prediction.csv","text/csv",use_container_width=True)
        else:
            st.markdown('<div class="result" style="min-height:250px"><div style="font-size:2.3rem">◌</div><h3>Ready when you are</h3><p class="muted">Complete the applicant form and select Estimate default risk. Your result will appear here.</p></div>',unsafe_allow_html=True)
        st.caption("Educational demonstration only. Do not use this output as the sole basis for a real credit decision.")

    # Applicant history is shown below the form and summary, not hidden in session state.
    st.write("")
    st.markdown("### Previous applicant assessments")
    st.markdown(
        '<p class="muted">Successful assessments from this active app session appear here. '
        'The newest assessment is shown first.</p>',
        unsafe_allow_html=True,
    )
    history = st.session_state.get("applicant_history", [])
    if history:
        history_df = pd.DataFrame(history).iloc[::-1].reset_index(drop=True)
        display_df = history_df.copy()
        if "estimated_default_probability" in display_df.columns:
            display_df["estimated_default_probability"] = display_df["estimated_default_probability"].map(lambda value: f"{value:.1%}")
        if "classification_threshold" in display_df.columns:
            display_df["classification_threshold"] = display_df["classification_threshold"].map(lambda value: f"{value:.0%}")
        if "loan_percent_income" in display_df.columns:
            display_df["loan_percent_income"] = display_df["loan_percent_income"].map(lambda value: f"{value:.1%}")
        st.caption(f"{len(history_df)} successful assessment(s) in this session")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        history_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all assessment history (CSV)",
            history_csv,
            "credit_risk_assessment_history.csv",
            "text/csv",
            use_container_width=True,
            key="download_assessment_history",
        )
        if st.button("Clear assessment history", key="clear_assessment_history"):
            st.session_state["applicant_history"] = []
            st.rerun()
    else:
        st.info("No previous assessments yet. Submit a valid applicant profile above, and it will appear here.")
    st.caption("History is kept only for the active Streamlit session. Download the CSV if you want to keep it after closing or restarting the app.")

with tab_insights:
    st.markdown("### What influences the model?")
    st.markdown('<p class="muted">Permutation importance measures how much validation performance changed when a feature was shuffled. It shows model reliance, not causation.</p>',unsafe_allow_html=True)
    if IMPORTANCE_PATH.exists():
        try:
            imp = load_importance(str(IMPORTANCE_PATH))
            if {"Feature","Importance"}.issubset(imp.columns):
                shown = imp.sort_values("Importance",ascending=True)
                chart_col, note_col = st.columns([1.2,.8],gap="large")
                with chart_col:
                    st.markdown('<div class="panel">',unsafe_allow_html=True)
                    st.markdown("**Permutation feature importance**")
                    st.bar_chart(shown.set_index("Feature")[["Importance"]],horizontal=True,use_container_width=True)
                    st.markdown("</div>",unsafe_allow_html=True)
                with note_col:
                    st.markdown("""
                    <div class="panel">
                    <div class="label">HOW TO READ THIS</div>
                    <h3>Higher importance means more model reliance</h3>
                    <p class="muted">A feature is important here when shuffling its values reduces validation average precision. Correlated features can share or mask importance.</p>
                    <p class="muted">Importance does not prove that a feature causes default or explain an individual's result.</p>
                    </div>""",unsafe_allow_html=True)
                st.dataframe(imp.sort_values("Importance",ascending=False),hide_index=True,use_container_width=True)
            else:
                st.info("The feature-importance file does not contain the expected columns.")
        except Exception as exc:
            st.warning(f"Could not load feature importance: {exc}")
    else:
        st.info("Feature-importance output not found. Run the training script to generate it.")
    d,e,f = st.columns(3)
    d.metric("Model","Random Forest")
    e.metric("Input features",len(features))
    f.metric("Threshold",f"{threshold:.0%}")

with tab_about:
    st.markdown("### About CreditLens")
    st.write("CreditLens is a portfolio demonstration that combines SQL data cleaning, Python analysis, Power BI reporting and a saved machine-learning pipeline.")
    d,e = st.columns(2,gap="large")
    with d:
        st.markdown("""<div class="panel"><div class="label">PROJECT WORKFLOW</div><h3>From data to prediction</h3><p class="muted">SQL cleaning and analysis · Python exploration · Power BI dashboards · Logistic Regression and Random Forest evaluation · Interactive applicant prediction.</p></div>""",unsafe_allow_html=True)
    with e:
        st.markdown("""<div class="panel"><div class="label">IMPORTANT LIMITATIONS</div><h3>Use with care</h3><p class="muted">Historical data may not represent future applicants. Probabilities are estimates, feature importance is not causation, and this demo is not a validated lending system. Do not enter real people's sensitive financial data.</p></div>""",unsafe_allow_html=True)

st.markdown('<div class="footer">CREDITLENS · EDUCATIONAL CREDIT-RISK DEMONSTRATION · POWERED BY RANDOM FOREST</div>',unsafe_allow_html=True)
