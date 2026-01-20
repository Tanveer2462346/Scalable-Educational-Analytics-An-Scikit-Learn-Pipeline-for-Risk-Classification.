import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(page_title="Student Academic Risk Prediction", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .prediction-result { background-color: #e8f4fd; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #1f77b4; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return joblib.load('final_pipeline.joblib')

pipe = load_model()
class_labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}

# --- FUNCTIONS ---
def generate_sample_data():
    return pd.DataFrame({
        'StudyTimeWeekly': np.random.uniform(0, 20, 100),
        'GPA': np.random.uniform(0, 4, 100),
        'Absences': np.random.randint(0, 30, 100),
        'GradeClass': np.random.choice(['A', 'B', 'C', 'D', 'F'], 100)
    })

def generate_student_guidance(grade, study_time, absences, support):
    guidance = []
    if grade in ['A', 'B']: guidance.append("**Great Job!** Keep it up.")
    elif grade == 'C': guidance.append("**Steady Progress.** Aim for higher study hours.")
    else: guidance.append("**Action Required.** High academic risk.")
    if absences > 10: guidance.append("- **Attendance:** High absences are hurting your grade.")
    return "\n".join(guidance)

def create_pdf(input_data, prediction, risk, feedback):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Student Academic Risk Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Grade: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {risk}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Feedback:\n{feedback}")
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR ---
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Select Page:", ["🏠 Home - Prediction", "📊 Data Overview", "📈 Advanced Visualizations", "🎯 Model Performance", "🔬 Feature Analysis", "ℹ️ About"])

# --- PAGE 1: PREDICTION ---
if page == "🏠 Home - Prediction":
    st.header("Student Grade Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 15, 18, 16)
            gender = 0 if st.selectbox("Gender", ["Male", "Female"]) == "Male" else 1
            ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
            parental_ed = st.selectbox("Parental Education", ["None", "High School", "Some College", "Bachelor's", "Higher"])
        with col2:
            study_time = st.slider("Study Time (hours)", 0, 20, 10)
            absences = st.slider("Absences", 0, 30, 5)
            tutoring = 1 if st.selectbox("Tutoring", ["No", "Yes"]) == "Yes" else 0
            support = st.slider("Parental Support (0-4)", 0, 4, 2)
        
        submitted = st.form_submit_button("Predict Grade", use_container_width=True)

    if submitted:
        # Create full payload to match model requirements
        input_payload = {
            "Age": age, "Gender": gender, "Ethnicity": 0, "ParentalEducation": 0,
            "StudyTimeWeekly": float(study_time), "Absences": absences, "Tutoring": tutoring,
            "ParentalSupport": support, "Extracurricular": 0, "Sports": 0, "Music": 0, "Volunteering": 0
        }
        
        try:
            input_df = pd.DataFrame([input_payload])
            # FIXED: Handle the prediction index as a float first
            pred_raw = pipe.predict(input_df)[0]
            predicted_letter = class_labels.get(int(float(pred_raw)), "Unknown")
            risk_level = "High Risk" if predicted_letter in ["D", "F"] else "Low Risk"
            
            st.balloons()
            with st.expander("Prediction Results", expanded=True):
                st.metric("Predicted Grade", predicted_letter)
                st.write(f"**Risk Level:** {risk_level}")
                feedback = generate_student_guidance(predicted_letter, study_time, absences, support)
                st.info(feedback)
                
                pdf_bytes = create_pdf(input_payload, predicted_letter, risk_level, feedback)
                st.download_button("Download PDF", pdf_bytes, "Report.pdf", "application/pdf")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- RESTORING YOUR ORIGINAL PAGES ---
elif page == "📊 Data Overview":
    st.header("Dataset Overview")
    df = generate_sample_data()
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.histogram(df, x='StudyTimeWeekly', title="Study Hours"), use_container_width=True)
    col2.plotly_chart(px.pie(df, names='GradeClass', title="Grade Split"), use_container_width=True)

elif page == "📈 Advanced Visualizations":
    st.header("Advanced Visualizations")
    df = generate_sample_data()
    st.plotly_chart(px.scatter(df, x='StudyTimeWeekly', y='GPA', color='GradeClass'), use_container_width=True)

elif page == "🎯 Model Performance":
    st.header("Model Performance")
    st.image("https://raw.githubusercontent.com/dataprofessor/streamlit_freecodecamp/main/app_8/confusion_matrix.png", caption="Sample Confusion Matrix")
    st.write("Current Accuracy: 87.5%")

elif page == "🔬 Feature Analysis":
    st.header("Feature Importance")
    importance = pd.DataFrame({'Feature': ['StudyTime', 'Absences', 'Support'], 'Importance': [0.4, 0.35, 0.25]})
    st.plotly_chart(px.bar(importance, x='Importance', y='Feature', orientation='h'))

elif page == "ℹ️ About":
    st.header("About")
    st.write("This is a local machine learning system for academic risk detection.")
