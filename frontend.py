import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime
import base64
import json
import os
import time
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from pose_detector import PoseDetector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# == Teachable Machine Integration ==
# =====================  T E A C H A B L E   M A C H I N E  =====================

import tensorflow as tf

TM_MODEL_PATH = "tm_model/keras_model.h5"
TM_LABELS_PATH = "tm_model/labels.txt"

# Global cached model + labels
_tm_model = None
_tm_labels = None


def load_tm_model():
    """Loads Teachable Machine model once globally."""
    global _tm_model, _tm_labels

    if _tm_model is not None:
        return _tm_model, _tm_labels

    try:
        # Load model
        _tm_model = tf.keras.models.load_model(TM_MODEL_PATH, compile=False)

        # Load labels
        with open(TM_LABELS_PATH, "r") as f:
            _tm_labels = [line.strip() for line in f.readlines() if line.strip()]

        if not _tm_labels:
            _tm_labels = ["Class 1", "Class 2", "Class 3"]

        return _tm_model, _tm_labels

    except Exception as e:
        st.error(f"Failed to load Teachable Machine model: {e}")
        return None, None


def predict_tm_posture(image_bgr):
    """
    Run Teachable Machine image model on a BGR image.
    Returns: (label, confidence_percent) or (None, None) on error.
    """
    model, labels = load_tm_model()
    if model is None or not labels:
        return None, None

    # Teachable Machine default input size (usually 224x224)
    IMG_SIZE = (224, 224)

    try:
        # BGR -> RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_normalized = img_resized.astype("float32") / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0)

        preds = model.predict(input_tensor)
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0][idx] * 100.0)
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        return label, conf
    except Exception as e:
        st.error(f"❌ TM prediction error: {e}")
        return None, None


# 🔗 IMPORT FROM LINKED BACKEND
try:
    from backend import (
        IntelliFitPatient,
        save_patient_data,
        load_patient_data,
        calculate_angle,
        get_exercise_landmarks_both,
        assess_medical_progress,
        pelvic_analyzer,
        report_generator,
        AdvancedPelvicAnalyzer,
        ProfessionalReportGenerator,
        MEDICAL_ROM_STANDARDS,
        patient_to_dict,
    )

    print("✅ Successfully linked with backend.py")
except ImportError as e:
    st.error(f"❌ Backend linking error: {str(e)}")
    st.error("📁 Make sure 'backend.py' exists in the same folder")
    st.stop()

# 🎨 PROFESSIONAL MEDICAL UI CONFIGURATION
st.set_page_config(
    page_title="🏥 IntelliFit Pro - Medical Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.intellifit.com/help",
        "Report a bug": "https://www.intellifit.com/bug",
        "About": "IntelliFit Pro - Professional Healthcare Analytics Platform v2.0",
    },
)

# 🎨 PROFESSIONAL MEDICAL STYLING (LIGHT CLINICAL THEME)
st.markdown(
    """
<style>
/* ========== GLOBAL APP STYLING ========== */
.stApp {
    background: #f3f4f6; /* light grey hospital-style */
    color: #0f172a;      /* dark blue/black text */
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Keep standard Streamlit text dark */
h1, h2, h3, h4, h5, h6, p, label, span, li, small {
    color: #0f172a;
}

/* ========== MEDICAL HEADER ========== */
.medical-header {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 60%, #1e40af 100%);
    padding: 1.5rem;
    border-radius: 16px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

.medical-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
}

.medical-header p {
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    opacity: 0.95;
}

/* ========== SIDEBAR ========== */
.css-1d391kg, [data-testid="stSidebar"] > div:first-child {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #0f172a;
}

/* ========== DASHBOARD CARDS ========== */
.dashboard-card {
    background: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin: 0.75rem 0;
    color: #111827;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.dashboard-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    border-color: #bfdbfe;
}

.dashboard-card h3 {
    color: #2563eb;
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.dashboard-card p {
    color: #4b5563;
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0.1rem 0;
}

/* ========== STATUS TAGS ========== */
.status-card {
    background: #ecfdf5;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    color: #065f46;
    margin: 0.35rem 0;
    border: 1px solid #bbf7d0;
    font-size: 0.85rem;
}

.status-card.warning {
    background: #fffbeb;
    color: #92400e;
    border-color: #fed7aa;
}

.status-card.danger {
    background: #fef2f2;
    color: #b91c1c;
    border-color: #fecaca;
}

/* ========== METRIC CARDS ========== */
.metric-container {
    background: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    text-align: center;
    margin: 0.5rem 0;
    color: #111827;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    margin: 0;
}

.metric-label {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ========== CAMERA FEED ========== */
.camera-feed {
    border: 1px solid #d1d5db;
    border-radius: 14px;
    background: #ffffff;
    padding: 0.75rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
}

.camera-feed img {
    border-radius: 10px;
    width: 100%;
    max-width: 640px;
    height: auto;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 999px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
}

/* ========== ALERTS ========== */

.alert-success {
    background: #ecfdf5;
    color: #065f46;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #10b981;
    font-size: 0.9rem;
}

.alert-warning {
    background: #fffbeb;
    color: #92400e;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #f59e0b;
    font-size: 0.9rem;
}

.alert-danger {
    background: #fef2f2;
    color: #b91c1c;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #ef4444;
    font-size: 0.9rem;
}

/* ========== FORM INPUTS ========== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
textarea {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #111827 !important;
    font-size: 0.9rem !important;
}

.stTextInput > div > div > input::placeholder,
textarea::placeholder {
    color: #9ca3af !important;
}

/* ========== LIVE INDICATOR ========== */
.live-indicator {
    background: #fee2e2;
    color: #b91c1c;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border: 1px solid #fecaca;
}

/* ========== FOOTER ========== */
.app-footer {
    text-align: center;
    color: #6b7280;
    padding: 2rem 0;
    border-top: 1px solid #e5e7eb;
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# 🔧 SESSION STATE INITIALIZATION
def init_session_state():
    """Initialize all session state variables"""
    session_vars = {
        "patient": None,
        "analysis_running": False,
        "current_analysis": None,
        "rep_count": 0,
        "exercise_stage": "down",
        "session_logs": [],
        "dashboard_view": "overview",
        "user_preferences": {"theme": "light", "notifications": True},
        "system_status": {"camera": "ready", "ai": "ready", "storage": "ready"},
        "selected_page": "dashboard",
        # live session data buffer + start time
        "live_analysis_buffer": [],
        "live_session_start": None,
        "last_exercise_report": None,
        "exercise_start_time": None,
    }

    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


init_session_state()


# 🏥 PROFESSIONAL SIDEBAR NAVIGATION
def render_professional_sidebar():
    """Render professional medical sidebar with proper navigation"""
    with st.sidebar:
        # Logo and Title
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1rem;">
            <h2 style="color: #2563eb; margin: 0; font-weight: 700;">🏥 IntelliFit Pro</h2>
            <p style="color: #6b7280; font-size: 0.8rem; margin: 0.5rem 0 0 0;">Medical Analytics Platform v2.0</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # System Status
        st.markdown("### 📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            <div class="status-card">
                🟢 AI Engine<br>
                <small>Online</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="status-card">
                📹 Camera<br>
                <small>Ready</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Navigation Menu
        st.markdown("### 🧭 Navigation")

        # Navigation buttons
        if st.button("🏠 Dashboard", key="nav_dashboard", width='stretch'):
            st.session_state.selected_page = "dashboard"
            st.rerun()

        if st.button(
            "👤 Patient Management", key="nav_patients", width='stretch'
        ):
            st.session_state.selected_page = "patients"
            st.rerun()

        if st.button(
            "📐 Pelvic Analysis", key="nav_analysis", width='stretch'
        ):
            st.session_state.selected_page = "analysis"
            st.rerun()

        if st.button("📋 Reports", key="nav_reports", width='stretch'):
            st.session_state.selected_page = "reports"
            st.rerun()

       
        # Patient Quick Info
        if st.session_state.patient:
            st.markdown("---")
            st.markdown("### 👤 Current Patient")
            st.markdown(
                f"""
            <div class="dashboard-card">
                <strong>{st.session_state.patient.name}</strong><br>
                <small>ID: {st.session_state.patient.patient_id}</small><br>
                <small>Age: {st.session_state.patient.age}</small><br>
                <small>Condition: {st.session_state.patient.condition}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Quick Actions
        st.markdown("---")
        

        # Footer
        st.markdown("---")
        st.markdown(
            """
        <div style="text-align: center; color: #6b7280; font-size: 0.7rem;">
            <strong>IntelliFit Pro</strong><br>
            Medical Grade Analytics<br>
            <small>v2.0 | Licensed for Clinical Use</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    return st.session_state.selected_page


# 📊 PROFESSIONAL DASHBOARD COMPONENTS
def render_dashboard_metrics():
    """Render professional dashboard metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_patients = 1 if st.session_state.patient else 0
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{total_patients}</div>
            <div class="metric-label">Active Patients</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        analyses_count = (
            len(st.session_state.patient.pelvic_history)
            if st.session_state.patient
            else 0
        )
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{analyses_count}</div>
            <div class="metric-label">Total Analyses</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        sessions_count = (
            len(st.session_state.patient.sessions) if st.session_state.patient else 0
        )
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{sessions_count}</div>
            <div class="metric-label">Exercise Sessions</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        avg_score = "N/A"
        if st.session_state.patient and st.session_state.patient.pelvic_history:
            scores = [h["posture_score"] for h in st.session_state.patient.pelvic_history]
            avg_score = f"{np.mean(scores):.1f}"

        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{avg_score}</div>
            <div class="metric-label">Avg Posture Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_professional_alerts():
    """Render professional medical alerts"""
    st.markdown("### 🚨 Clinical Alerts")

    if st.session_state.patient and st.session_state.patient.pelvic_history:
        latest_analysis = st.session_state.patient.pelvic_history[-1]
        score = latest_analysis["posture_score"]

        if score >= 8:
            st.markdown(
                """
            <div class="alert-success">
                ✅ <strong>Excellent Clinical Status</strong><br>
                Patient demonstrates excellent postural alignment. Continue current therapy protocol.
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif score >= 6:
            st.markdown(
                """
            <div class="alert-warning">
                ⚠️ <strong>Moderate Clinical Attention</strong><br>
                Patient shows moderate postural deviation. Consider therapy plan adjustments.
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="alert-danger">
                🚨 <strong>Immediate Clinical Intervention Required</strong><br>
                Patient shows significant postural dysfunction. Schedule urgent consultation.
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
        <div class="alert-warning">
            📋 <strong>No Clinical Data Available</strong><br>
            Register a patient and perform analysis to view clinical status alerts.
        </div>
        """,
            unsafe_allow_html=True,
        )


# 🏥 MAIN APPLICATION PAGES
def render_dashboard_page():
    """Render main professional dashboard"""
    st.markdown(
        """
    <div class="medical-header">
        <h1>🏥 IntelliFit Pro Dashboard</h1>
        <p>Professional Healthcare Analytics & Advanced Posture Assessment Platform</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Dashboard Metrics
    render_dashboard_metrics()

    # Two column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Recent Clinical Activity
        st.markdown("### 📈 Recent Clinical Activity")
        if st.session_state.patient and st.session_state.patient.pelvic_history:
            df_data = []
            for entry in st.session_state.patient.pelvic_history[-5:]:
                df_data.append(
                    {
                        "Date": entry["date"].strftime("%Y-%m-%d %H:%M"),
                        "Pelvic Tilt": f"{entry['left_pelvic_angle']:.1f}°",
                        "Posture Score": f"{entry['posture_score']:.1f}/10",
                        "Clinical Status": entry["interpretation"][:50]
                        + "..."
                        if len(entry["interpretation"]) > 50
                        else entry["interpretation"],
                    }
                )

            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.info(
                "📊 No recent clinical activity. Start by registering a patient and performing analysis."
            )

        # Progress Visualization
        if st.session_state.patient and len(st.session_state.patient.pelvic_history) > 1:
            st.markdown("### 📊 Clinical Progress Visualization")
            df_chart = pd.DataFrame(
                [
                    {
                        "Date": entry["date"],
                        "Posture Score": entry["posture_score"],
                        "Pelvic Tilt": abs(entry["left_pelvic_angle"]),
                    }
                    for entry in st.session_state.patient.pelvic_history
                ]
            )

            fig = px.line(
                df_chart,
                x="Date",
                y=["Posture Score", "Pelvic Tilt"],
                title="📈 Patient Clinical Progress Tracking",
                color_discrete_sequence=["#2563eb", "#f59e0b"],
            )
            fig.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font_color="#0f172a",
                height=400,
                showlegend=True,
            )
            st.plotly_chart(fig, width='stretch')

    with col2:
        # Clinical Status Alerts
        render_professional_alerts()

        # Patient Summary Card
        if st.session_state.patient:
            st.markdown("### 👤 Patient Summary")
            progress = st.session_state.patient.get_progress_summary()

            # Clinical status color coding
            clinical_status = progress.get("clinical_status", "No data")
            if "Excellent" in clinical_status:
                status_class = "status-card"
            elif "Good" in clinical_status:
                status_class = "status-card warning"
            else:
                status_class = "status-card danger"

            st.markdown(
                f"""
            <div class="dashboard-card">
                <h3>📊 Clinical Overview</h3>
                <p><strong>Patient:</strong> {st.session_state.patient.name}</p>
                <p><strong>Therapy Duration:</strong> {progress.get('therapy_duration_days', 0)} days</p>
                <p><strong>Total Assessments:</strong> {progress.get('total_analyses', 0)}</p>
                <p><strong>Exercise Sessions:</strong> {progress.get('total_sessions', 0)}</p>
            </div>
            
            <div class="{status_class}">
                <strong>Clinical Status</strong><br>
                <small>{clinical_status}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="dashboard-card">
                <h3>👤 No Active Patient</h3>
                <p>Please register a patient to begin clinical assessment and monitoring.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_patient_management_page():
    """Enhanced patient management with professional medical forms"""
    st.markdown(
        """
    <div class="medical-header">
        <h1>👤 Patient Management System</h1>
        <p>Professional Patient Registration & Comprehensive Medical History Management</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Professional Patient Registration
    st.markdown("### 📝 Professional Patient Registration")

    with st.form("enhanced_patient_registration", clear_on_submit=False):
        st.markdown("#### 📋 Patient Demographics & Contact Information")

        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("🆔 Patient ID*", placeholder="patient identifier")
            full_name = st.text_input("👤 Full Name*", placeholder="Enter Name")

        with col2:
            age = st.number_input("📅 Age*", min_value=0, max_value=120, value=25)
            gender = st.selectbox("⚧ Gender", ["Male", "Female"])

        with col3:
            email = st.text_input("📧 Email Address", placeholder="patient@email.com")

        st.markdown("#### 🏥 Medical Information")
        col1, col2 = st.columns(2)

        with col1:
            condition = st.selectbox(
                "🩺 Primary Diagnosis*",
                [
                    "Pelvic Floor Dysfunction",
                    "Lumbar Spine Disorder",
                    "Post-Surgical Rehabilitation",
                    "Spinal Alignment Disorder",
                    "Sports-Related Injury",
                    "Chronic Pain Syndrome",
                    "Postural Correction Needed",
                    "Musculoskeletal Disorder",
                    "Neurological Rehabilitation",
                    "Work-Related Injury",
                    "Motor Vehicle Accident",
                    "General Physiotherapy Assessment",
                ],
            )

            referring_physician = st.text_input(
                "👨‍⚕️ Referring Physician", placeholder="Dr. Last Name"
            )
            primary_complaint = st.text_area(
                "🩺 Chief Complaint", placeholder="Patient's primary concern or symptoms"
            )

        st.markdown("#### 📋 Clinical Assessment & History")
        col1, col2 = st.columns(2)

        with col1:
            pain_level = st.slider("🩹 Current Pain Level (0-10)", 0, 10, 0)
            previous_therapy = st.checkbox("Previous Physiotherapy Treatment")
            surgical_history = st.checkbox("Relevant Surgical History")

        with col2:
            medications = st.text_area(
                "💊 Current Medications", placeholder="List current medications"
            )
            medical_history = st.text_area(
                "🏥 Relevant Medical History", placeholder="Relevant medical conditions"
            )

        clinical_notes = st.text_area(
            "📝 Initial Clinical Notes",
            placeholder="Initial assessment notes, observations, and treatment goals",
        )

        # Submit button
        submitted = st.form_submit_button(
            "✅ Register Patient", type="primary", width='stretch'
        )

        if submitted:
            if patient_id.strip() and full_name.strip() and age > 0 and condition:
                # Create enhanced patient profile
                new_patient = IntelliFitPatient(
                    patient_id.strip(),
                    full_name.strip(),
                    age,
                    condition,
                    datetime.datetime.now(),
                )

                # Add enhanced professional attributes
                new_patient.gender = gender
                new_patient.email = email
                new_patient.referring_physician = referring_physician
                new_patient.primary_complaint = primary_complaint
                new_patient.clinical_notes = clinical_notes
                new_patient.current_pain_level = pain_level
                new_patient.medications = medications
                new_patient.medical_history = medical_history
                new_patient.previous_therapy = previous_therapy
                new_patient.surgical_history = surgical_history

                # Registration timestamp
                new_patient.registration_date = datetime.datetime.now()

                st.session_state.patient = new_patient

                if save_patient_data(new_patient):
                    st.markdown(
                        """
                    <div class="alert-success">
                        ✅ <strong>Patient Successfully Registered!</strong><br>
                        Professional patient profile has been created with complete medical information.<br>
                        Patient is now ready for clinical assessment and treatment planning.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                    <div class="alert-danger">
                        ❌ <strong>Registration Failed</strong><br>
                        Unable to save patient data to system. Please verify all information and try again.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                <div class="alert-warning">
                    ⚠️ <strong>Required Information Missing</strong><br>
                    Please complete all required fields marked with (*) before submitting registration.
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Current Patient Profile Display
    if st.session_state.patient:
        st.markdown("---")
        st.markdown("### 👤 Current Patient Profile")

        # Professional patient info display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            <div class="dashboard-card">
                <h3>📋 Demographics</h3>
                <p><strong>Name:</strong> {st.session_state.patient.name}</p>
                <p><strong>ID:</strong> {st.session_state.patient.patient_id}</p>
                <p><strong>Age:</strong> {st.session_state.patient.age}</p>
                <p><strong>Gender:</strong> {getattr(st.session_state.patient, 'gender', 'Not specified')}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="dashboard-card">
                <h3>🏥 Medical Information</h3>
                <p><strong>Diagnosis:</strong> {st.session_state.patient.condition}</p>
                <p><strong>Physician:</strong> {getattr(st.session_state.patient, 'referring_physician', 'Not specified')}</p>
                <p><strong>Start Date:</strong> {st.session_state.patient.therapy_start_date.strftime('%Y-%m-%d')}</p>
                <p><strong>Pain Level:</strong> {getattr(st.session_state.patient, 'current_pain_level', 'Not assessed')}/10</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            progress = st.session_state.patient.get_progress_summary()
            st.markdown(
                f"""
            <div class="dashboard-card">
                <h3>📈 Clinical Progress</h3>
                <p><strong>Duration:</strong> {progress.get('therapy_duration_days', 0)} days</p>
                <p><strong>Assessments:</strong> {progress.get('total_analyses', 0)}</p>
                <p><strong>Sessions:</strong> {progress.get('total_sessions', 0)}</p>
                <p><strong>Status:</strong> {progress.get('clinical_status', 'No data')[:30]}...</p>
                <p><strong>Last Visit:</strong> {datetime.datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )



# ---------------------- PELVIC ANALYSIS PAGES -------------------------
def render_pelvic_analysis_page():
    """Professional pelvic analysis with advanced clinical interface"""
    st.markdown(
        """
    <div class="medical-header">
        <h1>📐 Professional Pelvic Analysis</h1>
        <p>Advanced AI-Powered Clinical Posture Assessment & Diagnostic Analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.patient:
        st.markdown(
            """
        <div class="alert-warning">
            ⚠️ <strong>No Patient Selected</strong><br>
            Please register a patient in the Patient Management section before performing clinical analysis.
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Analysis method selection
    st.markdown("### 🎯 Clinical Analysis Method")
    analysis_method = st.radio(
        "Select Professional Analysis Method:",
        ["📷 Static Image Analysis", "🎥 Live Camera Assessment", "📊 Historical Analysis Review"],
        horizontal=True,
    )

    if analysis_method == "📷 Static Image Analysis":
        render_static_image_analysis()
    elif analysis_method == "🎥 Live Camera Assessment":
        render_live_camera_analysis()
    else:
        render_historical_analysis_review()


def render_static_image_analysis():
    """Professional static image analysis interface with TM"""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📤 Upload Clinical Image")

        uploaded_file = st.file_uploader(
            "Select patient assessment image",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear, full-body frontal or sagittal view image for optimal clinical analysis",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="camera-feed">', unsafe_allow_html=True)
            st.image(image, caption="📷 Patient Assessment Image", width=350)
            st.markdown("</div>", unsafe_allow_html=True)

            # Image quality check
            img_array = np.array(image)
            height, width = img_array.shape[:2]

            if width < 400 or height < 600:
                st.warning("⚠️ Image resolution may be too low for optimal analysis")
            else:
                st.success("✅ Image quality suitable for clinical analysis")

    with col2:
        if uploaded_file and st.button(
            "🔍 Perform Clinical Analysis", type="primary", width='stretch'
        ):
            with st.spinner("🔄 Processing advanced clinical analysis..."):
                # Convert and process image
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image_array

                # MediaPipe processing
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                )

                results = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    # Extract landmark coordinates
                    landmark_array = []
                    for lm in results.pose_landmarks.landmark:
                        landmark_array.append([lm.x, lm.y, lm.z, lm.visibility])

                    # Perform advanced pelvic analysis
                    analysis_result = pelvic_analyzer.calculate_pelvic_tilt(landmark_array)

                    # Teachable Machine prediction
                    tm_label, tm_conf = predict_tm_posture(image_bgr)
                    if tm_label is not None:
                        analysis_result["tm_label"] = tm_label
                        analysis_result["tm_confidence"] = tm_conf

                    if "error" not in analysis_result:
                        # Display comprehensive results
                        st.markdown("### 📊 Clinical Analysis Results")

                        # Primary metrics display
                        col_a, col_b, col_c, col_d, col_e = st.columns(5)

                        with col_a:
                            st.markdown(
                                f"""
                            <div class="metric-container">
                                <div class="metric-value">{analysis_result.get('pelvic_tilt_angle', 0):.1f}°</div>
                                <div class="metric-label">Pelvic Tilt</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col_b:
                            st.markdown(
                                f"""
                            <div class="metric-container">
                                <div class="metric-value">{analysis_result.get('lateral_tilt_angle', 0):.1f}°</div>
                                <div class="metric-label">Lateral Tilt</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col_c:
                            score = analysis_result.get("posture_score", 0)
                            score_color = (
                                "#10b981" if score >= 8 else "#f59e0b" if score >= 6 else "#ef4444"
                            )
                            st.markdown(
                                f"""
                            <div class="metric-container">
                                <div class="metric-value" style="color: {score_color}">{score:.1f}</div>
                                <div class="metric-label">Posture Score</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col_d:
                            confidence = analysis_result.get("measurement_confidence", 0)
                            st.markdown(
                                f"""
                            <div class="metric-container">
                                <div class="metric-value">{confidence:.0f}%</div>
                                <div class="metric-label">MP Confidence</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col_e:
                            tm_label = analysis_result.get("tm_label", "N/A")
                            tm_conf = analysis_result.get("tm_confidence", 0.0)
                            st.markdown(
                                f"""
                            <div class="metric-container">
                                <div class="metric-value">{tm_conf:.0f}%</div>
                                <div class="metric-label">TM: {tm_label}</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        # Professional clinical interpretation
                        st.markdown("### 🏥 Clinical Interpretation")

                        col_i, col_ii = st.columns(2)

                        with col_i:
                            st.markdown(
                                f"""
                            <div class="dashboard-card">
                                <h3>🔬 Clinical Assessment</h3>
                                <p><strong>Functional Impact:</strong><br>{analysis_result.get('functional_impact', 'Unable to assess')}</p>
                                <p><strong>Analysis Quality:</strong> {analysis_result.get('analysis_quality', 'Unknown').title()}</p>
                                <p><strong>Weight Distribution:</strong> {analysis_result.get('weight_distribution', {}).get('distribution', 'Not assessed')}</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col_ii:
                            st.markdown(
                                """
                            <div class="dashboard-card">
                                <h3>⚠️ Risk Factors</h3>
                            """,
                                unsafe_allow_html=True,
                            )

                            risk_factors = analysis_result.get("risk_factors", [])
                            for risk in risk_factors[:4]:
                                st.markdown(f"<p>• {risk}</p>", unsafe_allow_html=True)

                            st.markdown("</div>", unsafe_allow_html=True)

                        # Clinical recommendations
                        st.markdown("### 📋 Evidence-Based Clinical Recommendations")
                        recommendations = analysis_result.get("clinical_recommendations", [])

                        for i, rec in enumerate(recommendations[:6], 1):
                            priority = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
                            priority_color = (
                                "#ef4444"
                                if priority == "High"
                                else "#f59e0b"
                                if priority == "Medium"
                                else "#10b981"
                            )

                            st.markdown(
                                f"""
                            <div class="dashboard-card">
                                <h3 style="color: {priority_color};">#{i} - {priority} Priority</h3>
                                <p>{rec}</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        # Save to patient record (with TM data)
                        _ = st.session_state.patient.add_pelvic_analysis(
                            analysis_result.get("pelvic_tilt_angle", 0),
                            analysis_result.get("lateral_tilt_angle", 0),
                            analysis_result.get("posture_score", 0),
                            analysis_result,
                        )

                        if save_patient_data(st.session_state.patient):
                            st.markdown(
                                """
                            <div class="alert-success">
                                ✅ <strong>Clinical Analysis Complete</strong><br>
                                Assessment has been saved to patient's medical record with comprehensive clinical documentation, including Teachable Machine classification.
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    else:
                        st.markdown(
                            """
                        <div class="alert-danger">
                            ❌ <strong>Analysis Processing Error</strong><br>
                            Unable to complete pelvic analysis. Please ensure the image shows a clear full-body view.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                else:
                    st.markdown(
                        """
                    <div class="alert-warning">
                        ⚠️ <strong>Pose Detection Failed</strong><br>
                        No human pose detected in the image. Please ensure:
                        • Clear full-body view (head to feet visible)
                        • Good lighting and contrast
                        • Patient standing upright
                        • Minimal background distractions
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                pose.close()


def render_live_camera_analysis():
    """Professional live camera analysis interface with session-based recording"""
    st.markdown("#### 🎥 Live Clinical Assessment")

    col1, col2 = st.columns([3, 1])

    # ============== RIGHT SIDE: PROTOCOL + CONTROLS ==============
    with col2:
        st.markdown(
            """
        <div class="dashboard-card">
            <h3>📋 Assessment Protocol</h3>
            <p>• Position patient 6-8 feet from camera</p>
            <p>• Ensure complete body visibility</p>
            <p>• Optimize lighting conditions</p>
            <p>• Patient in relaxed standing position</p>
            <p>• Minimal clothing for landmark visibility</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Start button
        if st.button("📹 Start Live Assessment", type="primary", width='stretch'):
            st.session_state.analysis_running = True
            st.session_state.live_analysis_buffer = []  # clear previous session data
            st.session_state.live_session_start = datetime.datetime.now()
            st.rerun()

        # Stop button
        if st.button("⏹️ Stop Assessment", type="secondary", width='stretch'):
            st.session_state.analysis_running = False
            st.rerun()

        # Live badge
        if st.session_state.analysis_running:
            st.markdown(
                '<div class="live-indicator">🔴 LIVE ANALYSIS</div>',
                unsafe_allow_html=True,
            )

    # ============== LEFT SIDE: CAMERA + METRICS ==============
    with col1:
        camera_placeholder = st.empty()
        metrics_placeholder = st.empty()

        if st.session_state.analysis_running:
            try:
                cap = cv2.VideoCapture(0)
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                )

                frame_count = 0
                while cap.isOpened() and st.session_state.analysis_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    # Process every 3rd frame to reduce load
                    if frame_count % 3 != 0:
                        continue

                    # Resize for consistent processing
                    frame_resized = cv2.resize(frame, (640, 480))

                    # Process with MediaPipe
                    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    if results.pose_landmarks:
                        # Draw landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame_resized,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=2
                            ),
                            mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 0, 255), thickness=2
                            ),
                        )

                        # Build landmark array
                        landmark_array = []
                        for lm in results.pose_landmarks.landmark:
                            landmark_array.append([lm.x, lm.y, lm.z, lm.visibility])

                        # Pelvic analysis
                        analysis_result = pelvic_analyzer.calculate_pelvic_tilt(
                            landmark_array
                        )

                        # Teachable Machine prediction on this frame
                        tm_label, tm_conf = predict_tm_posture(frame_resized)
                        if tm_label is not None:
                            analysis_result["tm_label"] = tm_label
                            analysis_result["tm_confidence"] = tm_conf

                        if "error" not in analysis_result:
                            # Store latest single-frame analysis
                            st.session_state.current_analysis = analysis_result

                            # Sample every 9th processed frame
                            if frame_count % 9 == 0:
                                st.session_state.live_analysis_buffer.append(
                                    {
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "pelvic_tilt_angle": float(
                                            analysis_result.get("pelvic_tilt_angle", 0)
                                        ),
                                        "lateral_tilt_angle": float(
                                            analysis_result.get(
                                                "lateral_tilt_angle", 0
                                            )
                                        ),
                                        "posture_score": float(
                                            analysis_result.get("posture_score", 0)
                                        ),
                                        "quality_score": float(
                                            analysis_result.get("quality_score", 0)
                                        ),
                                        "analysis_quality": analysis_result.get(
                                            "analysis_quality", "unknown"
                                        ),
                                        "measurement_confidence": float(
                                            analysis_result.get(
                                                "measurement_confidence", 0
                                            )
                                        ),
                                        "tm_label": analysis_result.get(
                                            "tm_label", None
                                        ),
                                        "tm_confidence": float(
                                            analysis_result.get(
                                                "tm_confidence", 0.0
                                            )
                                        ),
                                    }
                                )

                            # Live metrics UI
                            with metrics_placeholder.container():
                                col_a, col_b, col_c, col_d, col_e = st.columns(5)

                                with col_a:
                                    st.metric(
                                        "🎯 Pelvic Tilt",
                                        f"{analysis_result.get('pelvic_tilt_angle', 0):.1f}°",
                                    )

                                with col_b:
                                    st.metric(
                                        "📐 Lateral Tilt",
                                        f"{analysis_result.get('lateral_tilt_angle', 0):.1f}°",
                                    )

                                with col_c:
                                    score = analysis_result.get("posture_score", 0)
                                    color_emoji = (
                                        "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
                                    )
                                    st.metric(
                                        f"{color_emoji} Score",
                                        f"{score:.1f}/10",
                                    )

                                with col_d:
                                    st.metric(
                                        "📊 MP Quality",
                                        analysis_result.get(
                                            "analysis_quality", "Unknown"
                                        ).title(),
                                    )

                                with col_e:
                                    label_display = analysis_result.get(
                                        "tm_label", "N/A"
                                    )
                                    conf_display = analysis_result.get(
                                        "tm_confidence", 0.0
                                    )
                                    st.metric(
                                        "🤖 TM Class",
                                        f"{label_display} ({conf_display:.0f}%)",
                                    )

                    # Show live video feed
                    try:
                        camera_placeholder.image(
                            cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB),
                            caption="🎥 Live Clinical Assessment Feed",
                            width=640,
                        )
                    except Exception:
                        camera_placeholder.markdown("**Live feed unavailable**")

                    # Safety limit (~1 min)
                    if frame_count > 1800:
                        break

                cap.release()
                pose.close()

            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.session_state.analysis_running = False

        else:
            st.info("📹 Click 'Start Live Assessment' to begin real-time clinical analysis")

    # ============== SESSION SUMMARY & SAVE ==============
    st.markdown("---")

    if st.session_state.live_analysis_buffer:
        st.markdown("### 📈 Live Session Summary")

        pelvic_vals = [
            d["pelvic_tilt_angle"] for d in st.session_state.live_analysis_buffer
        ]
        lat_vals = [
            d["lateral_tilt_angle"] for d in st.session_state.live_analysis_buffer
        ]
        score_vals = [d["posture_score"] for d in st.session_state.live_analysis_buffer]

        avg_pelvic = float(np.mean(pelvic_vals))
        avg_lateral = float(np.mean(lat_vals))
        avg_score = float(np.mean(score_vals))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Pelvic Tilt", f"{avg_pelvic:.1f}°")
        with col2:
            st.metric("Avg Lateral Tilt", f"{avg_lateral:.1f}°")
        with col3:
            st.metric("Avg Posture Score", f"{avg_score:.1f}/10")

        if st.session_state.live_session_start:
            duration_sec = (
                datetime.datetime.now() - st.session_state.live_session_start
            ).total_seconds()
            st.caption(
                f"Session duration ~ {int(duration_sec)} seconds • "
                f"{len(st.session_state.live_analysis_buffer)} sampled measurements"
            )

        if st.button("💾 Save Live Session to Patient Record", type="primary"):
            analysis_data = {
                "pelvic_tilt_angle": avg_pelvic,
                "lateral_tilt_angle": avg_lateral,
                "posture_score": avg_score,
                "analysis_quality": "session_average",
                "measurement_confidence": 100.0,
                "time_series": st.session_state.live_analysis_buffer,
                "session_start": st.session_state.live_session_start.isoformat()
                if st.session_state.live_session_start
                else None,
                "session_end": datetime.datetime.now().isoformat(),
            }

            _ = st.session_state.patient.add_pelvic_analysis(
                left_angle=avg_pelvic,
                right_angle=avg_lateral,
                posture_score=avg_score,
                analysis_data=analysis_data,
            )

            if save_patient_data(st.session_state.patient):
                st.success(
                    "✅ Live session saved to patient medical record (with full time-series data)!"
                )
                st.session_state.live_analysis_buffer = []
                st.session_state.live_session_start = None
            else:
                st.error("❌ Failed to save live session data.")
    else:
        st.info(
            "No live session data recorded yet. Start an assessment to begin recording."
        )


def render_historical_analysis_review():
    """Professional historical analysis review"""
    if not st.session_state.patient.pelvic_history:
        st.info(
            "📊 No historical analysis data available. Perform some assessments first."
        )
        return

    st.markdown("#### 📈 Historical Clinical Analysis Review")

    # Create comprehensive historical data table
    df_data = []
    for i, entry in enumerate(st.session_state.patient.pelvic_history):
        df_data.append(
            {
                "Assessment #": i + 1,
                "Date": entry["date"].strftime("%Y-%m-%d %H:%M"),
                "Pelvic Tilt (°)": f"{entry['left_pelvic_angle']:.1f}",
                "Lateral Tilt (°)": f"{entry.get('right_pelvic_angle', entry['analysis_data'].get('lateral_tilt_angle', 0)):.1f}",
                "Posture Score": f"{entry['posture_score']:.1f}/10",
                "Clinical Status": entry["interpretation"][:40]
                + "..."
                if len(entry["interpretation"]) > 40
                else entry["interpretation"],
                "Quality": entry["analysis_data"]
                .get("analysis_quality", "Good")
                .title(),
                "Confidence": f"{entry['analysis_data'].get('measurement_confidence', 0):.0f}%",
            }
        )

    df = pd.DataFrame(df_data)

    st.dataframe(df, width='stretch', hide_index=True)

    # Trend analysis
    st.markdown("#### 📈 Clinical Trend Analysis")

    fig = go.Figure()

    dates = [entry["date"] for entry in st.session_state.patient.pelvic_history]
    posture_scores = [entry["posture_score"] for entry in st.session_state.patient.pelvic_history]
    pelvic_angles = [
        abs(entry["left_pelvic_angle"]) for entry in st.session_state.patient.pelvic_history
    ]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=posture_scores,
            mode="lines+markers",
            name="Posture Score",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pelvic_angles,
            mode="lines+markers",
            name="Pelvic Deviation (°)",
            line=dict(color="#f59e0b", width=3),
            marker=dict(size=8),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="📊 Comprehensive Clinical Progress Tracking",
        xaxis_title="Assessment Date",
        yaxis_title="Posture Score (0-10)",
        yaxis2=dict(
            title="Pelvic Deviation (degrees)",
            overlaying="y",
            side="right",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#0f172a",
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(249, 250, 251, 0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
    )

    st.plotly_chart(fig, width='stretch')

    # Clinical summary statistics
    st.markdown("#### 📋 Clinical Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = np.mean(posture_scores)
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Average Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        score_trend = (
            posture_scores[-1] - posture_scores[0]
            if len(posture_scores) > 1
            else 0
        )
        trend_color = (
            "#10b981"
            if score_trend > 0
            else "#ef4444"
            if score_trend < 0
            else "#64748b"
        )
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {trend_color}">{score_trend:+.1f}</div>
            <div class="metric-label">Score Change</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        best_score = max(posture_scores)
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{best_score:.1f}</div>
            <div class="metric-label">Best Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        consistency = 100 - (np.std(posture_scores) / np.mean(posture_scores) * 100)
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value">{consistency:.0f}%</div>
            <div class="metric-label">Consistency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_reports_page():
    """Reports page that shows both JSON data and the full clinical text report."""
    st.markdown(
        """
    <div class="medical-header">
        <h1>📋 Clinical Data, JSON & Text Reports</h1>
        <p>View and export the complete structured patient record and the formatted clinical report</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.patient:
        st.markdown(
            """
        <div class="alert-warning">
            ⚠️ <strong>No Patient Selected</strong><br>
            Please register a patient and perform clinical assessments before viewing reports.
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    patient_json = patient_to_dict(st.session_state.patient)
    pelvic_history = st.session_state.patient.pelvic_history

    st.markdown("### 👤 Patient Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
        <div class="dashboard-card">
            <h3>📋 Demographics</h3>
            <p><strong>Name:</strong> {patient_json['name']}</p>
            <p><strong>ID:</strong> {patient_json['patient_id']}</p>
            <p><strong>Age:</strong> {patient_json['age']}</p>
            <p><strong>Gender:</strong> {patient_json.get('gender', 'Not specified')}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="dashboard-card">
            <h3>🏥 Medical Info</h3>
            <p><strong>Condition:</strong> {patient_json['condition']}</p>
            <p><strong>Referring Physician:</strong> {patient_json.get('referring_physician', 'Not specified')}</p>
            <p><strong>Start Date:</strong> {patient_json['therapy_start_date']}</p>
            <p><strong>Pain Level:</strong> {patient_json.get('current_pain_level', 0)}/10</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        last_entry = pelvic_history[-1] if pelvic_history else None
        last_score = last_entry["posture_score"] if last_entry else "N/A"

        st.markdown(
            f"""
        <div class="dashboard-card">
            <h3>📈 Analysis Overview</h3>
            <p><strong>Total Assessments:</strong> {len(pelvic_history)}</p>
            <p><strong>Last Posture Score:</strong> {last_score}</p>
            <p><strong>Sessions:</strong> {len(patient_json.get('sessions', []))}</p>
            <p><strong>Data Format:</strong> JSON + Text</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    selected_entry = None
    if pelvic_history:
        st.markdown("### 🧪 Select Assessment for Clinical Report")

        options = [
            f"Assessment #{i+1} – {entry['date'].strftime('%Y-%m-%d %H:%M')} – Score: {entry['posture_score']:.1f}/10"
            for i, entry in enumerate(pelvic_history)
        ]
        idx = st.selectbox(
            "Choose an assessment to generate the clinical text report from:",
            options=list(range(len(options))),
            format_func=lambda i: options[i],
        )

        selected_entry = pelvic_history[idx]
        pelvic_analysis = selected_entry.get("analysis_data", {})

        clinical_report_text = report_generator._generate_text_report(
            patient_json, pelvic_analysis
        )
    else:
        clinical_report_text = None

    st.markdown("### 📄 Clinical Report")

    if not pelvic_history:
        st.info(
            "No pelvic assessments available yet. Perform an analysis to generate a clinical report."
        )
    else:
        st.text_area(
            "Clinical Report [Don't Edit]",
            value=clinical_report_text,
            height=650,
        )

        # ⬇️ JSON DOWNLOAD ONLY (NO VISUAL EXPOSURE)
        json_bytes = json.dumps(patient_json, indent=2).encode("utf-8")
        st.download_button(
            label="📥 Download Raw Clinical Data (JSON)",
            data=json_bytes,
            file_name=f"patient_{patient_json['patient_id']}_clinical_record.json",
            mime="application/json",
            width='stretch',
            help="For audit, research, or system integration only. Not intended for routine clinical review.",
        )

        # ⬇️ Text Report Download
        report_bytes = clinical_report_text.encode("utf-8")
        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=report_bytes,
            file_name=f"IntelliFit_Clinical_Report_{patient_json['patient_id']}.txt",
            mime="text/plain",
            width='stretch',
        )


# 🚀 MAIN APPLICATION CONTROLLER
def main():
    """Main application controller with proper routing"""
    try:
        selected_page = render_professional_sidebar()

        if selected_page == "dashboard":
            render_dashboard_page()
        elif selected_page == "patients":
            render_patient_management_page()
        elif selected_page == "analysis":
            render_pelvic_analysis_page()
        elif selected_page == "reports":
            render_reports_page()
        else:
            render_dashboard_page()

        st.markdown("---")
        st.markdown(
            """
        <div class="app-footer">
             <strong>🏥 IntelliFit Pro</strong> - Professional Healthcare Analytics Platform<br>
            <small>Medical-Grade AI Analysis | Clinical Documentation | Evidence-Based Recommendations</small><br>
            <small>© 2025 IntelliFit Technologies. Licensed for Professional Medical Use.</small><br>
            <small><em>This software is designed for use by qualified healthcare professionals as a clinical assessment tool.</em></small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("🔧 Please restart the application or contact system administrator.")

# 🚀 APPLICATION ENTRY POINT
if __name__ == "__main__":
    main()
