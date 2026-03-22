import streamlit as st
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Disease Akinator",
    page_icon="🧞",
    layout="centered"
)

# =====================================================
# CONSTANTS
# =====================================================
MAX_QUESTIONS = 25
CONFIDENCE_THRESHOLD = 0.90

# =====================================================
# GLOBAL CSS
# =====================================================
st.markdown("""
<style>
/* Card style for each section */
.section-card {
    background: #f8f9fa;
    border: 1.5px solid #e0e0e0;
    border-radius: 16px;
    padding: 28px 24px 20px 24px;
    margin-bottom: 18px;
    text-align: center;
    transition: box-shadow 0.2s;
}
.section-card:hover {
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
}
.section-icon {
    font-size: 48px;
    margin-bottom: 10px;
}
.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
}
.section-desc {
    font-size: 14px;
    color: #555;
    margin-bottom: 16px;
}
.badge-done {
    background: #e8f5e9;
    color: #2e7d32;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 10px;
}
.badge-soon {
    background: #e3f2fd;
    color: #1565c0;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 10px;
}
/* Answer buttons */
div.stButton > button {
    background: #f5f5f5 !important;
    color: #000 !important;
    border: 2px solid #cfcfcf !important;
    border-radius: 14px !important;
    padding: 14px !important;
    font-size: 18px !important;
}
div.stButton > button:hover {
    background: #e9e9e9 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD SYMPTOM MODEL ASSETS
# =====================================================
@st.cache_resource
def load_symptom_assets():
    model = joblib.load("final_ensemble.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

# =====================================================
# SESSION STATE INIT
# =====================================================
def init_session():
    if "page" not in st.session_state:
        st.session_state.page = "home"

init_session()

# =====================================================
# HELPERS — SYMPTOM MODULE
# =====================================================
def init_symptom_state(num_features):
    st.session_state.user_state = np.zeros((1, num_features))
    st.session_state.asked = []
    st.session_state.done = False
    st.session_state.round_questions = 0
    st.session_state.show_checkpoint = False
    st.session_state.final_probs = None

def select_next_question(features):
    for i, f in enumerate(features):
        if f not in st.session_state.asked:
            return i, f
    return None, None

def update_user_state(idx, answer):
    mapping = {"Yes": 1.0, "Probably": 0.8, "Probably Not": 0.2, "No": 0.0}
    st.session_state.user_state[0, idx] = mapping[answer]

def get_probs(model):
    return model.predict_proba(st.session_state.user_state)[0]

# =====================================================
# PAGE: HOME
# =====================================================
def page_home():
    st.markdown("""
    <h1 style='text-align:center; font-size:2.2rem;'>🧞 Disease Akinator</h1>
    <p style='text-align:center; color:#666; font-size:15px; margin-bottom:28px;'>
        AI-powered multi-modal disease diagnosis system
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ---- Section 1: Symptom Prediction ----
    with col1:
        st.markdown("""
        <div class="section-card">
            <div class="section-icon">🩺</div>
            <div class="badge-done">✓ Available</div>
            <div class="section-title">Symptom Checker</div>
            <div class="section-desc">Answer a few questions about your symptoms and I'll guess your disease — Akinator style.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Diagnosis", key="btn_symptom", use_container_width=True):
            st.session_state.page = "symptom"
            st.rerun()

    # ---- Section 2: Image Classification ----
    with col2:
        st.markdown("""
        <div class="section-card">
            <div class="section-icon">🔬</div>
            <div class="badge-soon">Coming Soon</div>
            <div class="section-title">Skin Disease Scanner</div>
            <div class="section-desc">Upload a photo of a skin condition and our CNN model will classify the disease.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload Image", key="btn_image", use_container_width=True):
            st.session_state.page = "image"
            st.rerun()

    # ---- Section 3: Doctor Portal ----
    with col3:
        st.markdown("""
        <div class="section-card">
            <div class="section-icon">👨‍⚕️</div>
            <div class="badge-soon">Coming Soon</div>
            <div class="section-title">Doctor Portal</div>
            <div class="section-desc">Medical professionals can contribute new disease data — symptoms or images — to improve the model.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Doctor Login", key="btn_doctor", use_container_width=True):
            st.session_state.page = "doctor"
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <p style='text-align:center; font-size:12px; color:#aaa;'>
        ⚠️ For educational purposes only. Always consult a qualified medical professional.
    </p>
    """, unsafe_allow_html=True)

# =====================================================
# PAGE: SYMPTOM PREDICTION
# =====================================================
def page_symptom():
    model, features = load_symptom_assets()
    num_features = len(features)

    # Init state if coming fresh from home
    if "user_state" not in st.session_state:
        init_symptom_state(num_features)

    # Back button
    if st.button("← Back to Home"):
        # Clear symptom state
        for key in ["user_state", "asked", "done", "round_questions",
                    "show_checkpoint", "final_probs"]:
            st.session_state.pop(key, None)
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h2 style='text-align:center;'>🧞 Symptom Checker</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666;'>Answer the questions and I'll guess your disease.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # --- QUESTION FLOW ---
    if not st.session_state.done and not st.session_state.show_checkpoint:
        q_index, question = select_next_question(features)

        if question is None:
            st.session_state.show_checkpoint = True
            st.rerun()

        # Progress indicator
        progress = len(st.session_state.asked) / MAX_QUESTIONS
        st.progress(progress, text=f"Question {len(st.session_state.asked) + 1} of {MAX_QUESTIONS}")

        st.markdown(f"""
        <div style="background:#f4f6fb; padding:1.5rem; border-radius:1rem;
                    text-align:center; font-size:1.2rem; font-weight:600; margin:12px 0;">
            <span style="color:#e53935;">Are you experiencing: {question}</span>?
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        answer = None
        if st.button("✅ Yes",         use_container_width=True): answer = "Yes"
        elif st.button("🤔 Probably",       use_container_width=True): answer = "Probably"
        elif st.button("❓ Probably Not",   use_container_width=True): answer = "Probably Not"
        elif st.button("❌ No",          use_container_width=True): answer = "No"

        if answer:
            update_user_state(q_index, answer)
            st.session_state.asked.append(question)
            st.session_state.round_questions += 1

            # Predict directly (same as original working code)
            probs = model.predict_proba(st.session_state.user_state)[0]
            st.session_state.final_probs = probs

            if probs.max() >= CONFIDENCE_THRESHOLD:
                st.session_state.done = True
            elif st.session_state.round_questions >= MAX_QUESTIONS:
                st.session_state.show_checkpoint = True

            st.rerun()

    # --- CHECKPOINT ---
    elif st.session_state.show_checkpoint and not st.session_state.done:
        probs = st.session_state.final_probs
        idx = probs.argmax()
        st.markdown("## 🔎 Best guess so far")
        st.markdown(f"""
        <div style="background:#fff3e0; padding:1.5rem; border-radius:1rem; text-align:center;">
            <h2>{model.classes_[idx]}</h2>
            <h3>Confidence: {probs[idx]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Is this correct?")
        if st.button("✅ Yes, this is correct!", use_container_width=True):
            st.session_state.done = True
            st.session_state.show_checkpoint = False
            st.rerun()
        if st.button("❌ No, continue asking", use_container_width=True):
            st.session_state.round_questions = 0
            st.session_state.show_checkpoint = False
            st.rerun()

    # --- FINAL RESULT ---
    else:
        probs = st.session_state.final_probs
        order = probs.argsort()[::-1]
        st.markdown("## 🎯 Diagnosis Complete!")
        st.markdown(f"""
        <div style="background:#e8f5e9; padding:1.5rem; border-radius:1rem; text-align:center;">
            <h2>🩺 {model.classes_[order[0]]}</h2>
            <h3>Confidence: {probs[order[0]]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # Show top 3 alternatives
        st.markdown("#### Other possibilities:")
        for i in range(1, min(4, len(order))):
            st.markdown(f"- **{model.classes_[order[i]]}** — {probs[order[i]]*100:.1f}%")

        st.warning("⚠️ This prediction is for educational purposes only. Please consult a medical professional.")

        if st.button("🔄 Start New Diagnosis", use_container_width=True):
            init_symptom_state(num_features)
            st.rerun()

        if st.button("← Back to Home", use_container_width=True):
            init_symptom_state(num_features)
            st.session_state.page = "home"
            st.rerun()

# =====================================================
# PAGE: IMAGE CLASSIFICATION (placeholder — Part 2)
# =====================================================
def page_image():
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h2 style='text-align:center;'>🔬 Skin Disease Scanner</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.info("🚧 **Coming Soon!** The CNN-based image classification module is currently being trained. Check back soon.")

    st.markdown("""
    <div style="text-align:center; padding:40px; background:#f8f9fa; border-radius:16px; margin-top:20px;">
        <div style="font-size:60px;">🔬</div>
        <h3 style="color:#888;">Image Classification Module</h3>
        <p style="color:#aaa; font-size:14px;">
            Upload a skin image → ResNet50 CNN → Disease prediction<br>
            Trained on DermNet NZ dataset (23 classes, 19,500 images)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- This section will be activated in Part 2 ---
    # uploaded_file = st.file_uploader("Upload skin image", type=["jpg","jpeg","png"])
    # if uploaded_file:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption="Uploaded Image", use_column_width=True)
    #     if st.button("Classify Disease"):
    #         result = classify_skin_image(image)
    #         st.success(f"Predicted: {result}")

# =====================================================
# PAGE: DOCTOR PORTAL (placeholder — Part 3)
# =====================================================
def page_doctor():
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h2 style='text-align:center;'>👨‍⚕️ Doctor Portal</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.info("🚧 **Coming Soon!** The doctor contribution portal is under development.")

    st.markdown("""
    <div style="text-align:center; padding:40px; background:#f8f9fa; border-radius:16px; margin-top:20px;">
        <div style="font-size:60px;">👨‍⚕️</div>
        <h3 style="color:#888;">Doctor Contribution Portal</h3>
        <p style="color:#aaa; font-size:14px;">
            Verified doctors can:<br>
            • Upload skin images + label disease<br>
            • Add symptom → disease mappings<br>
            • Data stored in MySQL for model retraining
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- This section will be activated in Part 3 ---
    # st.text_input("Full Name")
    # st.number_input("Age", min_value=25, max_value=80)
    # st.selectbox("Degree", ["MBBS", "MD", "MS", "DM", "MCh"])
    # ...

# =====================================================
# ROUTER
# =====================================================
page = st.session_state.get("page", "home")

if page == "home":
    page_home()
elif page == "symptom":
    page_symptom()
elif page == "image":
    page_image()
elif page == "doctor":
    page_doctor()
