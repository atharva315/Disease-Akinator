import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from db_utils import (
    init_db, register_doctor, login_doctor,
    submit_symptom_contribution, submit_image_contribution,
    get_contribution_stats
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="",
    layout="centered"
)

# =====================================================
# CONSTANTS
# =====================================================
MAX_QUESTIONS = 25
CONFIDENCE_THRESHOLD = 0.87

# =====================================================
# GLOBAL CSS
# =====================================================
st.markdown("""
<style>
.section-card {
    background: #f8f9fa;
    border: 1.5px solid #e0e0e0;
    border-radius: 16px;
    padding: 28px 24px 20px 24px;
    margin-bottom: 18px;
    text-align: center;
    transition: box-shadow 0.2s;
}
.section-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.08); }
.section-icon { font-size: 48px; margin-bottom: 10px; }
.section-title { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
.section-desc { font-size: 14px; color: #555; margin-bottom: 16px; }
.badge-done {
    background: #e8f5e9; color: #2e7d32; font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; display: inline-block; margin-bottom: 10px;
}
.badge-soon {
    background: #e3f2fd; color: #1565c0; font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; display: inline-block; margin-bottom: 10px;
}
div.stButton > button {
    background: #f5f5f5 !important; color: #000 !important;
    border: 2px solid #cfcfcf !important; border-radius: 14px !important;
    padding: 14px !important; font-size: 18px !important;
}
div.stButton > button:hover { background: #e9e9e9 !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD ASSETS
# =====================================================
@st.cache_resource
def load_symptom_assets():
    model    = joblib.load("final_ensemble.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

@st.cache_resource
def load_skin_assets():
    skin_model  = tf.keras.models.load_model("skin_model.h5")
    class_names = joblib.load("class_names.pkl")
    return skin_model, class_names

# =====================================================
# SESSION + DB INIT
# =====================================================
def init_session():
    if "page" not in st.session_state:
        st.session_state.page = "home"

init_session()
init_db()

# =====================================================
# HELPERS — SYMPTOM MODULE
# =====================================================
def init_symptom_state(num_features):
    st.session_state.user_state    = np.zeros((1, num_features))
    st.session_state.asked         = []
    st.session_state.done          = False
    st.session_state.round_questions = 0
    st.session_state.show_checkpoint = False
    st.session_state.final_probs   = None

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
    <h1 style='text-align:center; font-size:2.2rem;'> Disease Predictor</h1>
    <p style='text-align:center; color:#666; font-size:15px; margin-bottom:28px;'>
        AI-powered multi-modal disease diagnosis system
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <div class="section-icon"></div>
            <div class="badge-done">✓ Available</div>
            <div class="section-title">Symptom Checker</div>
            <div class="section-desc">Answer a few questions about your symptoms to predict your disease.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Diagnosis", key="btn_symptom", use_container_width=True):
            st.session_state.page = "symptom"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="section-card">
            <div class="section-icon"></div>
            <div class="badge-done">✓ Available</div>
            <div class="section-title">Skin Disease Scanner</div>
            <div class="section-desc">Upload a photo of a skin condition for classifying your disease.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload Image", key="btn_image", use_container_width=True):
            st.session_state.page = "image"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="section-card">
            <div class="section-icon"></div>
            <div class="badge-done">✓ Available</div>
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
# PAGE: SYMPTOM PREDICTION — UNCHANGED
# =====================================================
def page_symptom():
    model, features = load_symptom_assets()
    num_features = len(features)

    if "user_state" not in st.session_state:
        init_symptom_state(num_features)

    if st.button("← Back to Home"):
        for key in ["user_state", "asked", "done", "round_questions",
                    "show_checkpoint", "final_probs"]:
            st.session_state.pop(key, None)
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h2 style='text-align:center;'>🧞 Symptom Checker</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666;'>Answer the questions and I'll guess your disease.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.done and not st.session_state.show_checkpoint:
        q_index, question = select_next_question(features)
        if question is None:
            st.session_state.show_checkpoint = True
            st.rerun()

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
        if st.button(" Yes",           use_container_width=True): answer = "Yes"
        elif st.button(" Probably",     use_container_width=True): answer = "Probably"
        elif st.button(" Probably Not", use_container_width=True): answer = "Probably Not"
        elif st.button(" No",           use_container_width=True): answer = "No"

        if answer:
            update_user_state(q_index, answer)
            st.session_state.asked.append(question)
            st.session_state.round_questions += 1
            probs = model.predict_proba(st.session_state.user_state)[0]
            st.session_state.final_probs = probs
            if probs.max() >= CONFIDENCE_THRESHOLD:
                st.session_state.done = True
            elif st.session_state.round_questions >= MAX_QUESTIONS:
                st.session_state.show_checkpoint = True
            st.rerun()

    elif st.session_state.show_checkpoint and not st.session_state.done:
        probs = st.session_state.final_probs
        idx   = probs.argmax()
        st.markdown("## 🔎 Best guess so far")
        st.markdown(f"""
        <div style="background:#fff3e0; padding:1.5rem; border-radius:1rem; text-align:center;">
            <h2>{model.classes_[idx]}</h2>
            <h3>Confidence: {probs[idx]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### Is this correct?")
        if st.button(" Yes, this is correct!", use_container_width=True):
            st.session_state.done = True
            st.session_state.show_checkpoint = False
            st.rerun()
        if st.button(" No, continue asking", use_container_width=True):
            st.session_state.round_questions = 0
            st.session_state.show_checkpoint = False
            st.rerun()

    else:
        probs = st.session_state.final_probs
        order = probs.argsort()[::-1]
        st.markdown("##  Diagnosis Complete!")
        st.markdown(f"""
        <div style="background:#e8f5e9; padding:1.5rem; border-radius:1rem; text-align:center;">
            <h2>🩺 {model.classes_[order[0]]}</h2>
            <h3>Confidence: {probs[order[0]]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
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
# HELPER — Image Preprocessing — UNCHANGED
# =====================================================
def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((160, 160))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_skin(pil_image, skin_model, class_names, top_k=5):
    arr   = preprocess_image(pil_image)
    probs = skin_model.predict(arr)[0]
    top_idx = probs.argsort()[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]

# =====================================================
# PAGE: IMAGE CLASSIFICATION — UNCHANGED
# =====================================================
def page_image():
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h2 style='text-align:center;'>🔬 Skin Disease Scanner</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666;'>Upload a clear photo of the affected skin area.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading skin disease model..."):
        skin_model, class_names = load_skin_assets()

    uploaded_file = st.file_uploader(
        "Upload skin image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Take a clear, well-lit photo of the skin condition."
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div style="background:#f4f6fb; padding:14px; border-radius:10px;
                        font-size:13px; color:#1a1a1a;">
                <b>File:</b> {uploaded_file.name}<br>
                <b>Size:</b> {pil_image.size[0]} × {pil_image.size[1]} px<br>
                <b>Model:</b> ResNet50 (DermNet NZ)<br>
                <b>Classes:</b> {len(class_names)} skin conditions
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔬 Classify Disease", use_container_width=True):
            with st.spinner("Analysing image..."):
                results = predict_skin(pil_image, skin_model, class_names, top_k=5)
            top_name, top_conf = results[0]
            conf_color = "#e8f5e9" if top_conf >= 0.6 else "#fff3e0" if top_conf >= 0.35 else "#fce4ec"
            st.markdown(f"""
            <div style="background:{conf_color}; padding:1.5rem; border-radius:1rem;
                        text-align:center; margin:16px 0; color:#1a1a1a;">
                <div style="font-size:14px; color:#555; margin-bottom:4px;">Primary Prediction</div>
                <h2 style="margin:0; color:#1a1a1a;">🩺 {top_name}</h2>
                <h3 style="margin:6px 0 0; color:#1a1a1a;">Confidence: {top_conf*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#### Top 5 Predictions")
            for name, conf in results:
                bar_color = "#4caf50" if conf == top_conf else "#90caf9"
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between;
                                font-size:13px; margin-bottom:3px;">
                        <span>{name}</span><span><b>{conf*100:.1f}%</b></span>
                    </div>
                    <div style="background:#eee; border-radius:6px; height:10px;">
                        <div style="background:{bar_color}; width:{conf*100:.1f}%;
                                    height:10px; border-radius:6px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.warning("⚠️ This is for educational purposes only. Always consult a qualified dermatologist.")
    else:
        st.markdown("""
        <div style="text-align:center; padding:50px; background:#f8f9fa;
                    border-radius:16px; margin-top:10px;">
            <div style="font-size:56px;">📷</div>
            <h3 style="color:#888; margin-top:12px;">No image uploaded yet</h3>
            <p style="color:#aaa; font-size:14px;">
                Supported conditions: Acne, Eczema, Psoriasis, Melanoma,<br>
                Rosacea, Tinea, Urticaria and 16 more skin diseases.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# PAGE: DOCTOR PORTAL — Section 3 (LIVE)
# =====================================================
def page_doctor():
    if st.button("← Back to Home"):
        for key in ["doctor_logged_in", "doctor_data"]:
            st.session_state.pop(key, None)
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h2 style='text-align:center;'>👨‍⚕️ Doctor Portal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666;'>Verified medical professionals only.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.get("doctor_logged_in", False):
        _doctor_auth()
        return

    doctor = st.session_state.doctor_data
    st.markdown(f"""
    <div style="background:#e8f5e9; padding:12px 18px; border-radius:10px;
                color:#1a1a1a; font-size:14px; margin-bottom:16px;">
        ✅ Logged in as <b>Dr. {doctor['full_name']}</b>
        ({doctor['degree']}) · {doctor.get('hospital','') or 'No hospital listed'}
    </div>
    """, unsafe_allow_html=True)

    stats = get_contribution_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Doctors",   stats["doctors"])
    c2.metric("Symptom Entries", stats["symptom_entries"])
    c3.metric("Image Entries",   stats["image_entries"])

    st.markdown("---")
    st.markdown("### What would you like to contribute?")

    tab1, tab2 = st.tabs(["🧬 Add Symptoms → Disease", "📷 Upload Image → Disease"])

    with tab1:
        st.markdown("##### Enter a disease name and its associated symptoms")
        st.markdown("<p style='font-size:13px; color:#888;'>This data will retrain the symptom prediction model.</p>",
                    unsafe_allow_html=True)
        disease_name  = st.text_input("Disease Name *", placeholder="e.g. Dengue Fever", key="sym_disease")
        st.markdown("**Symptoms** — enter one per line *")
        symptoms_text = st.text_area("Symptoms", placeholder="Fever\nHeadache\nJoint pain\nRash\nNausea",
                                     height=180, key="sym_symptoms", label_visibility="collapsed")
        notes_sym     = st.text_area("Additional clinical notes (optional)",
                                     placeholder="Any relevant notes...", height=80, key="sym_notes")

        if st.button(" Submit Symptom Data", use_container_width=True, key="btn_sym"):
            if not disease_name.strip():
                st.error("Please enter a disease name.")
            else:
                symptoms_list = [s for s in symptoms_text.strip().split("\n") if s.strip()]
                if len(symptoms_list) < 2:
                    st.error("Please enter at least 2 symptoms.")
                else:
                    with st.spinner("Saving to database..."):
                        ok, msg = submit_symptom_contribution(
                            doctor_id=doctor["id"], disease_name=disease_name.strip(),
                            symptoms_list=symptoms_list, notes=notes_sym.strip()
                        )
                    if ok:
                        st.success(f" {msg}")
                        st.markdown(f"""
                        <div style="background:#f1f8e9; padding:12px; border-radius:8px;
                                    font-size:13px; color:#1a1a1a; margin-top:8px;">
                            <b>Disease:</b> {disease_name}<br>
                            <b>Symptoms ({len(symptoms_list)}):</b> {', '.join(symptoms_list)}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f" {msg}")

    with tab2:
        st.markdown("##### Upload a labelled skin disease image")
        st.markdown("<p style='font-size:13px; color:#888;'>This data will retrain the image classification model.</p>",
                    unsafe_allow_html=True)
        disease_img  = st.text_input("Disease Name *", placeholder="e.g. Melanoma", key="img_disease")
        uploaded_img = st.file_uploader("Upload skin image (JPG / PNG) *",
                                        type=["jpg", "jpeg", "png"], key="img_upload")

        if uploaded_img:
            col_prev, col_inf = st.columns([1, 1])
            with col_prev:
                st.image(uploaded_img, caption="Preview", use_container_width=True)
            with col_inf:
                pil_prev = Image.open(uploaded_img)
                st.markdown(f"""
                <div style="background:#f4f6fb; padding:12px; border-radius:8px;
                            font-size:13px; color:#1a1a1a;">
                    <b>File:</b> {uploaded_img.name}<br>
                    <b>Size:</b> {pil_prev.size[0]} × {pil_prev.size[1]} px
                </div>
                """, unsafe_allow_html=True)

        notes_img = st.text_area("Clinical notes (optional)",
                                 placeholder="Any relevant observations...", height=80, key="img_notes")

        if st.button("✅ Submit Image", use_container_width=True, key="btn_img"):
            if not disease_img.strip():
                st.error("Please enter the disease name.")
            elif not uploaded_img:
                st.error("Please upload an image.")
            else:
                with st.spinner("Uploading to database..."):
                    uploaded_img.seek(0)
                    ok, msg = submit_image_contribution(
                        doctor_id=doctor["id"], disease_name=disease_img.strip(),
                        image_bytes=uploaded_img.read(), image_name=uploaded_img.name,
                        notes=notes_img.strip()
                    )
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

    st.markdown("---")
    if st.button("🚪 Log Out", key="logout"):
        for key in ["doctor_logged_in", "doctor_data"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── AUTH HELPER ───────────────────────────────────────────────────
def _doctor_auth():
    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        st.markdown("##### Login with your registered email")
        email_login = st.text_input("Email address", key="login_email",
                                    placeholder="doctor@hospital.com")
        if st.button("Login →", use_container_width=True, key="btn_login"):
            if not email_login.strip():
                st.error("Please enter your email.")
            else:
                with st.spinner("Checking..."):
                    doctor = login_doctor(email_login.strip().lower())
                if doctor:
                    st.session_state.doctor_logged_in = True
                    st.session_state.doctor_data = doctor
                    st.success(f"Welcome back, Dr. {doctor['full_name']}!")
                    st.rerun()
                else:
                    st.error("Email not found. Please register first.")

    with tab_register:
        st.markdown("##### Create your doctor account")
        col_a, col_b = st.columns(2)
        with col_a:
            full_name = st.text_input("Full Name *", placeholder="Dr. Rahul Sharma", key="reg_name")
            age       = st.number_input("Age *", min_value=25, max_value=80, value=35, key="reg_age")
            degree    = st.selectbox("Degree *",
                                     ["MBBS","MD","MS","DM","MCh","BDS","MDS","DNB","Other"],
                                     key="reg_degree")
        with col_b:
            hospital  = st.text_input("Hospital / Clinic", placeholder="City Hospital, Pune", key="reg_hospital")
            city      = st.text_input("City", placeholder="Pune", key="reg_city")
            email_reg = st.text_input("Email *", placeholder="doctor@hospital.com", key="reg_email")

        st.markdown("<p style='font-size:12px; color:#aaa;'>* Required fields</p>", unsafe_allow_html=True)

        if st.button("Register →", use_container_width=True, key="btn_register"):
            if not all([full_name.strip(), email_reg.strip()]):
                st.error("Please fill in all required fields.")
            elif "@" not in email_reg:
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Creating account..."):
                    ok, msg, doctor_id = register_doctor(
                        full_name=full_name.strip(), age=int(age), degree=degree,
                        hospital=hospital.strip(), city=city.strip(),
                        email=email_reg.strip().lower()
                    )
                if ok:
                    st.success(f"✅ {msg} You can now log in.")
                else:
                    st.error(f"❌ {msg}")

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
