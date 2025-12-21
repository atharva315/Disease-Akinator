import streamlit as st
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG (AKINATOR STYLE)
# =====================================================
st.set_page_config(
    page_title="Disease Akinator",
    page_icon="🧞",
    layout="centered"
)

# =====================================================
# LOAD MODEL FILES (CACHED FOR SPEED)
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("final_ensemble.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, features = load_assets()

NUM_FEATURES = len(features)
CONFIDENCE_THRESHOLD = 0.95

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "user_state" not in st.session_state:
    st.session_state.user_state = np.zeros((1, NUM_FEATURES))
    st.session_state.asked = []
    st.session_state.done = False
    st.session_state.possible_mask = np.ones(NUM_FEATURES, dtype=bool)

# =====================================================
# UI HEADER (GENIE STYLE)
# =====================================================
st.markdown(
    "<h1 style='text-align:center;'>🧞 Disease Akinator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:16px;'>"
    "Answer the questions and I will guess your disease.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def select_next_question():
    """
    Adaptive question selection:
    - Only unasked
    - Only allowed by possible_mask
    - Fast (no entropy recomputation each time)
    """
    for i, f in enumerate(features):
        if f not in st.session_state.asked and st.session_state.possible_mask[i]:
            return i, f
    return None, None


def update_user_state(idx, answer):
    mapping = {
        "Yes": 1.0,
        "Probably": 0.75,
        "Probably Not": 0.25,
        "No": 0.0
    }
    st.session_state.user_state[0, idx] = mapping[answer]

    # Adaptive pruning (fast)
    if answer in ["Yes", "Probably"]:
        st.session_state.possible_mask[idx] = True
    else:
        st.session_state.possible_mask[idx] = False


def check_prediction():
    probs = model.predict_proba(st.session_state.user_state)[0]
    best_idx = probs.argmax()
    return probs[best_idx], model.classes_[best_idx]


# =====================================================
# MAIN AKINATOR FLOW
# =====================================================
if not st.session_state.done:

    q_index, question = select_next_question()

    if question is None:
        st.session_state.done = True
    else:
        # Question card (Akinator style)
        st.markdown(
            f"""
            <div style="
                background-color:#f5f7fb;
                padding:30px;
                border-radius:15px;
                text-align:center;
                font-size:22px;
                font-weight:600;
            ">
            Do you have <span style="color:#ff4b4b;">{question}</span>?
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        if col1.button("✅ Yes", use_container_width=True):
            answer = "Yes"
        elif col2.button("🤔 Probably", use_container_width=True):
            answer = "Probably"
        elif col3.button("😐 Probably Not", use_container_width=True):
            answer = "Probably Not"
        elif col4.button("❌ No", use_container_width=True):
            answer = "No"
        else:
            answer = None

        if answer:
            update_user_state(q_index, answer)
            st.session_state.asked.append(question)

            best_prob, best_disease = check_prediction()

            if best_prob >= CONFIDENCE_THRESHOLD:
                st.session_state.done = True
                st.session_state.final_disease = best_disease
                st.session_state.final_confidence = best_prob

            st.rerun()

# =====================================================
# FINAL RESULT SCREEN
# =====================================================
else:
    st.markdown("## 🎯 I’ve got it!")

    st.markdown(
        f"""
        <div style="
            background:#e8f5e9;
            padding:25px;
            border-radius:15px;
            text-align:center;
        ">
        <h2>{st.session_state.final_disease}</h2>
        <h3>Confidence: {st.session_state.final_confidence*100:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "⚠️ This prediction is for educational purposes only. "
        "Please consult a medical professional."
    )

    if st.button("🔁 Start New Diagnosis"):
        st.session_state.user_state = np.zeros((1, NUM_FEATURES))
        st.session_state.asked = []
        st.session_state.done = False
        st.session_state.possible_mask = np.ones(NUM_FEATURES, dtype=bool)
        st.rerun()
