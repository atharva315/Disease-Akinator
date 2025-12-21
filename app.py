import streamlit as st
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Disease Akinator",
    page_icon="🧠",
    layout="centered"
)

# =====================================================
# GLOBAL CSS (CLEAN AKINATOR STYLE)
# =====================================================
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #ffffff;
}

/* Question card */
.question-card {
    background-color: #f5f6fa;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 25px;
}

/* Highlight symptom */
.symptom {
    color: #ff3b3b;
}

/* Answer buttons */
.answer-btn button {
    background-color: #0f1115 !important;
    color: white !important;
    font-size: 18px !important;
    padding: 14px !important;
    border-radius: 14px !important;
    border: none !important;
    width: 100% !important;
}

/* Hover effect */
.answer-btn button:hover {
    background-color: #1c1f26 !important;
}

/* Result card */
.result-card {
    background-color: #e8f5e9;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
}

/* Mobile fix */
@media (max-width: 600px) {
    .question-card {
        font-size: 18px;
        padding: 20px;
    }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL FILES
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
# SESSION STATE
# =====================================================
if "user_state" not in st.session_state:
    st.session_state.user_state = np.zeros((1, NUM_FEATURES))
    st.session_state.asked = []
    st.session_state.done = False

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div style="display:flex; justify-content:center; align-items:center; gap:10px;">
    <h1>Disease Akinator</h1>
</div>
<p style="text-align:center;">Answer the questions and I will guess your disease.</p>
<hr>
""", unsafe_allow_html=True)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def select_next_question():
    for i, f in enumerate(features):
        if f not in st.session_state.asked:
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

def get_probs():
    return model.predict_proba(st.session_state.user_state)[0]

# =====================================================
# MAIN FLOW
# =====================================================
if not st.session_state.done:

    q_index, question = select_next_question()

    if question is None:
        st.session_state.done = True
    else:
        # Question card
        st.markdown(
            f"""
            <div class="question-card">
                Are you suffering with
                <span class="symptom">{question}</span>?
            </div>
            """,
            unsafe_allow_html=True
        )

        # Answer buttons (BLACK, CLEAR TEXT)
        col1 = st.container()
        with col1:
            if st.markdown('<div class="answer-btn">', unsafe_allow_html=True):
                pass

            if st.button("Yes"):
                answer = "Yes"
            elif st.button("Probably"):
                answer = "Probably"
            elif st.button("Probably Not"):
                answer = "Probably Not"
            elif st.button("No"):
                answer = "No"
            else:
                answer = None

            st.markdown('</div>', unsafe_allow_html=True)

        if answer:
            update_user_state(q_index, answer)
            st.session_state.asked.append(question)

            probs = get_probs()
            best_prob = probs.max()

            if best_prob >= CONFIDENCE_THRESHOLD:
                st.session_state.final_probs = probs
                st.session_state.done = True

            st.rerun()

# =====================================================
# FINAL RESULT
# =====================================================
else:
    probs = st.session_state.final_probs
    order = probs.argsort()[::-1]

    st.markdown("## 🎯 I’ve got it!")

    st.markdown(
        f"""
        <div class="result-card">
            <h2>{model.classes_[order[0]]}</h2>
            <h3>Confidence: {probs[order[0]]*100:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Other Possible Matches")
    for i in order[1:3]:
        st.markdown(
            f"- **{model.classes_[i]}** : {probs[i]*100:.2f}%"
        )

    st.warning(
        "This prediction is for educational purposes only. "
        "Please consult a medical professional."
    )

    if st.button("Start New Diagnosis"):
        st.session_state.clear()
        st.rerun()
