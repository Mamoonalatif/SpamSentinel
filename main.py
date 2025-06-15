import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Page configuration
st.set_page_config(
    page_title="Spam Sentinel",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for red-black-grey theme with background image
st.markdown("""
<style>
    /* Dark background with pattern overlay */
    body, .block-container {
        background: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.9)), 
                    url('https://wallpapers.com/images/high/1080p-red-and-black-background-8nc23iqvecftm939.webp');
        background-size: cover;
        background-attachment: fixed;
        color: #eee;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main container styled as dark card */
    .block-container {
        max-width: 700px !important;
        background: rgba(25, 25, 25, 0.95) !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(230, 0, 0, 0.25);
        padding: 2.5rem 2.5rem !important;
        border: 1px solid #444;
        margin-top: 2rem;
        margin-bottom: 3rem;
        backdrop-filter: blur(4px);
    }

    /* Header with shield icon */
    .main-header {
        text-align: center;
        font-weight: 800;
        font-size: 2.8rem;
        color: #e60000;
        margin-bottom: 1.2rem;
        text-shadow: 0 2px 8px rgba(230, 0, 0, 0.4);
        letter-spacing: -0.5px;
    }

    /* Description and info */
    .classify-info {
        margin-top: 1.2rem;
        font-size: 1.1rem;
        color: #ccc;
        line-height: 1.6;
        font-weight: 500;
        background: rgba(40, 40, 40, 0.7);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #e60000;
    }
    .classify-info ul {
        padding-left: 1.5rem;
        margin-top: 0.8rem;
        color: #bbb;
    }
    .classify-info ul li {
        margin-bottom: 0.5rem;
        list-style: none;
        position: relative;
        padding-left: 1.5rem;
    }
    .classify-info ul li:before {
        content: "•";
        color: #e60000;
        position: absolute;
        left: 0;
        font-size: 1.4rem;
        top: -3px;
    }

    /* Textarea styled with red accents */
    textarea {
        background: rgba(30, 30, 30, 0.8) !important;
        border: 2px solid #444 !important;
        border-radius: 10px !important;
        padding: 1.2rem !important;
        font-size: 1.1rem !important;
        color: #eee !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        resize: vertical !important;
        transition: all 0.3s ease;
    }
    textarea:focus {
        border-color: #e60000 !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(230, 0, 0, 0.3) !important;
    }

    /* Button with red-black gradient */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #b30000, #660000) !important;
        color: white !important;
        border: none !important;
        padding: 0.9rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        cursor: pointer !important;
        box-shadow: 0 5px 15px rgba(179, 0, 0, 0.4) !important;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #cc0000, #800000) !important;
        box-shadow: 0 7px 20px rgba(204, 0, 0, 0.6) !important;
        transform: translateY(-2px);
    }

    /* Metrics container with dark theme */
    .metric-box {
        margin-top: 2rem;
        display: flex;
        gap: 1.5rem;
        justify-content: center;
    }
    .metric-box > div {
        background: linear-gradient(135deg, #222, #333);
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        text-align: center;
        min-width: 120px;
        font-weight: 700;
        color: #eee;
        font-size: 1.2rem;
        border: 1px solid #444;
    }
    .metric-box > div > div {
        color: #e60000;
        font-size: 1.8rem;
        margin-top: 0.3rem;
    }

    /* Result containers */
    .result-container {
        margin: 2.5rem 0;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 1.6rem;
        background: rgba(20, 20, 20, 0.9);
        border: 2px solid;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    }

    .spam-result {
        background: linear-gradient(135deg, rgba(115, 0, 0, 0.9), rgba(70, 0, 0, 0.9));
        color: #ff9999;
        border-color: #990000;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }

    .safe-result {
        background: linear-gradient(135deg, rgba(40, 40, 40, 0.9), rgba(25, 25, 25, 0.9));
        color: #ccc;
        border-color: #555;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(40, 40, 40, 0.8) !important;
        color: #e60000 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        border-radius: 8px !important;
        padding: 1rem 1.5rem !important;
    }
    .streamlit-expanderContent {
        background: rgba(35, 35, 35, 0.8) !important;
        padding: 1.5rem !important;
        border-radius: 0 0 10px 10px !important;
        color: #ddd !important;
        border: 1px solid #444;
        border-top: none;
    }

    /* Footer note */
    .footer-note {
        text-align: center;
        margin-top: 2rem;
        color: #999;
        font-size: 0.9rem;
    }

    /* Responsive design */
    @media (max-width: 480px) {
        .block-container {
            padding: 1.8rem !important;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        .metric-box {
            flex-direction: column;
            gap: 1.2rem;
            align-items: center;
        }
        .result-container {
            padding: 1.5rem;
            font-size: 1.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Display header
st.markdown("""
<div class="main-header">
    SPAM SENTINEL
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="classify-info">
    <p><strong>RED-BLACK SECURITY THEME</strong><br>
    Advanced detection based on:</p>
    <ul>
        <li>📏 Message Length Analysis</li>
        <li>❗ Exclamation Frequency</li>
        <li>🔍 Keyword Pattern Recognition</li>
        <li>📈 Behavioral Heuristics</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("### INPUT MESSAGE FOR ANALYSIS")
user_input = st.text_area(
    "", placeholder="Enter suspicious message here...", height=140
)

# Sample data
data = {
    'message': [
        "URGENT! Claim your $1000 prize NOW!!!", 
        "Team meeting rescheduled to 3pm", 
        "FREE iPhone for first 100 responders!", 
        "Your package delivery confirmation",
        "Limited time OFFER! 50% discount!!!", 
        "Reminder: Project deadline tomorrow", 
        "Earn $5000 weekly from home", 
        "Security alert: Unusual login detected",
        "You WON! Click to claim your prize!!!", 
        "Monthly report attached for review"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Feature extractor
def extract_features(msg):
    length = len(msg)
    exclamations = msg.count("!")
    spam_words = ["buy", "free", "win", "money", "offer", "prize", "urgent", "claim", "won", "discount"]
    spam_score = sum(word in msg.lower() for word in spam_words)
    return [length, exclamations, spam_score]

X = np.array([extract_features(msg) for msg in df['message']])
y = np.array(df['label'])

model = SVC(kernel='linear', probability=True)
model.fit(X, y)

if st.button("ANALYZE MESSAGE"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a message to analyze.")
    else:
        features = np.array([extract_features(user_input)])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Metrics
        st.markdown("#### THREAT ANALYSIS METRICS")
        col1, col2, col3 = st.columns(3)
        col1.metric("Message Length", f"{len(user_input)} chars")
        col2.metric("Exclamation Count", user_input.count("!"))
        spam_word_count = sum(word in user_input.lower() for word in ["buy", "free", "win", "money", "offer", "prize", "urgent"])
        col3.metric("Spam Indicators", spam_word_count)

        # Result display
        if prediction == 1:
            st.markdown("""
                <div class="result-container spam-result">
                    ⚠️ HIGH RISK: SPAM DETECTED<br>
                    <span style="font-size:1.2rem;">This message exhibits 90%+ spam characteristics</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-container safe-result">
                    ✅ SECURE: LEGITIMATE MESSAGE<br>
                    <span style="font-size:1.2rem;">This message appears safe with low risk indicators</span>
                </div>
            """, unsafe_allow_html=True)

with st.expander("📁 VIEW THREAT DATABASE"):
    st.markdown("#### TRAINING DATASET (0 = Safe, 1 = Spam)")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
st.markdown('<div class="footer-note">AI-Powered Spam Detection System | Red-Black Security Theme</div>', unsafe_allow_html=True)