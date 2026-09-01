import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# =================================================
# PAGE CONFIGURATION
# =================================================

st.set_page_config(
    page_title="SentimentIQ | AI Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =================================================
# FILE PATHS
# =================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "linear_svm_sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# =================================================
# LOAD MODEL AND VECTORIZER
# =================================================

@st.cache_resource
def load_resources():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


try:

    model, vectorizer = load_resources()

except Exception as error:

    st.error("Unable to load the trained model or vectorizer.")
    st.exception(error)
    st.stop()


# =================================================
# SESSION STATE
# =================================================

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "decision_scores" not in st.session_state:
    st.session_state.decision_scores = None

if "history" not in st.session_state:
    st.session_state.history = []


# =================================================
# CALLBACK FUNCTIONS
# =================================================

def set_positive_example():

    st.session_state.text_input = (
        "This product is amazing and I would definitely "
        "recommend it to everyone."
    )

    st.session_state.prediction = None
    st.session_state.decision_scores = None


def set_neutral_example():

    st.session_state.text_input = (
        "The product arrived yesterday and works as expected."
    )

    st.session_state.prediction = None
    st.session_state.decision_scores = None


def set_negative_example():

    st.session_state.text_input = (
        "This was a disappointing experience and the product "
        "did not work properly."
    )

    st.session_state.prediction = None
    st.session_state.decision_scores = None


def clear_text():

    st.session_state.text_input = ""
    st.session_state.prediction = None
    st.session_state.decision_scores = None


def clear_history():

    st.session_state.history = []


# =================================================
# HELPER FUNCTIONS
# =================================================

def analyze_text(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    scores = model.decision_function(text_vector)

    scores = np.asarray(scores)

    if scores.ndim > 1:
        scores = scores[0]

    return prediction, scores


def get_word_count(text):

    return len(text.split())


def normalize_scores(scores):

    scores = np.asarray(scores, dtype=float)

    if scores.ndim == 0:

        scores = np.array([float(scores)])

    shifted_scores = scores - np.min(scores)

    total = np.sum(shifted_scores)

    if total == 0:

        return np.ones(len(scores)) / len(scores)

    return shifted_scores / total


# =================================================
# CUSTOM CSS
# =================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1300px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 1rem;
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =================================================
# SIDEBAR
# =================================================

with st.sidebar:

    st.title("💬 SentimentIQ")

    st.caption(
        "Machine Learning Powered Sentiment Analysis"
    )

    st.markdown("---")

    st.subheader("🤖 Final Model")

    st.success(
        "Linear Support Vector Machine"
    )

    st.subheader("📝 Text Processing")

    st.info(
        "TF-IDF Vectorization"
    )

    st.subheader("🎯 Sentiment Classes")

    st.write("😊 Positive")
    st.write("😐 Neutral")
    st.write("😞 Negative")

    st.markdown("---")

    st.subheader("📊 Model Performance")

    st.metric(
        "Test Accuracy",
        "78.82%"
    )

    st.metric(
        "Weighted F1 Score",
        "75.99%"
    )

    st.markdown("---")

    st.button(
        "🗑️ Clear Session History",
        use_container_width=True,
        on_click=clear_history
    )


# =================================================
# HEADER
# =================================================

st.title("💬 SentimentIQ")

st.subheader(
    "Machine Learning Powered Sentiment Analysis"
)

st.write(
    "Analyze reviews, comments, and written text using "
    "Natural Language Processing and Machine Learning."
)

st.caption(
    "MACHINE LEARNING • NLP • TEXT CLASSIFICATION"
)

st.markdown("")


# =================================================
# PROJECT METRICS
# =================================================

metric_1, metric_2, metric_3, metric_4 = st.columns(4)


with metric_1:

    st.metric(
        "Models Evaluated",
        "6"
    )


with metric_2:

    st.metric(
        "Sentiment Classes",
        "3"
    )


with metric_3:

    st.metric(
        "Test Accuracy",
        "78.82%"
    )


with metric_4:

    st.metric(
        "Final Model",
        "Linear SVM"
    )


st.markdown("---")


# =================================================
# TEXT INPUT
# =================================================

left_column, right_column = st.columns(
    [2.2, 1],
    gap="large"
)


with left_column:

    st.subheader("📝 Enter Your Text")

    st.text_area(
        "Text Input",
        key="text_input",
        height=220,
        placeholder=(
            "Example: This product is excellent and "
            "I really enjoyed using it!"
        ),
        label_visibility="collapsed"
    )

    user_text = st.session_state.text_input

    count_1, count_2 = st.columns(2)

    with count_1:

        st.caption(
            f"Characters: {len(user_text)}"
        )

    with count_2:

        st.caption(
            f"Words: {get_word_count(user_text)}"
        )


with right_column:

    st.subheader("⚡ Quick Examples")

    st.button(
        "😊 Positive Example",
        use_container_width=True,
        on_click=set_positive_example
    )

    st.button(
        "😐 Neutral Example",
        use_container_width=True,
        on_click=set_neutral_example
    )

    st.button(
        "😞 Negative Example",
        use_container_width=True,
        on_click=set_negative_example
    )

    st.markdown("")

    st.button(
        "🧹 Clear Text",
        use_container_width=True,
        on_click=clear_text
    )


# =================================================
# ANALYZE BUTTON
# =================================================

st.markdown("")

analyze_button = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True,
    type="primary"
)


# =================================================
# PREDICTION
# =================================================

if analyze_button:

    user_text = st.session_state.text_input.strip()

    if not user_text:

        st.warning(
            "Please enter some text before analyzing."
        )

    else:

        with st.spinner(
            "Analyzing sentiment..."
        ):

            prediction, scores = analyze_text(
                user_text
            )

        st.session_state.prediction = prediction
        st.session_state.decision_scores = scores

        history_item = {

            "Text": (
                user_text[:80] + "..."
                if len(user_text) > 80
                else user_text
            ),

            "Sentiment": prediction
        }

        st.session_state.history.insert(
            0,
            history_item
        )

        st.session_state.history = (
            st.session_state.history[:10]
        )


# =================================================
# RESULT SECTION
# =================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    st.markdown("---")

    st.subheader("✨ Analysis Result")

    result_column_1, result_column_2 = st.columns(
        [1, 3]
    )


    with result_column_1:

        st.metric(
            "Predicted Sentiment",
            prediction
        )


    with result_column_2:

        if prediction == "Positive":

            st.success(
                "😊 The submitted text expresses an overall "
                "positive sentiment."
            )

        elif prediction == "Negative":

            st.error(
                "😞 The submitted text expresses an overall "
                "negative sentiment."
            )

        else:

            st.info(
                "😐 The submitted text expresses an overall "
                "neutral sentiment."
            )


    # =============================================
    # DECISION SCORES
    # =============================================

    st.subheader("📊 Model Decision Scores")

    scores = st.session_state.decision_scores

    class_names = list(model.classes_)

    scores = np.asarray(scores)

    if len(scores) == len(class_names):

        normalized_scores = normalize_scores(scores)

        score_dataframe = pd.DataFrame({

            "Sentiment": class_names,

            "Score": normalized_scores * 100

        })

        chart_data = score_dataframe.set_index(
            "Sentiment"
        )

        st.bar_chart(chart_data)

        st.dataframe(
            score_dataframe.style.format({
                "Score": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These values are normalized decision scores from "
            "the Linear SVM model and are not probabilities."
        )


# =================================================
# SESSION HISTORY
# =================================================

if st.session_state.history:

    st.markdown("---")

    st.subheader(
        "🕒 Session Prediction History"
    )

    history_dataframe = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_dataframe,
        use_container_width=True,
        hide_index=True
    )


# =================================================
# TECHNICAL DETAILS
# =================================================

st.markdown("---")

with st.expander(
    "🔬 Technical Details"
):

    st.markdown(
        """
        ### Final Model

        **Linear Support Vector Machine**

        ### Text Processing

        **TF-IDF Vectorization**

        ### Model Performance

        - Test Accuracy: **78.82%**
        - Weighted F1 Score: **75.99%**
        - Models Evaluated: **6**

        ### Workflow

        1. User enters text.
        2. TF-IDF converts the text into numerical features.
        3. The Linear SVM model analyzes the features.
        4. The predicted sentiment is displayed.

        The final Linear SVM model was selected after comparing
        six machine learning models.
        """
    )


# =================================================
# ABOUT PROJECT
# =================================================

with st.expander(
    "ℹ️ About SentimentIQ"
):

    st.markdown(
        """
        SentimentIQ is a Natural Language Processing and Machine
        Learning project that classifies text into three categories:

        - 😊 Positive
        - 😐 Neutral
        - 😞 Negative

        The application uses TF-IDF vectorization to transform
        text into numerical features and a Linear Support Vector
        Machine model to perform sentiment classification.
        """
    )


# =================================================
# FOOTER
# =================================================

st.markdown("---")

st.caption(
    "SentimentIQ • Natural Language Processing • "
    "Machine Learning • Linear SVM"
)