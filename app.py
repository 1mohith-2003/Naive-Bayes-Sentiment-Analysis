import pickle
import streamlit as st
import text_preprocessing  # Required when loading the saved vectorizer

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    try:
        with open("model_name.txt", "r") as file:
            model_name = file.read().strip()
    except FileNotFoundError:
        model_name = "Naive Bayes"
    return model, vectorizer, model_name

model, vectorizer, model_name = load_artifacts()

st.title("Customer Review Sentiment Analyzer")
st.write("Enter a customer review to predict whether the sentiment is positive or negative.")
st.caption(f"Model: {model_name}")

review = st.text_area(
    "Write your review",
    placeholder="Example: The food was fantastic and the service was excellent!",
    height=130
)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        features = vectorizer.transform([review])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index]) * 100

        if prediction == 1:
            st.success("Positive review")
            st.markdown("### 😊 Positive sentiment")
        else:
            st.error("Negative review")
            st.markdown("### 🙁 Negative sentiment")

        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption("Confidence is the model's estimated class probability, not a guarantee.")

st.divider()
st.subheader("Try sample reviews")
col1, col2 = st.columns(2)
with col1:
    if st.button("Use positive sample"):
        st.session_state["sample_review"] = "The food was fantastic and the service was excellent."
with col2:
    if st.button("Use negative sample"):
        st.session_state["sample_review"] = "Worst service ever. The food was cold and terrible."

if "sample_review" in st.session_state:
    st.info(f"Sample review: {st.session_state['sample_review']}")
    st.caption("Copy the sample into the review box above to analyze it.")
