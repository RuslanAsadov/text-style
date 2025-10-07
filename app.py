import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Определение стиля текста",
    page_icon="📝",
    layout="centered",
)


@st.cache_resource
def load_classifier():
    """Загружает zero-shot классификатор один раз."""
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


labels = [
    "научный стиль",
    "художественный стиль",
    "публицистический стиль",
    "разговорный стиль",
]

st.title("Определение стиля текста")

user_text = st.text_area("Текст для анализа:", height=200)

if st.button("Определить стиль"):
    cleaned_text = user_text.strip()
    if not cleaned_text:
        st.warning("Пожалуйста, введите текст для анализа.")
    else:
        classifier = load_classifier()
        with st.spinner("Определяем стиль..."):
            result = classifier(cleaned_text, candidate_labels=labels)

        best_label = result["labels"][0]
        best_score = result["scores"][0]

        st.success(
            f"Определённый стиль: **{best_label}** ({best_score * 100:.1f}%)")
