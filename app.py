import streamlit as st
from transformers import pipeline
import pandas as pd


st.set_page_config(
    page_title="Определение стиля текста",
    page_icon="📝",
    layout="centered",
)


@st.cache_resource
def load_classifier():
    """Загружает модель один раз для повторного использования между запросами."""
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


labels = [
    "научный стиль — текст с терминами, строгой структурой и анализом",
    "художественный стиль — текст с образными описаниями, эмоциями и метафорами",
    "публицистический стиль — текст с рассуждениями о событиях, общественной жизнью и призывами",
    "разговорный стиль — текст с простыми словами, повседневной речью и неформальным общением",
]

examples = {
    "Научный текст": "В данной статье рассматриваются методы оптимизации градиентного спуска для задач глубокого обучения.",
    "Художественный текст": "Ветер шептал в кронах деревьев, словно делился своими тайнами, а луна, бледная и одинокая, плыла среди звёзд, освещая тропинку к дому.",
    "Публицистический текст": "Сегодня мир стоит на пороге глобальных перемен, и каждый из нас должен осознавать свою ответственность.",
    "Разговорный текст": "Да ну, брось, зачем тебе туда идти? Там скучно, лучше давай у меня соберёмся, чай попьём и просто пообщаемся.",
}


def detect_style(text: str):
    """Запускает zero-shot классификацию и возвращает результаты."""
    classifier = load_classifier()
    result = classifier(text, candidate_labels=labels)
    return result


st.title("Определение стиля текста")
st.write(
    "Введите собственный текст или выберите пример, чтобы определить, относится ли он к научному, художественному, публицистическому или разговорному стилю."
)

example_choice = st.selectbox(
    "Примеры текста:",
    options=["— Ввести свой текст —"] + list(examples.keys()),
)

default_text = "" if example_choice == "— Ввести свой текст —" else examples[
    example_choice]

user_text = st.text_area("Текст для анализа:", value=default_text, height=200)

analyze = st.button("Определить стиль")

if analyze:
    cleaned_text = user_text.strip()
    if not cleaned_text:
        st.warning("Пожалуйста, введите текст или выберите пример для анализа.")
    else:
        with st.spinner("Определяем стиль..."):
            result = detect_style(cleaned_text)

        best_label = result["labels"][0]
        best_score = result["scores"][0]
        short_label = best_label.split("—")[0].strip()

        st.success(
            f"Определённый стиль: **{short_label}** ({best_score * 100:.1f}%)")

        details = pd.DataFrame(
            {
                "Стиль": [label.split("—")[0].strip() for label in result["labels"]],
                "Описание": result["labels"],
                "Уверенность (%)": [round(score * 100, 2) for score in result["scores"]],
            }
        )
        st.dataframe(details, use_container_width=True)

st.caption(
    "Модель использует zero-shot классификацию (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).")
