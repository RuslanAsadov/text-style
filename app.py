import streamlit as st


st.set_page_config(
    page_title="Определение стиля текста",
    page_icon="📝",
    layout="centered",
)

st.title("Определение стиля текста")
st.write("Введите текст и нажмите кнопку, чтобы инициировать анализ. ")

user_text = st.text_area("Текст для анализа:", height=200)

if st.button("Определить стиль"):
    cleaned_text = user_text.strip()
    if cleaned_text:
        st.info(
            "Анализ будет добавлен на следующем шаге. "
            "Пока что отображается только введённый текст."
        )
        st.write(cleaned_text)
    else:
        st.warning("Пожалуйста, введите текст для анализа.")
