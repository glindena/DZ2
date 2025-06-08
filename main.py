import os
import pandas as pd
import streamlit as st

from src.utils import prepare_data, train_model, read_model

st.set_page_config(
    page_title="Прогноз стоимости недвижимости",
)


st.header("Обучение модели")

train_data = prepare_data()
train_model(train_data)
st.success("Модель обучена и сохранена!")

st.header("Предсказание стоимости недвижимости")
model = read_model('rf_fitted.pkl')

if model:
    st.write("Введите параметры объекта:")

    total_square = st.slider("Площадь (м²)", 30, 1000, 55)

    rooms = st.selectbox("Количество комнат",
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    )

    city = st.selectbox("Город", options=["Балашиха", "Видное", "Дзержинский",
                                          "Долгопрудный", "Ивантеевка", "Королёв",
                                          "Котельники", "Красногорск", "Лобня",
                                          "Лыткарино", "Люберцы", "Москва", "Московский",
                                          "Мытищи", "Одинцово", "Подольск", "Пушкино",
                                          "Реутов", "Химки", "Щербинка", "Щёлково"])

    input_DF = pd.DataFrame(        {
           "total_square": [total_square],
           "rooms": [int(rooms)],
           "city": [city]
    })

    input_df_encoded = pd.get_dummies(input_DF, drop_first=False)

    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    input_df_encoded = input_df_encoded[model_features]

    if st.button("Предсказать цену"):
        predicted_price = model.predict(input_df_encoded)[0]
        st.success(f"Предполагаемая цена: {predicted_price:.2f} рублей")
else:
    st.info("Загрузите данные и обучите модель для предсказаний.")