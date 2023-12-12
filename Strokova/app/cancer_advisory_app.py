import streamlit as st
from model import create_model
import numpy as np

def main():
    st.subheader('Рекомендательная система по выбору методик лечения рака легких на основе обследований')

    race_mapping = {'Другая': 0, 'Азиатская': 1, 'Европейская': 2}
    selected_race = st.selectbox('Укажите Вашу расу:', list(race_mapping.keys()))

    gender_mapping = {'Женский': 0, 'Мужской': 1}
    selected_gender = st.selectbox('Укажите Ваш пол:', list(gender_mapping.keys()))

    age_mapping = {'До 70': 0, 'После 70': 1}
    selected_age = st.selectbox('Укажите Ваш возраст:', list(age_mapping.keys()))

    smoking_mapping = {'В настоящее время': 0, 'Курение в прошлом (бросил более 1 года)': 1, 'Курение в прошлом (бросил более 1 месяца до 1 года)': 2}
    selected_smoking = st.selectbox('Укажите статус курения:', list(smoking_mapping.keys()))

    ecog_mapping = {'0-1': 0, '2': 1}
    selected_ecog = st.selectbox('Укажите ECOG:', list(ecog_mapping.keys()))

    tumor_burden_mapping = {'Да': 0, 'Нет': 1}
    selected_tumor_burden = st.selectbox('Присутствует опухолевая нагрузка?', list(tumor_burden_mapping.keys()))

    kras_mapping = {'Да': 0, 'Нет': 1}
    selected_kras = st.selectbox('Присутствуют мутации KRAS?', list(kras_mapping.keys()))

    p53_mapping = {'Да': 0, 'Нет': 1}
    selected_p53 = st.selectbox('Присутствуют мутации p53?', list(p53_mapping.keys()))

    stk11_mapping = {'Да': 0, 'Нет': 1}
    selected_stk11 = st.selectbox('Присутствуют мутации STK111?', list(stk11_mapping.keys()))

    keap1_mapping = {'Да': 0, 'Нет': 1}
    selected_keap1 = st.selectbox('Присутствуют мутации KEAP1?', list(keap1_mapping.keys()))

    hlt_mapping = {'До 42 дней': 0, 'От 43 до 60 дней': 1, 'Более 61 дня': 2}
    selected_hlt = st.selectbox('Укажите срок от окончания ХЛТ:', list(hlt_mapping.keys()))

    molecular_status_mapping = {'ALK позитивный': 0, 'EGFR ex19': 1, 'EGFR ex21': 2, 'EGFR редкий вариант': 3, 'Не исследовался': 4, 'Нет мутаций': 5}
    selected_molecular_status = st.selectbox('Укажите Ваш молекулярный статус:', list(molecular_status_mapping.keys()))

    pd_l1_status_mapping = {'Более 1%': 0, 'Менее 1%': 1, 'Не исследовался': 2}
    selected_pd_l1_status = st.selectbox('PD-L1 статус:', list(pd_l1_status_mapping.keys()))

    preference_mapping = {'Результативность лечения': 0, 'Сохранение качества жизни': 1}
    selected_preference = st.selectbox('Предпочтение по ответу на терапию:', list(preference_mapping.keys()))


    input_data = [
        race_mapping[selected_race],
        gender_mapping[selected_gender],
        age_mapping[selected_age],
        smoking_mapping[selected_smoking],
        ecog_mapping[selected_ecog],
        tumor_burden_mapping[selected_tumor_burden],
        kras_mapping[selected_kras],
        p53_mapping[selected_p53],
        stk11_mapping[selected_stk11],
        keap1_mapping[selected_keap1],
        hlt_mapping[selected_hlt],
        molecular_status_mapping[selected_molecular_status],
        pd_l1_status_mapping[selected_pd_l1_status],
        preference_mapping[selected_preference]
    ]

    expert_response_mapping = {'Алектиниб': 0, 'Дурвалумаб': 1, 'Наблюдение': 2, 'Осимертиниб': 3}
    confidence_mapping = {'50%': 0, '75%': 1, '100%': 2}
    alternative_mapping = {'Алектиниб': 0, 'Дурвалумаб': 1, 'Наблюдение': 2, 'Нет': 3, 'Осимертиниб': 4}

    if st.button('Подтвердить'):
        model = create_model()
        new_data = np.array([input_data])
        predictions = model.predict(new_data)
        rounded_predictions = np.round(predictions)

        expert_response = list(expert_response_mapping.keys())[list(expert_response_mapping.values()).index(rounded_predictions[0][0])]
        confidence = list(confidence_mapping.keys())[list(confidence_mapping.values()).index(rounded_predictions[0][1])]
        alternative = list(alternative_mapping.keys())[list(alternative_mapping.values()).index(rounded_predictions[0][2])]

        st.subheader('Рекомендации:')
        st.write('Ответ эксперта:', expert_response)
        st.write('Уверенность:', confidence)
        st.write('Альтернатива:', alternative)

if __name__ == '__main__':
    main()