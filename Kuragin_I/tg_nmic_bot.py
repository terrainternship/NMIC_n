import logging
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ConversationHandler, MessageHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
import os
import numpy as np
from keras.models import load_model
import json
import asyncio 
# Загрузка .env и токена
load_dotenv()
TOKEN = os.environ.get('TOKEN')

# Настройка логгирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Загрузка и конфигурация модели
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        logging.info("Модель успешно загружена.")
        return model
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {str(e)}")
        return None

# Укажите путь к вашей модели
model_path = 'nmic_model.h5'
loaded_model = load_saved_model(model_path)
# Проверка работы модели (удалите в финальной версии)
if loaded_model:
    loaded_model.summary()

# Определение шагов в диалоге
RACE, GENDER, AGE, SMOKING, ECOG, TUMOR_LOAD, KRAS, P53, STK11, KEAP1, TIME_FROM_CHT, MOLECULAR_STATUS, PDL1_STATUS, PATIENT_PREFERENCE, EXPERTS_RESPONSE, CONFIRMATION = range(16) 
# Словари порогов для каждого вопроса
# (замените default_threshold_value на значение по умолчанию, если вам нужно)
RACE_THRESHOLD = {'1': 1, '2': 2, '3': 3}
GENDER_THRESHOLD = {'1': 1, '2': 2}
AGE_THRESHOLD = {'1': 1, '2': 2}
SMOKING_THRESHOLD = {'1': 1, '2': 2, '3': 3}
ECOG_THRESHOLD = {'1': 1, '2': 2}
TUMOR_LOAD_THRESHOLD = {'1': 1, '2': 2}
KRAS_THRESHOLD = {'1': 1, '2': 2}
P53_THRESHOLD = {'1': 1, '2': 2}
STK11_THRESHOLD = {'1': 1, '2': 2}
KEAP1_THRESHOLD = {'0': 1, '2': 2}
TIME_FROM_CHT_THRESHOLD = {'1': 1, '2': 2, '3': 3}
MOLECULAR_STATUS_THRESHOLD = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, }
PDL1_STATUS_THRESHOLD = {'1': 1, '2': 2, '3': 3, '4': 4}
PATIENT_PREFERENCE_THRESHOLD = {'1': 1, '2': 2}
EXPERTS_RESPONSE_THRESHOLD = {'1': 1, '2': 2, '3': 3, '4': 4}

prediction_labels = {
    1: "Дурвалумаб",
    2: "Наблюдение",
    3: "Осимертиниб",
    4: "Алектиниб"
}
def update_user_data(user_data, question, value):
    user_data[question.lower()] = value

async def start(update, context):
    keyboard = [
        [InlineKeyboardButton('Азиатская', callback_data='1'),
         InlineKeyboardButton('Европейская', callback_data='2'),
         InlineKeyboardButton('Другая', callback_data='3')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Выберите Вашу расу:', reply_markup=reply_markup)
    context.user_data['race'] = None

    return RACE 

async def gender(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'race', query.data)
    keyboard = [
        [InlineKeyboardButton('Мужской', callback_data='1'),
         InlineKeyboardButton('Женский', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text('Выберите Ваш пол:', reply_markup=reply_markup)
    context.user_data['gender'] = query.data
   
    return GENDER 

async def age(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'gender', query.data)
    keyboard = [
        [InlineKeyboardButton('До 70', callback_data='1'),
         InlineKeyboardButton('Более 70', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите Ваш возраст:', reply_markup=reply_markup)
    context.user_data['age'] = query.data
    
    return AGE

async def smoking(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'age', query.data)
    keyboard = [
        [InlineKeyboardButton('В настоящее время', callback_data='1')],
        [InlineKeyboardButton('Курение в прошлом (бросил более 1 мес до 1 года)', callback_data='2')],
        [InlineKeyboardButton('Курение в прошлом (бросил более 1 года)', callback_data='3')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите Ваш статус курения:', reply_markup=reply_markup)
    context.user_data['smoking'] = query.data
    
    return SMOKING

async def ecog(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'smoking', query.data)
    keyboard = [
        [InlineKeyboardButton('0-1', callback_data='1'),
         InlineKeyboardButton('2', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите Ваш статус ECOG:', reply_markup=reply_markup)
    context.user_data['ecog'] = query.data
    
    return ECOG

async def tumor_load(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'ecog', query.data)
    keyboard = [
        [InlineKeyboardButton('Да', callback_data='1'),
         InlineKeyboardButton('нет', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Есть опухолевая нагрузка? (симптомная опухоль):', reply_markup=reply_markup)
    context.user_data['tumor_load'] = query.data
    
    return TUMOR_LOAD

async def kras(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'tumor_load', query.data)
    keyboard = [
        [InlineKeyboardButton('Да', callback_data='1'),
         InlineKeyboardButton('нет', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Имеются ли Ко-мутации KRAS :', reply_markup=reply_markup)
    context.user_data['kras'] = query.data
    
    return KRAS

async def p53(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'kras', query.data)
    keyboard = [
        [InlineKeyboardButton('Да', callback_data='1'),
         InlineKeyboardButton('нет', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Имеются ли Ко-мутации p53:', reply_markup=reply_markup)
    context.user_data['p53'] = query.data

    return P53

async def stk11(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'p53', query.data)
    keyboard = [
        [InlineKeyboardButton('Да', callback_data='1'),
         InlineKeyboardButton('нет', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Имеются ли Ко-мутации stk11:', reply_markup=reply_markup)
    context.user_data['stk11'] = query.data
    
    return STK11

async def keap1(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'stk11', query.data)
    keyboard = [
        [InlineKeyboardButton('Да', callback_data='1'),
         InlineKeyboardButton('нет', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Имеются ли Ко-мутации keap1:', reply_markup=reply_markup)
    context.user_data['keap1'] = query.data
    
    return KEAP1

async def time_from_cht(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'keap1', query.data)
    keyboard = [
        [InlineKeyboardButton('до 42 дней', callback_data='1'),
         InlineKeyboardButton('от 43 до 60 дней', callback_data='2'),
         InlineKeyboardButton('более 61 дня', callback_data='3')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите срок от окончания ХЛТ :', reply_markup=reply_markup)
    context.user_data['time_from_cht'] = query.data
    
    return TIME_FROM_CHT

async def molecular_status(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'time_from_cht', query.data)
    keyboard = [
        [InlineKeyboardButton('нет мутаций', callback_data='1')],
        [InlineKeyboardButton('не исследовались', callback_data='2')],
        [InlineKeyboardButton('EGFR редкий вариант', callback_data='3')],
        [InlineKeyboardButton('EGFR ex19', callback_data='4')],
        [InlineKeyboardButton('EGFR ex21', callback_data='5')],
        [InlineKeyboardButton('ALK позитивный', callback_data='6')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите Молекулярный статус (только для неплоскоклеточного рака):', reply_markup=reply_markup)
    context.user_data['molecular_status'] = query.data
    
    return MOLECULAR_STATUS

async def pdl1_status(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'molecular_status', query.data)
    keyboard = [
        [InlineKeyboardButton('Не исследовался', callback_data='1'),
         InlineKeyboardButton('Менее 1%', callback_data='2'),
         InlineKeyboardButton('Более 1%', callback_data='3')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите PD-L1 статус', reply_markup=reply_markup)
    context.user_data['pdl1_status'] = query.data
    return PDL1_STATUS

async def patient_preference(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'pdl1_status', query.data)
    keyboard = [
        [InlineKeyboardButton('Результативность лечения', callback_data='1')],
        [InlineKeyboardButton('Сохранение качества жизни', callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите предпочтение пациента по ответу на терапию:', reply_markup=reply_markup)
    context.user_data['patient_preference'] = query.data
    
    return PATIENT_PREFERENCE  

async def confirmation(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'patient_preference', query.data)

    keyboard = [
        [InlineKeyboardButton('Отправить данные на обработку', callback_data='confirm')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text('Подтвердите отправку данных:', reply_markup=reply_markup)

    return CONFIRMATION

def process_user_responses(user_responses):
    processed_data = np.array(list(map(int, user_responses.values())))
    return processed_data.reshape(1, -1)

def predict_model(user_responses):
    try:
        processed_data = process_user_responses(user_responses)
        predictions = loaded_model.predict(processed_data)
        
        logging.info(f"Processed data: {processed_data}")
        logging.info(f"Model predictions: {predictions}")
        
        # Пример обработки предсказаний
        predicted_index = np.argmax(predictions[0])
        predicted_label = prediction_labels[predicted_index]
        return {"experts_response": predicted_label}
    except Exception as e:
        logging.error(f"Ошибка при предсказании модели: {str(e)}")
        return {"experts_response": "Ошибка"}  # Замените на реальное значение по умолчанию


# Функция для обработки ответов на кнопки
async def button(update, context):
    query = update.callback_query
    user_data = context.user_data
    user_responses = {
        'race': user_data.get('race', '0'),
        'gender': user_data.get('gender', '0'),
        'age': user_data.get('age', '0'),
        'smoking': user_data.get('smoking', '0'),
        'ecog': user_data.get('ecog', '0'),
        'tumor_load': user_data.get('tumor_load', '0'),
        'kras': user_data.get('kras', '0'),
        'p53': user_data.get('p53', '0'),
        'stk11': user_data.get('stk11', '0'),
        'keap1': user_data.get('keap1', '0'),
        'time_from_cht': user_data.get('time_from_cht', '0'),
        'molecular_status': user_data.get('molecular_status', '0'),
        'pdl1_status': user_data.get('pdl1_status', '0'),
        'patient_preference': user_data.get('patient_preference', '0'),
    }
    logging.info(f"user_data: {user_data}")
    logging.info(f"user_responses: {user_responses}")
    # Получите ответы от модели
    predictions = predict_model(user_responses)

    # Формирование ответа пользователю
    response_message = f'Ваше лечение: {predictions["experts_response"]}'
    await query.message.reply_text(response_message)

    logging.info(f"Отправленный пользователю ответ: {response_message}")

    return ConversationHandler.END


def main():
    application = Application.builder().token(TOKEN).build() 
    logging.info('Бот запущен...')

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            RACE: [CallbackQueryHandler(gender)],
            GENDER: [CallbackQueryHandler(age)],
            AGE: [CallbackQueryHandler(smoking)],
            SMOKING: [CallbackQueryHandler(ecog)],
            ECOG: [CallbackQueryHandler(tumor_load)], 
            TUMOR_LOAD: [CallbackQueryHandler(kras)],
            KRAS: [CallbackQueryHandler(p53)],
            P53: [CallbackQueryHandler(stk11)],
            STK11: [CallbackQueryHandler(keap1)],
            KEAP1: [CallbackQueryHandler(time_from_cht)],
            TIME_FROM_CHT:[CallbackQueryHandler(molecular_status)], 
            MOLECULAR_STATUS: [CallbackQueryHandler(pdl1_status)], 
            PDL1_STATUS: [CallbackQueryHandler(patient_preference)], 
            PATIENT_PREFERENCE: [CallbackQueryHandler(confirmation)],
            CONFIRMATION: [CallbackQueryHandler(button)], 
        },
        fallbacks=[CommandHandler('start', start)],
    )
    application.add_handler(conv_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
