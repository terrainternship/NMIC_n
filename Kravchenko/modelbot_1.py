import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
import os 
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import asyncio 

# Определение логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # или другой уровень логирования по вашему выбору

# Создание обработчика для вывода сообщений в консоль
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Добавление обработчика к логгеру
logger.addHandler(console_handler)


def load_saved_model(model_path):
    try:
        model = tf.saved_model.load(model_path)
        print("Модель успешно загружена.")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        return None  # Возвращаем None в случае ошибки

# Укажите путь к вашей модели AutoKeras
model_path = 'C:/BOT/S/model'
loaded_model = load_saved_model(model_path)

load_dotenv()
TOKEN = os.environ.get('TOKEN')

# Определение шагов в диалоге
CHOOSING, GENDER, AGE, SMOKING, ECOG, TUMOR_LOAD, KRAS, P53, STK11, KEAP1, TIME_FROM_CHT, MOLECULAR_STATUS, PDL1_STATUS, PATIENT_PREFERENCE, EXPERTS_RESPONSE, COMMENTS = range(16) 

# Словари порогов для каждого вопроса
# (замените default_threshold_value на значение по умолчанию, если вам нужно)
CHOOSING_THRESHOLD = {'0': 0, '2': 1, '1': 2}
GENDER_THRESHOLD = {'1': 0, '0': 1}
AGE_THRESHOLD = {'1': 0, '0': 1}
SMOKING_THRESHOLD = {'0': 0, '2': 1, '1': 2}
ECOG_THRESHOLD = {'0': 0, '1': 1}
TUMOR_LOAD_THRESHOLD = {'0': 0, '1': 1}
KRAS_THRESHOLD = {'0': 0, '1': 1}
P53_THRESHOLD = {'0': 0, '1': 1}
STK11_THRESHOLD = {'0': 0, '1': 1}
KEAP1_THRESHOLD = {'0': 0, '1': 1}
TIME_FROM_CHT_THRESHOLD = {'1': 0, '2': 1, '0': 2}
MOLECULAR_STATUS_THRESHOLD = {'5': 0, '4': 1, '3': 2, '1': 3, '2': 4, '0': 5, '6': 6}
PDL1_STATUS_THRESHOLD = {'2': 0, '1': 1, '0': 2, '3': 3}
PATIENT_PREFERENCE_THRESHOLD = {'0': 0, '1': 1}
EXPERTS_RESPONSE_THRESHOLD = {'1': 0, '2': 1, '3': 2, '0': 3}
COMMENTS_THRESHOLD = {'0': 0, '10': 1, '7': 2, '5': 3, '6': 4, '4': 5, '2': 6, '3': 7, '1': 8, '8': 9, '9': 10}

def update_user_data(user_data, question, value):
    user_data[question.lower()] = value

async def start(update, context):
    keyboard = [
        [InlineKeyboardButton('Азиатская', callback_data='0'),
         InlineKeyboardButton('Европейская', callback_data='2'),
         InlineKeyboardButton('Другая', callback_data='1')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text('Выберите Вашу расу:', reply_markup=reply_markup)

     # Установите начальное значение 'race' в user_data
    context.user_data['race'] = None

    return CHOOSING 

async def gender(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'race', query.data)

    keyboard = [
        [InlineKeyboardButton('Мужской', callback_data='1'),
         InlineKeyboardButton('Женский', callback_data='0')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите Ваш пол:', reply_markup=reply_markup)

    # Установите начальное значение 'gender' в user_data
    context.user_data['gender'] = query.data

    return GENDER 

async def age(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'gender', query.data)

    keyboard = [
        [InlineKeyboardButton('До 70', callback_data='1'),
         InlineKeyboardButton('Более 70', callback_data='0')]
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
        [InlineKeyboardButton('В настоящее время', callback_data='0')],
        [InlineKeyboardButton('Курение в прошлом (бросил более 1 мес до 1 года)', callback_data='2')],
        [InlineKeyboardButton('Курение в прошлом (бросил более 1 года)', callback_data='1')]
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
        [InlineKeyboardButton('0-1', callback_data='0'),
         InlineKeyboardButton('2', callback_data='1')]
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
        [InlineKeyboardButton('Да', callback_data='0'),
         InlineKeyboardButton('нет', callback_data='1')]
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
        [InlineKeyboardButton('Да', callback_data='0'),
         InlineKeyboardButton('нет', callback_data='1')]
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
        [InlineKeyboardButton('Да', callback_data='0'),
         InlineKeyboardButton('нет', callback_data='1')]
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
        [InlineKeyboardButton('Да', callback_data='0'),
         InlineKeyboardButton('нет', callback_data='1')]
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
        [InlineKeyboardButton('Да', callback_data='0'),
         InlineKeyboardButton('нет', callback_data='1')]
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
         InlineKeyboardButton('более 61 дня', callback_data='0')]
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
        [InlineKeyboardButton('нет мутаций', callback_data='5')],
        [InlineKeyboardButton('не исследовались', callback_data='4')],
        [InlineKeyboardButton('EGFR редкий вариант', callback_data='3')],
        [InlineKeyboardButton('EGFR ex19', callback_data='1')],
        [InlineKeyboardButton('EGFR ex21', callback_data='2')],
        [InlineKeyboardButton('ALK позитивный', callback_data='0')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите Молекулярный статус (только для неплоскоклеточного рака):', reply_markup=reply_markup)
    
    context.user_data['molecular_status'] = query.data
    
    return MOLECULAR_STATUS

# Функция для обработки тринадцатого вопроса
async def pdl1_status(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'patient_preference', query.data)

    keyboard = [
        [InlineKeyboardButton('Не исследовался', callback_data='2'),
         InlineKeyboardButton('от 43 до 60 дней', callback_data='1'),
         InlineKeyboardButton('более 61 дня', callback_data='0')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите PD-L1 статус', reply_markup=reply_markup)
    
    context.user_data['pdl1_status'] = query.data
    
    return PDL1_STATUS

# Функция для обработки четырнадцатого вопроса
async def patient_preference(update, context):
    query = update.callback_query
    user_data = context.user_data
    update_user_data(user_data, 'patient_preference', query.data)

    keyboard = [
        [InlineKeyboardButton('Результативность лечения', callback_data='0')],
        [InlineKeyboardButton('Сохранение качества жизни', callback_data='1')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.reply_text('Выберите предпочтение пациента по ответу на терапию:', reply_markup=reply_markup)
    
    # Используйте query.data вместо user_data['patient_preference']
    context.user_data['patient_preference'] = query.data
    
    return PATIENT_PREFERENCE

def update_user_data(user_data, question, value):
    user_data[question.lower()] = value

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

    # Получите ответы от модели
    predictions = predict_model(user_responses)

    await query.message.reply_text(
        f'Вы выбрали: '
        f'{user_data["race"]}, {user_data["gender"]}, {user_data["age"]}, {user_data["smoking"]}, '
        f'{user_data["ecog"]}, {user_data["tumor_load"]}, {user_data["kras"]}, {user_data["p53"]} '
        f'{user_data["stk11"]}, {user_data["keap1"]}, {user_data["time_from_cht"]}, {user_data["molecular_status"]} '
        f'{user_data["pdl1_status"]}, {user_data["patient_preference"]}, '
        f'{predictions["experts_response"]}, {predictions["comments"]}'
    )

    # Заканчиваем конверсацию
    return ConversationHandler.END

def process_user_responses(user_responses):
    processed_data = np.array(list(map(int, user_responses.values())))
    return processed_data.reshape(1, -1)

def predict_model(user_responses):
    try:
        processed_data = process_user_responses(user_responses)
        predictions = loaded_model.predict(processed_data)

        # Преобразование предсказаний в формат ответа
        experts_response_key = np.argmax(predictions[0])
        comments_key = np.argmax(predictions[1])

        return {
            'experts_response': str(experts_response_key),
            'comments': str(comments_key)
        }
    except Exception as e:
        print(f"Ошибка при предсказании модели: {str(e)}")
        return {'experts_response': '0', 'comments': '0'}  # Замените на реальное значение по умолчанию


def main():
    application = Application.builder().token(TOKEN).build() 
    print('Бот запущен...')

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING: [CallbackQueryHandler(gender)],
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
            PATIENT_PREFERENCE: [CallbackQueryHandler(button)], 
        },
        fallbacks=[],
    )
    application.add_handler(conv_handler)

    application.run_polling()

if __name__ == '__main__':
    main()