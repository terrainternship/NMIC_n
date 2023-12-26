# v1.0
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

 # Load the pre-trained Keras model
from tensorflow.keras.models import load_model
model = load_model('mnic_model_keras.h5') 

# Loading pre-trained OneHotEncoder model
with open('encoder_ohe.pkl', 'rb') as file:
    encoder_ohe = pickle.load(file)

# Loading pre-trained LabelEncoder model
with open('encoder_lb.pkl', 'rb') as file:
    encoder_lb = pickle.load(file)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# translation dict mapping
translations = {
    "Азиатская": "Азиатская",
    "Европейская": "Европейская",
    "Другая": "Другая",
    "male": "Мужской",
    "female": "Женский",
    "up_to_70": "До 70",
    "more_than_70": ">70",
    "smoking": "В настоящее время",
    "quit_smoking_before_year": "Курение в прошлом (бросил более 1 месяца до 1 года)",
    "quit_smoking_after_year": "Курение в прошлом (бросил более 1 года)",
    "0_1": "0-1",
    "2": "2",
    "yes": "да",
    "no": "нет",
    "up_to_42": "до 42 дней", 
    "in_between_43_and_60": "от 43 до 60 дней", 
    "more_than_61": "более 61 дня",
    "no_mutation": "нет мутаций", 
    "unexplored": "не исследовались",
    "EGFR_rare": "EGFR редкий вариант",
    "EGFR_ex19": "EGFR ex19",
    "EGFR_ex21": "EGFR ex21",
    "ALK_positive": "ALK позитивный",
    "pdl1_unexplored": "Не исследовался",
    "less_1_percent": "Менее 1%",
    "more_1_percent": "Более 1%",
    "therapy_result": "Результативность лечения", 
    "therapy_save": "сохранение качества жизни"
}

# Global variable to store user responses
user_responses = {}

load_dotenv()
TOKEN = os.environ.get("TOKEN")

# Dialogue steps
QUESTIONS = range(14)

# Questions
questions = [
    "Ваша раса:",
    "Ваш пол:",
    "Ваш возраст:",
    "Статус курения:",
    "Статус ECOG:",
    "Есть опухолевая нагрузка? (симптомная опухоль):",
    "Имеются ли Ко-мутации KRAS:",
    "Имеются ли Ко-мутации p53:",
    "Имеются ли Ко-мутации stk11:",
    "Имеются ли Ко-мутации keap1:",
    "Cрок после окончания ХЛТ:",
    "Молекулярный статус (для неплоскоклеточного рака):",
    "PD-L1 статус:",
    "Предпочтение пациента по ответу на терапию:"
]

# Answer options
answers = [
    ["Азиатская", "Европейская", "Другая"],
    ["male", "female"],
    ["up_to_70", "more_than_70"],
    ["smoking", "quit_smoking_before_year", "quit_smoking_after_year"],
    ["0_1", "2"],
    ["yes", "no"],
    ["yes", "no"],
    ["yes", "no"],
    ["yes", "no"],
    ["yes", "no"],
    ["up_to_42", "in_between_43_and_60", "more_than_61"],
    ["no_mutation", "unexplored", "EGFR_rare", "EGFR_ex19", "EGFR_ex21", "ALK_positive"],
    ["pdl1_unexplored", "less_1_percent", "more_1_percent"],
    ["therapy_result", "therapy_save"]
]


def update_user_data(user_data, question, value):
    user_data[question.lower()] = value

# Command /start initiates the dialogue
async def start(update, context):
    keyboard = [
        [InlineKeyboardButton(answer, callback_data=answer) for answer in answers[0]]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(questions[0], reply_markup=reply_markup)

    # Set the first question
    context.user_data["question_number"] = 0
    return QUESTIONS[0]
    

# Function to display questions and answer options
async def ask_question(update, context):
    query = update.callback_query
    question_number = context.user_data.get("question_number")


    if question_number is not None and 0 <= question_number < len(QUESTIONS):
        
        # Display keyboard with answer options and translate
        keyboard = [
            [InlineKeyboardButton(translations.get(answer, answer), callback_data=answer) for answer in answers[question_number]]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send question and keyboard to the user text=questions[question_number]
        # await update.message.reply_text(questions[question_number], reply_markup=reply_markup)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=questions[question_number], reply_markup=reply_markup)

        # If this is the last question, hide the keyboard
        # if question_number == len(QUESTIONS) - 1:
        # await query.bot.edit_message_reply_markup(chat_id=update.effective_chat.id, message_id=update.callback_query.message.message_id, reply_markup=InlineKeyboardMarkup([]))

    return QUESTIONS[question_number] if question_number is not None else QUESTIONS[0]




# Function to handle answers
async def handle_answer(update, context):
    query = update.callback_query
    question_number = context.user_data["question_number"]

    # Get the selected answer and store it in a variable
    selected_answer = query.data
    user_responses[questions[question_number]] = selected_answer

    # Set the next question
    context.user_data["question_number"] += 1

    # If all questions have been asked, end the survey and show the results
    if question_number == len(QUESTIONS) - 1:
        await show_results(update, context)
        return QUESTIONS[-1]
        # return ConversationHandler.END

    # Hide the keyboard after handling the answer
    # await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))

    # Display the next question
    return await ask_question(update, context)


# Function to display the results and write to a CSV file
async def show_results(update, context):
    chat_id = update.effective_chat.id
    
    # Display the user's responses and translate
    for question, answer in user_responses.items():
        translated_answer = translations.get(answer, answer)
        await context.bot.send_message(chat_id, f"{question} {translated_answer}")
        # await context.bot.send_message(chat_id, f"{question}: {answer}")
    
        
    # Preparing dataset to ML model input form
    # Create a Pandas DataFrame from user_responses
    df = pd.DataFrame(list(user_responses.items()), columns=['Question', 'Answer'])
    
    # Remove colons from column headers
    new_column_names = [
        'Раса', 'Пол', 'Возраст', 'Статус курения', 'ECOG',
        'Есть опухолевая нагрузка? (симптомная опухоль)', 'Ко-мутации KRAS',
        'Ко-мутации p53.', 'Ко-мутации STK11', 'Ко-мутации KEAP1',
        'Срок от окончания ХЛТ',
        'Молекулярный статус (только для неплоскоклеточного рака)',
        'PD-L1 статус', 'Предпочтение пациента по ответу на терапию'
    ]
    

    # Translation answer
    df['Answer'] = df['Answer'].map(translations)

    # Transpose the DataFrame
    df = df.transpose()

    # # Set the first row as column headers
    df.columns = df.iloc[0]

    # # Drop the first row
    df = df[1:]

    # # Reset the row indices and rename the columns
    df = df.reset_index(drop=True)

    df.columns = new_column_names

    df.to_csv('user_choice.csv', index=False, encoding="utf-8")

    # Transform all numeric fields to ECOG strings accept 2 or 0-1 values
    df = df.applymap(lambda x: str(x) if isinstance(x, (int, float)) else x)

    # Make predictions using the Keras model
    X_unseen_encoded = encoder_ohe.transform(df)
    predictions = model.predict(X_unseen_encoded)

    y_classes = np.argmax(predictions, axis=1)

    result = encoder_lb.inverse_transform(y_classes)[0]
    
    # Display the predictions

    if result:
        await context.bot.send_message(chat_id, f"Рекомендация: {result}")
    else:
        await context.bot.send_message(chat_id, "Что-то пошло не так, ведутся технические работы")

    # await context.bot.send_message(chat_id, f"Рекомендация: {result}")


    # Reset variables
    reset_variables()


# Function to reset variables
def reset_variables():
    global user_responses
    user_responses = {}

def main():
    application = Application.builder().token(TOKEN).build()
    print("Commencing countdown, engines on...")

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(handle_answer))

    application.run_polling()

if __name__ == '__main__':
    main()