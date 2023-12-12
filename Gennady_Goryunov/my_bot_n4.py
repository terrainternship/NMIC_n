from telegram.ext import Updater, CommandHandler, ConversationHandler, MessageHandler, filters, Application, CallbackQueryHandler, BaseHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
import os
from keras.models import load_model
import numpy as np    # Нампай для массивов

# Загрузка модели
model = load_model('my_model_cancer_pred.h5')
unswer = ['Ответ эксперта (Лактионов)_Алектиниб, альтернатива: Дурвалумаб, комментарий: Возможна низкая эффективность Дурвалумаба, Алектиниб без доказательной базы',
       'Ответ эксперта (Лактионов)_Алектиниб, альтернатива: Наблюдение, комментарий: Возможна низкая эффективность Дурвалумаба, Алектиниб без доказательной базы',
       'Ответ эксперта (Лактионов)_Дурвалумаб',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Алектиниб, комментарий: Возможна низкая эффективность Дурвалумаба, Алектиниб без доказательной базы',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Наблюдение',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Наблюдение, комментарий: Вероятность 45% PD-L1 < 1% с потенциально низкой эффективностью Дурвалумаба',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Наблюдение, комментарий: Возможна низкая эффективность Дурвалумаба при PD-1 < 1%',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Наблюдение, комментарий: Возможна низкая эффективность Дурвалумаба при PD-1 < 1%, возможно есть активирующие мутации',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Наблюдение, комментарий: Возможно есть мутации и эффективность Дурвалумаба будет низкой',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Наблюдение, комментарий: Возможно есть мутации и эффективность Дурвалумаба будет низкой\n, Вероятность 45% PD-L1 < 1% с потенциально низкой эффективностью Дурвалумаба',
       'Ответ эксперта (Лактионов)_Дурвалумаб, альтернатива: Наблюдение, комментарий: Возможно есть мутации и эффективность Дурвалумаба будет низкой\n, Возможна низкая эффективность Дурвалумаба при PD-1 < 1%',
       'Ответ эксперта (Лактионов)_Дурвалумаб, комментарий: Возможно есть мутации и эффективность Дурвалумаба будет низкой',
       'Ответ эксперта (Лактионов)_Наблюдение',
       'Ответ эксперта (Лактионов)_Наблюдение, альтернатива: Алектиниб, комментарий: Возможна низкая эффективность Дурвалумаба, Алектиниб без доказательной базы',
       'Ответ эксперта (Лактионов)_Наблюдение, альтернатива: Дурвалумаб, комментарий: Вероятность 45% PD-L1 < 1% с потенциально низкой эффективностью Дурвалумаба',
       'Ответ эксперта (Лактионов)_Наблюдение, альтернатива: Дурвалумаб, комментарий: Возможна низкая эффективность Дурвалумаба при PD-1 < 1%',
       'Ответ эксперта (Лактионов)_Наблюдение, альтернатива: Дурвалумаб, комментарий: Возможна низкая эффективность Дурвалумаба при PD-1 < 1%, возможно есть активирующие мутации',
       'Ответ эксперта (Лактионов)_Наблюдение, альтернатива: Дурвалумаб, комментарий: Нет доказательной базы для назначения после перерыва >61 дня после ХЛТ',
       'Ответ эксперта (Лактионов)_Наблюдение, альтернатива: Осимертиниб, комментарий: Возможна низкая эффективность Дурвалумаба, Осимертиниб без доказательной базы',
       'Ответ эксперта (Лактионов)_Наблюдение, альтернатива: Осимертиниб, комментарий: Возможна низкая эффективность Дурвалумаба, Осимертиниб без доказательной базы, эффективность Осимертиниба ниже при 21 экзоне',
       'Ответ эксперта (Лактионов)_Наблюдение, комментарий: Вероятность 45% PD-L1 < 1% с потенциально низкой эффективностью Дурвалумаба',
       'Ответ эксперта (Лактионов)_Наблюдение, комментарий: Возможна низкая эффективность Дурвалумаба, Осимертиниб без доказательной базы',
       'Ответ эксперта (Лактионов)_Наблюдение, комментарий: Возможно есть мутации и эффективность Дурвалумаба будет низкой\n, Вероятность 45% PD-L1 < 1% с потенциально низкой эффективностью Дурвалумаба',
       'Ответ эксперта (Лактионов)_Наблюдение, комментарий: Возможно есть мутации и эффективность Дурвалумаба будет низкой\n, Возможна низкая эффективность Дурвалумаба при PD-1 < 1%',
       'Ответ эксперта (Лактионов)_Наблюдение, комментарий: Нет доказательной базы для назначения после перерыва >61 дня после ХЛТ',
       'Ответ эксперта (Лактионов)_Осимертиниб, альтернатива: Наблюдение, комментарий: Возможна низкая эффективность Дурвалумаба, Осимертиниб без доказательной базы',
       'Ответ эксперта (Лактионов)_Осимертиниб, альтернатива: Наблюдение, комментарий: Возможна низкая эффективность Дурвалумаба, Осимертиниб без доказательной базы, эффективность Осимертиниба ниже при 21 экзоне']

model.summary()
check_predict=np.argmax(model.predict([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]]))
print(check_predict)
print(unswer[check_predict])
# Возьмем временные окружения из .env
load_dotenv()

# загружаем токен бота
TOKEN = os.environ.get("TOKEN")

# Определение состояний
FIRST, SECOND, THIRD, FOURTH, FIFTH, SIXTH, SEVENTH, EIGHTH, NINTH, TENTH   = range(10)

# Функция для обработки команды /start
async def start(update, context):
    
    # создаем спиок Inline кнопок первого вопроса
    keyboard = [[InlineKeyboardButton("более 61 дня", callback_data=0), # все кнопки на одном уровне
                 InlineKeyboardButton("до 42 дней", callback_data=1),
                 InlineKeyboardButton("от 43 до 60 дней", callback_data=2)]]
    
    # создаем Inline клавиатуру
    reply_markup = InlineKeyboardMarkup(keyboard)
       # прикрепляем клавиатуру к сообщению
    await update.message.reply_text("Укажите срок окончания ХЛТ", reply_markup=reply_markup)
 #   await update.callback_query.edit_message_text("Укажите срок окончания ХЛТ", reply_markup=reply_markup)
#    await context.bot.send_message(chat_id=update.effective_chat.id, text="Укажите срок окончания ХЛТ", reply_markup=reply_markup)

    return SECOND

# Функция для обработки команды /start
async def first_state(update, context):
    
    # создаем спиок Inline кнопок первого вопроса
    keyboard = [[InlineKeyboardButton("более 61 дня", callback_data=0), # все кнопки на одном уровне
                 InlineKeyboardButton("до 42 дней", callback_data=1),
                 InlineKeyboardButton("от 43 до 60 дней", callback_data=2)]]
    
    # создаем Inline клавиатуру
    reply_markup = InlineKeyboardMarkup(keyboard)
       # прикрепляем клавиатуру к сообщению
  #  await update.message.reply_text("Укажите срок окончания ХЛТ", reply_markup=reply_markup)
  #  await update.callback_query.edit_message_text("Укажите срок окончания ХЛТ", reply_markup=reply_markup)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Укажите срок окончания ХЛТ", reply_markup=reply_markup)
#    await context.bot.send_message(chat_id=update.effective_chat.id, text="Укажите срок окончания ХЛТ", reply_markup=reply_markup)

    return SECOND

# Функция для обработки первого состояния
async def second_state(update, context):
    keyboard = [[InlineKeyboardButton("ALK позитивный", callback_data=3),
                 InlineKeyboardButton("EGFR ex19", callback_data=4),                
                 InlineKeyboardButton("EGFR ex21", callback_data=5),                
                 InlineKeyboardButton("EGFR редкий вариант", callback_data=6),                
                 InlineKeyboardButton("Нет мутаций", callback_data=7),                
                 InlineKeyboardButton("Не исследовался", callback_data=8)]]
    # Создаем разметку для кнопок
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Отправляем вопрос с кнопками пользователю
 #   await update.message.reply_text('Второй вопрос:', reply_markup=reply_markup)

    if update.callback_query is not None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Молекулярный статус (только для неплоскоклеточного рака)", reply_markup=reply_markup)
      #  await update.callback_query.edit_message_text("Молекулярный статус (только для неплоскоклеточного рака)", reply_markup=reply_markup)
    else:
        # Обработка случая, когда не было нажато ни одной кнопки
        await update.message.reply_text("Ошибка: Не было нажато ни одной кнопки.", reply_markup=reply_markup)
    # прикрепляем клавиатуру к сообщению
   # await update.callback_query.edit_message_text("Молекулярный статус (только для неплоскоклеточного рака)", reply_markup=reply_markup)

    return SIXTH

# Функция для обработки шестого состояния
async def sixth_state(update, context):
    keyboard = [[InlineKeyboardButton("Более 1%", callback_data=9),
                 InlineKeyboardButton("Менее 1%", callback_data=10),
                 InlineKeyboardButton("Не исследовался", callback_data=11)]]
    # Создаем разметку для кнопок
    reply_markup = InlineKeyboardMarkup(keyboard) #, one_time_keyboard=True) # параметр скывает кнопку после нажатия)
    # Отправляем вопрос с кнопками пользователю
   # update.message.reply_text('Второй вопрос:', reply_markup=reply_markup)

    # прикрепляем клавиатуру к сообщению
    await context.bot.send_message(chat_id=update.effective_chat.id, text="PD-L1 статус", reply_markup=reply_markup)
  #  await update.callback_query.edit_message_text("PD-L1 статус", reply_markup=reply_markup)

    return SEVENTH

# Функция для обработки седьмого состояния
async def seventh_state(update, context):
    keyboard = [[InlineKeyboardButton("Результативность лечения", callback_data=12),
                 InlineKeyboardButton("Cохранение качества жизни", callback_data=13)]]
    # Создаем разметку для кнопок
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Отправляем вопрос с кнопками пользователю
   # update.message.reply_text('Второй вопрос:', reply_markup=reply_markup)

    # прикрепляем клавиатуру к сообщению
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Предпочтение пациента по ответу на терапию", reply_markup=reply_markup)
   # await update.callback_query.edit_message_text("Предпочтение пациента по ответу на терапию", reply_markup=reply_markup)

    return EIGHTH

# Функция для обработки заключительного состояния
async def eighth_state(update, context):
   # text = update.message.text
    # Обработка текста из второго состояния
#    await update.message.reply_text(f"Ты написал: {text}. Работа с ботом завершена.")
#    await update.message.reply_text(f"Работа с ботом завершена.")
    bot_unswer=context.user_data[0]
    bot_unswer += context.user_data[1]
    bot_unswer += context.user_data[2]
    bot_unswer += context.user_data[3]
 #   await update.callback_query.edit_message_text(f"Работа с ботом завершена.") # {bot_unswer} ")

  #  print("-----начинается предсказание 1-------")
  #  check_predict=np.argmax(model.predict([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]]))
    print("-----начинается предсказание -------")
    print(bot_unswer)
    check_predict=np.argmax(model.predict([list(bot_unswer)]))
    print("-----предсказание  закончено-------")
    print(check_predict)
    print(unswer[check_predict])
    await context.bot.send_message(chat_id=update.callback_query.message.chat_id, text=unswer[check_predict])
    await context.bot.send_message(chat_id=update.callback_query.message.chat_id, text="Новый запрос")
  #  await update.callback_query.edit_message_text(unswer[check_predict]) # {bot_unswer} ")
    return await first_state(update, context)
  #  return ConversationHandler.END

async def cancel(update, context):
    await update.message.reply_text("Отмена.")
    return ConversationHandler.END

# функция обработки нажатия на кнопки Inline клавиатуры
async def button(update, context):
    # параметры входящего запроса при нажатии на кнопку
    query = update.callback_query
    match query.data:
        case "0":
#            await query.edit_message_text(text=f"Срок окончания ХЛТ более 61 дня")
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Срок окончания ХЛТ более 61 дня")
            context.user_data[0]=[1,0,0]
            return await second_state(update, context)
        case "1":
 #           await query.edit_message_text(text=f"Срок окончания ХЛТ до 42 дней")
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Срок окончания ХЛТ до 42 дней")
            context.user_data[0]=[0,1,0]
            return await second_state(update, context)
        case "2":
 #           await query.edit_message_text(text=f"Срок окончания ХЛТ от 43 до 60 дней"
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Срок окончания ХЛТ от 43 до 60 дней")
            context.user_data[0]=[0,0,1]
            return await second_state(update, context)
        case "3":
 #           await query.edit_message_text(text=f"Молекулярный статус (только для неплоскоклеточного рака)_ ALK позитивный - да")
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Молекулярный статус (только для неплоскоклеточного рака)_ ALK позитивный")
            context.user_data[1]= [1,0,0,0,0,0]
            return await sixth_state(update, context)
        case "4":
 #           await query.edit_message_text(text=f"Молекулярный статус (только для неплоскоклеточного рака)_ ALK позитивный - нет")
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Молекулярный статус (только для неплоскоклеточного рака)_ EGFR ex19")
            context.user_data[1]= [0,1,0,0,0,0]
            return await sixth_state(update, context)
        case "5":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Молекулярный статус (только для неплоскоклеточного рака)_ EGFR ex21")
            context.user_data[1]= [0,0,1,0,0,0]
            return await sixth_state(update, context)
        case "6":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Молекулярный статус (только для неплоскоклеточного рака)_ EGFR редкий вариант")
            context.user_data[1]= [0,0,0,1,0,0]
            return await sixth_state(update, context)
        case "7":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Молекулярный статус (только для неплоскоклеточного рака) - Не исследовался")
            context.user_data[1]= [0,0,0,0,1,0]
            return await sixth_state(update, context)
        case "8":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Молекулярный статус (только для неплоскоклеточного рака) - Нет мутации")
            context.user_data[1]= [0,0,0,0,0,1]
            return await sixth_state(update, context)
        case "9":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"PD-L1 статус - Более 1%")
            context.user_data[2]= [1,0,0]
            return await seventh_state(update, context)
        case "10":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"PD-L1 статус - Менее 1%")
            context.user_data[2]= [0,1,0]
            return await seventh_state(update, context)
        case "11":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"PD-L1 статус - Не исследовался")
            context.user_data[2]= [0,0,1]
            return await seventh_state(update, context)
        case "12":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Предпочтение пациента по ответу на терапию - Результативность лечения")
            context.user_data[3]= [1]
            return await eighth_state(update, context)
        case "13":
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([]))
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"Предпочтение пациента по ответу на терапию - Cохранение качества жизни")
            context.user_data[3]= [0]
            return await eighth_state(update, context)
    return ConversationHandler.END

def main():
    # Создание объекта Updater и передача токена вашего бота
    
    updater = Application.builder().token(TOKEN).build()
    print("Бот запущен...")

    # Создание объекта ConversationHandler
    conv_handler = ConversationHandler(   #  per_message=True,
        entry_points=[CommandHandler('start', start)],
        states={
            FIRST: [MessageHandler(filters.TEXT, first_state)],
            SECOND: [MessageHandler(filters.TEXT, second_state)],
          #  FIRST: [CallbackQueryHandler(first_state)],   
            SIXTH: [MessageHandler(filters.TEXT, sixth_state)],
            SEVENTH: [MessageHandler(filters.TEXT, seventh_state)],
            EIGHTH: [MessageHandler(filters.TEXT, eighth_state)],
          #  SECOND: [CallbackQueryHandler(second_state)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # добавляем обработчик нажатия  Inline кнопок
    updater.add_handler(CallbackQueryHandler(button))

    # Добавление ConversationHandler в Updater
    updater.add_handler(conv_handler)

    # Запуск бота
    updater.run_polling()

    # Остановка бота при нажатии Ctrl-C
    # updater.wait_until_idle() 

if __name__ == '__main__':
    main()