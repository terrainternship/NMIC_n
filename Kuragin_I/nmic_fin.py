import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Установка размера графиков
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

# Загрузка данных
data_path = 'C:/Users/user/Desktop/My DOCS/1 Projects/aineuro_int/dataset_pht3_send.csv'
d_s = pd.read_csv(data_path, index_col=0)

d_s.head(10)

d_s = d_s.drop(columns=['Поставьте галочку, если уверены на 100%',	'Если 75%',	'Если 50%',	'Альтернатива, если 50% - обязательно',	'Комментарий'], axis=0)

d_s.head(10)

for name, values in d_s.items():
  print(name, ":", d_s[name].unique())

data_colnames = d_s.columns.tolist()
lst_count = len(data_colnames)
for i in range(lst_count):
  i_elem = data_colnames[i]
  print(f"{i}->{lst_count}: '{i_elem}'")

def getDictionary(data):
  dict={}

  for name in data_colnames:
   index = data_colnames.index(name) 
   uniq = data[name].unique()
   if uniq[0]!='0':
     a = np.insert(uniq, 0, '0') 
     
     dict[index] =a 
   else: 
    dict [index] = data[name].unique() 
  return dict 

dictionaryDefault = getDictionary(d_s) 
print(dictionaryDefault)

def getArgmaxData(values):
  all_data = [] 

  for val in values: 
    ohe = [] 

    for i in range(len(val)): 
      currentList =  dictionaryDefault[i].tolist() 
      currentIndex = currentList.index(val[i]) 

      a = np.argmax(list(to_categorical(currentIndex, len(currentList)).astype('int')))
      ohe.append(a) 
    all_data.append(ohe) 

  return all_data


all_data = getArgmaxData(d_s.values) 
categorical = pd.DataFrame(all_data,columns=data_colnames)
categorical.head(10)

x_data = categorical.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
x_data.head(5)

y_data = categorical.iloc[:, [14]]
y_data.head(5)

CLASS_COUNT = 10


x_train, x_test, y_train, y_test = train_test_split(x_data, 
                                                    y_data, 
                                                    test_size = 0.1, 
                                                    shuffle=True, 
                                                    random_state=42) 

print('Обучающая выборка данных', len(x_train))
print('Обучающая выборка меток', len(y_train))
print()
print('Тестовая выборка данных', len(x_test))
print('Тестовая выборка меток', len(y_test))

print(x_train.shape)
print(y_train.shape)

y_train = to_categorical(y_train, CLASS_COUNT)
y_test = to_categorical(y_test, CLASS_COUNT)

print(y_train.shape)
model = Sequential()

model.add(Dense(140, input_dim=14, activation='relu', name='Class_2'))  
model.add(Dropout(0.3))
model.add(Dense(CLASS_COUNT, activation='softmax', name='Class_4'))  
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

store_learning = model.fit(x_train,
                  y_train,
                  batch_size=100,
                  epochs=20,
                  validation_split=0.2,  
                  shuffle=True,
                  verbose=1)

print(store_learning.history)

plt.figure(1, figsize=(18, 5))
plt.subplot(1, 2, 1)
plt.plot(store_learning.history['loss'],
         label='Значение ошибки на обучающем наборе')
plt.plot(store_learning.history['val_loss'],
         label='Значение ошибки на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Значение ошибки')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(store_learning.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(store_learning.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
     


scores = model.evaluate(x_test,
                        y_test,
                        verbose=1
                        )

print(type(scores))
print(scores)

print('Процент верных ответов на тестовых данных:', round(scores[1],2) * 100, '%')

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
print("Первые несколько предсказаний:")
for i in range(5):
    print(f"Предсказанная метка: {predicted_classes[i]}, Реальная метка: {np.argmax(y_test[i])}")


# Предсказываем результаты на тестовых данных
predictions = model.predict(x_test)

# Преобразуем предсказания из one-hot encoded в числовые метки
predicted_classes = np.argmax(predictions, axis=1)

# Преобразуем реальные метки из one-hot encoded в числовые метки
true_classes = np.argmax(y_test, axis=1)

# Выводим первые несколько предсказаний
print("Первые несколько предсказаний:")
for i in range(5):
    # Используем словарь для получения описательной метки
    predicted_label = dictionaryDefault[14][predicted_classes[i]]  # Предполагается, что dictionaryDefault[14] содержит метки классов
    true_label = dictionaryDefault[14][true_classes[i]]
    print(f"Предсказанная метка: {predicted_label}, Реальная метка: {true_label}")
def predict_from_dataset(row_index):
    # Убедитесь, что индекс находится в пределах диапазона датафрейма
    if row_index >= len(d_s) or row_index < 0:
        raise ValueError("Индекс вне диапазона датафрейма.")

    # Получаем строку датафрейма по индексу, исключая целевую переменную
    row_data = d_s.iloc[row_index, :-1]  # Исключаем последний столбец (целевая переменная)

    # Преобразуем строку в формат, подходящий для модели
    processed_row = getArgmaxData([row_data.values])  # Используем вашу функцию getArgmaxData

    # Преобразуем в массив NumPy
    processed_row_array = np.array(processed_row)

    # Выполняем предсказание
    prediction = model.predict(processed_row_array)

    # Получаем индекс предсказанного класса
    predicted_class_index = np.argmax(prediction, axis=1)

    # Получаем описательную метку предсказания
    predicted_label = dictionaryDefault[14][predicted_class_index[0]]

    return predicted_label

# Тестирование функции с использованием строки из датафрейма
row_index = 200000  # Пример индекса
predicted_label = predict_from_dataset(row_index)
print(f"Предсказанная метка для строки {row_index}: {predicted_label}")

import os
model.save('nmic_model.h5')
model.save('my_saved_nmic_model')
import json

# Сохраняем словарь в JSON файл
with open('dictionaryDefault.json', 'w' , encoding='utf-8') as file:
    json.dump(dictionaryDefault, file)
