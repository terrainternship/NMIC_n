from preprocess import preprocess
from model import training

dataset_path = 'dataset.xlsx'
x_train, x_test, y_train, y_test = preprocess(dataset_path)

model_weights_path = training(x_train, x_test, y_train, y_test)
print("Модель обучена и веса сохранены в файле", model_weights_path)