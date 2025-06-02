Housing Price Prediction Model (Best Model Branch)
Лекции и инструкции на семинары доступны на Яндекс.Диске.

Описание проекта
Проект направлен на создание модели машинного обучения для прогнозирования цен на жилье.
В ветке best_model используется модель дерева решений (DecisionTreeRegressor) с набором признаков, включающим площадь, этажность дома, количество комнат (one-hot encoding), а также индикаторы первого и последнего этажа.

Структура проекта
text
housing_price_prediction/
├── data/
│   ├── raw/                # Исходные данные
│   ├── processed/          # Обработанные данные
├── models/                 # Обученные модели (DecisionTreeRegressor)
├── notebooks/              # Jupyter notebooks
├── service/                # Сервис предсказания цены на недвижимость
│   ├── templates/          # Шаблоны для веб-приложения
│   └── app.py              # Flask приложение
├── src/                    # Исходный код
│   ├── lifecycle.py        # Жизненный цикл модели
│   ├── models.py           # Модели машинного обучения
│   └── utils.py            # Вспомогательные функции
├── requirements.txt        # Требования к зависимостям
└── README.md
Требования
bash
pip install -r requirements.txt
Данные
Используемые данные включают следующие характеристики:

Площадь жилья

Этажность дома

Количество комнат (one-hot encoding)

Индикаторы первого и последнего этажа

Как запустить
Клонируйте репозиторий:

bash
git clone https://github.com/yourusername/housing_price_prediction.git
Установите зависимости:

bash
pip install -r requirements.txt
Запустите предобработку и обучение модели:

bash
python src/lifecycle.py --model_type t
Запустите Flask-сервис:

bash
python service/app.py --model models/linear_regression_1.pkl
Откройте браузер и перейдите по адресу http://localhost:5000 для использования веб-интерфейса.

Модель машинного обучения
Linear Regression

Метрики оценки
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Результаты
После обучения модели достигаются следующие результаты:

MSE: 81905542163831.47
RMSE: 9050168.07
R²: 0.681989
MAE: 6591915.85 рублей
Модель сохранена в файл ../models/linear_regression_1.pkl


