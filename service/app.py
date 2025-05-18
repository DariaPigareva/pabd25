import os
import argparse
from flask import Flask, render_template, request, jsonify
from logging.config import dictConfig
import joblib

dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "service/flask.log",
            "formatter": "default",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
})

app = Flask(__name__)

# Автоматическое определение путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "linear_regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/numbers", methods=["POST"])
def process_numbers():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Пустой запрос")

        # Извлекаем параметры из формы
        area = float(data["area"])
        rooms = int(data["rooms"])
        floor = int(data["floor"])
        total_floors = int(data["total_floors"])

        # Стандартизация площади
        area_scaled = app.config["scaler"].transform([[area]])[0][0]

        # Формируем признаки в правильном порядке (как при обучении)
        features = [[
            area_scaled,            # area_scaled
            (rooms == 1),           # rooms_1
            (rooms == 2),           # rooms_2
            (rooms == 3),           # rooms_3
            (floor == 1),           # first_floor
            (floor == total_floors) # last_floor
        ]]

        # Предсказание цены
        price = app.config["model"].predict(features)[0]
        price = int(price)

        # Формируем читаемый ответ
        message = (
            f"Оценочная стоимость квартиры площадью {area} м², "
            f"{rooms}-комнатная, на {floor} этаже из {total_floors}: "
            f"{price:,} руб.".replace(',', ' ')
        )

        return jsonify({
            "status": "success",
            "input": {
                "Площадь (м²)": area,
                "Количество комнат": rooms,
                "Этаж": floor,
                "Этажей в доме": total_floors
            },
            "predicted_price": price,
            "currency": "RUB",
            "message": message
        })

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка: {str(e)}"
        }), 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", 
                      help="Path to model file",
                      default=MODEL_PATH)
    parser.add_argument("-s", "--scaler",
                      help="Path to scaler file",
                      default=SCALER_PATH)
    args = parser.parse_args()

    # Проверка существования файлов
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.scaler):
        raise FileNotFoundError(f"Scaler file not found: {args.scaler}")

    # Загрузка модели и скалера
    app.config["model"] = joblib.load(args.model)
    app.config["scaler"] = joblib.load(args.scaler)
    
    app.logger.info(f"Model loaded: {args.model}")
    app.logger.info(f"Scaler loaded: {args.scaler}")
    
    app.run(host='0.0.0.0', port=5000)



