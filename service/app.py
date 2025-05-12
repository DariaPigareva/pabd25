import argparse
from flask import Flask, render_template, request, jsonify
from logging.config import dictConfig
import joblib

# Настройка логирования
dictConfig(
    {
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
    }
)

app = Flask(__name__)

MODEL_DEFAULT_PATH = "models/linear_regression_model.pkl"

@app.route("/")
def index():
    """Отображение главной страницы с формой"""
    return render_template("index.html")

@app.route("/api/numbers", methods=["POST"])
def process_numbers():
    """API для предсказания цены по площади из JSON-запроса"""
    data = request.get_json()
    app.logger.info(f"Request data: {data}")

    try:
        area = float(data["area"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"status": "error", "data": "Некорректные входные данные"}), 400

    try:
        price_pred = app.config["model"].predict([[area]])[0]
        price_pred = int(price_pred)
    except Exception as e:
        app.logger.error(f"Ошибка предсказания: {e}")
        return jsonify({"status": "error", "data": "Ошибка при предсказании"}), 500

    return jsonify({"status": "success", "data": price_pred})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск Flask-сервиса с ML-моделью")
    parser.add_argument("-m", "--model", default=MODEL_DEFAULT_PATH, help="Путь к файлу модели")
    args = parser.parse_args()

    # Загрузка модели
    app.config["model"] = joblib.load(args.model)
    app.logger.info(f"Используется модель: {args.model}")

    # Запуск сервера
    app.run(debug=True)
