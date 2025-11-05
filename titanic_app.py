from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os
import sys

# AI-ассистент: использован DeepSeek для создания веб-сервиса

app = Flask(__name__)


# Проверка виртуального окружения
def check_environment():
    in_venv = sys.prefix != sys.base_prefix
    print(f"🎯 Виртуальное окружение: {'✅ АКТИВНО' if in_venv else '❌ НЕ АКТИВНО'}")
    return in_venv


# Загрузка модели
def load_model():
    try:
        with open('titanic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ ML-модель загружена успешно!")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None


model = load_model()

# HTML шаблон
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚢 Titanic Survival Predictor</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .survived { background: #d4edda; color: #155724; }
        .not-survived { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>🚢 Titanic Survival Predictor</h1>
    <form method="POST" action="/predict">
        <!-- Поля формы -->
        <div class="form-group">
            <label>Класс билета:</label>
            <select name="pclass" required>
                <option value="1">1-й класс</option>
                <option value="2">2-й класс</option>
                <option value="3">3-й класс</option>
            </select>
        </div>

        <div class="form-group">
            <label>Пол:</label>
            <select name="sex" required>
                <option value="male">Мужской</option>
                <option value="female">Женский</option>
            </select>
        </div>

        <div class="form-group">
            <label>Возраст:</label>
            <input type="number" name="age" min="0" max="100" required>
        </div>

        <button type="submit">🔮 Предсказать выживание</button>
    </form>

    {% if result %}
    <div class="result {% if result.prediction == 1 %}survived{% else %}not-survived{% endif %}">
        <h3>Результат:</h3>
        <p>Вероятность выживания: {{ result.probability }}%</p>
        <p>Прогноз: {{ "✅ ВЫЖИВ" if result.prediction == 1 else "❌ НЕ ВЫЖИВ" }}</p>
    </div>
    {% endif %}
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return "Модель не загружена", 500

        # Получение данных из формы
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])

        # Кодирование и предсказание
        sex_encoded = 1 if sex == 'female' else 0
        features = np.array([[pclass, sex_encoded, age, 0, 0, 50.0, 0]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        result = {
            'prediction': int(prediction),
            'probability': f"{probability:.1f}"
        }

        return render_template_string(HTML_TEMPLATE, result=result)

    except Exception as e:
        return f"Ошибка: {str(e)}", 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API для програмmatic доступа"""
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()

        return jsonify({
            'prediction': int(prediction),
            'probability': probability,
            'survival_chance': f"{probability[1] * 100:.1f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Запуск Titanic ML Service")
    print("=" * 50)

    check_environment()

    if not os.path.exists('titanic_model.pkl'):
        print("❌ Сначала обучите модель: python train_titanic_model.py")
        exit(1)

    print("✅ Сервис запускается на http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)