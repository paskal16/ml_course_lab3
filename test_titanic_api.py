import requests
import json


def test_api():
    url = "http://localhost:5000"

    # Тест веб-интерфейса
    response = requests.get(url)
    print(f"Главная страница: {response.status_code}")

    # Тест API
    api_url = f"{url}/api/predict"
    test_data = {'features': [1, 1, 29.0, 0, 0, 50.0, 0]}  # Женщина, 1-й класс

    response = requests.post(api_url, json=test_data)
    print(f"API Response: {response.json()}")


if __name__ == "__main__":
    test_api()