from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import requests

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# --- ЗАГРУЗКА МОДЕЛИ ---
try:
    # Добавляем папку models/ перед именем файла
    model = joblib.load('models/umbrella_model.pkl')
    scaler = joblib.load('models/umbrella_scaler.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    print("✅ Модели загружены из папки models/")
except FileNotFoundError:
    print("❌ ОШИБКА: Файлы не найдены в папке models/.")
    exit()

# --- ФУНКЦИИ ПОГОДЫ ---
def degrees_to_cardinal(d):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((d + 11.25)/22.5)
    return dirs[ix % 16]

def get_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=ru&format=json"
    try:
        r = requests.get(url).json()
        if 'results' in r: return r['results'][0]['latitude'], r['results'][0]['longitude'], r['results'][0]['name']
        return None, None, None
    except: return None, None, None

def get_weather_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "wind_speed_unit": "kmh"
    }
    try: return requests.get(url, params=params).json()
    except: return None

def prepare_data(weather_json):
    cur = weather_json['current']
    data = {
        'MinTemp': [cur['temperature_2m']], 'Temp9am': [cur['temperature_2m']],
        'Humidity9am': [cur['relative_humidity_2m']], 'Pressure9am': [cur['surface_pressure']],
        'WindSpeed9am': [cur['wind_speed_10m']], 'Cloud9am': [round(cur['cloud_cover']/100*8)],
        'WindDir9am': [degrees_to_cardinal(cur['wind_direction_10m'])]
    }
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    return df, data

# --- МАРШРУТЫ (ROUTES) ---

@app.route('/')
def home():
    # Теперь мы отдаем HTML файл
    return render_template('index.html')

@app.route('/predict')
def predict():
    city = request.args.get('city', 'Moscow')
    lat, lon, real_name = get_coordinates(city)
    
    if not lat: return jsonify({'error': 'Город не найден'}), 404
    
    w_data = get_weather_data(lat, lon)
    if not w_data: return jsonify({'error': 'Нет данных о погоде'}), 500
    
    try:
        X, raw = prepare_data(w_data)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]
        
        return jsonify({
            'city': real_name,
            'temp': raw['Temp9am'][0],
            'rain_predicted': bool(pred), # True или False
            'prob': round(prob * 100, 1),
            'desc': "Высокая вероятность осадков" if pred else "Осадков не ожидается"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)