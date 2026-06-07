import pandas as pd


def direct_test_prediction(temperature, vibration, pressure, rpm, age_days):
    # MLflow se load karne ke bajaye, jo model variables aapke environment mein hain unhe direct use karte hain
    # Humne columns ka exact wahi sequence rakha hai jo training mein tha
    input_data = pd.DataFrame([{
        'temperature': float(temperature),
        'vibration': float(vibration),
        'pressure': float(pressure),
        'rpm': float(rpm),
        'age_days': int(age_days)
    }]).values

    # Model variable (jo aapne abhi bina scaling ke train kiya hai)
    prediction = model.predict(input_data)[0]

    return {
        'will_fail': bool(prediction),
        'recommendation': 'Schedule maintenance' if int(prediction) == 1 else 'Normal operation'
    }


# --- Testing 3 Scenarios ---
scenarios = [
    {'name': 'Normal', 'temp': 70, 'vib': 0.4, 'press': 95, 'rpm': 1500, 'age': 100},
    {'name': 'High Risk', 'temp': 95, 'vib': 0.9, 'press': 135, 'rpm': 1500, 'age': 320},
    {'name': 'Medium', 'temp': 85, 'vib': 0.6, 'press': 110, 'rpm': 1500, 'age': 200}
]

print('Direct Model Predictions (Bypassing MLflow Registry):\n')
for s in scenarios:
    result = direct_test_prediction(
        s['temp'], s['vib'], s['press'], s['rpm'], s['age']
    )
    print(f"{s['name']:12} → {result['recommendation']}")