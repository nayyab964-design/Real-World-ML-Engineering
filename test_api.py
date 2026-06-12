import json
import requests

API_URL = "http://localhost:8000"

print("Testing /health endpoint...")
response = requests.get(f"{API_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

test_cases = [
    {
        "name": "Normal Operation",
        "data": {"temperature": 70, "vibration": 0.4, "pressure": 95, "rpm": 1500, "age_days": 100}
    },
    {
        "name": "High Risk",
        "data": {"temperature": 95, "vibration": 0.9, "pressure": 135, "rpm": 1500, "age_days": 320}
    }
]

for test in test_cases:
    print(f'Testing: {test["name"]}')
    response = requests.post(f"{API_URL}/predict", json=test["data"])
    result = response.json()
    print(f'  Will Fail:   {result["will_fail"]}')
    print(f'  Probability: {result["probability"]:.3f}')
    print(f'  Latency:     {result["latency_ms"]} ms\n')