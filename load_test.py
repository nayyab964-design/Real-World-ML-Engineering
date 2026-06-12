import random
import time
import requests

API_URL = "http://localhost:8000"
NUM_REQUESTS = 100

def generate_random_request():
    return {
        "temperature": random.uniform(60, 100),
        "vibration": random.uniform(0.3, 1.0),
        "pressure": random.uniform(80, 140),
        "rpm": random.uniform(1400, 1600),
        "age_days": random.randint(0, 365)
    }

print(f"Starting load test: {NUM_REQUESTS} requests...")
start_time = time.time()
successes, failures = 0, 0
latencies = []

for i in range(NUM_REQUESTS):
    try:
        req_start = time.time()
        response = requests.post(f"{API_URL}/predict", json=generate_random_request())
        latencies.append((time.time() - req_start) * 1000)
        if response.status_code == 200: successes += 1
        else: failures += 1
        if (i + 1) % 20 == 0: print(f"Progress: {i+1}/{NUM_REQUESTS}")
    except Exception as e:
        failures += 1
        print(f"Error: {e}")

total_time = time.time() - start_time
print("\n" + "=" * 50 + "\nLOAD TEST RESULTS\n" + "=" * 50)
print(f"Total Requests: {NUM_REQUESTS}\nSuccessful:     {successes}\nFailed:         {failures}")
print(f"Success Rate:   {(successes/NUM_REQUESTS):.1%}\nTotal Time:     {total_time:.2f}s")
if latencies:
    print(f"Avg Latency:    {(sum(latencies)/len(latencies)):.2f} ms")
print("=" * 50)