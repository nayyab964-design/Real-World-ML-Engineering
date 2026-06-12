import time
from datetime import datetime
import requests

API_URL = "http://localhost:8000"

def display_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=3)
        response.raise_for_status()
        metrics = response.json()

        print("\n" + "=" * 50)
        print(f'METRICS DASHBOARD - {datetime.now().strftime("%H:%M:%S")}')
        print("=" * 50)
        print(f'Total Requests:      {metrics["total_requests"]}')
        print(f'Failures Predicted:  {metrics["failures_predicted"]}')
        print(f'Failure Rate:        {metrics["failure_rate"]:.1%}')
        print(f'Avg Latency:         {metrics["avg_latency_ms"]:.2f} ms')
        print(f'Errors:              {metrics["errors"]}')
        print("=" * 50)
    except Exception as e:
        print(f"\nCould not fetch metrics: {e}")

if __name__ == "__main__":
    while True:
        try:
            display_metrics()
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break