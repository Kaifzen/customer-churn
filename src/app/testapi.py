import argparse
import json
import sys
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


SAMPLE_CUSTOMER = {
	"gender": "Female",
	"Partner": "No",
	"Dependents": "No",
	"PhoneService": "Yes",
	"MultipleLines": "No",
	"InternetService": "Fiber optic",
	"OnlineSecurity": "No",
	"OnlineBackup": "No",
	"DeviceProtection": "No",
	"TechSupport": "No",
	"StreamingTV": "Yes",
	"StreamingMovies": "Yes",
	"Contract": "Month-to-month",
	"PaperlessBilling": "Yes",
	"PaymentMethod": "Electronic check",
	"tenure": 1,
	"MonthlyCharges": 85.0,
	"TotalCharges": 85.0,
}


def _get_json(url: str, timeout: float = 10.0):
	req = request.Request(url, method="GET")
	with request.urlopen(req, timeout=timeout) as resp:
		body = resp.read().decode("utf-8")
		return resp.status, json.loads(body)


def _post_json(url: str, payload: dict, timeout: float = 10.0):
	data = json.dumps(payload).encode("utf-8")
	req = request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with request.urlopen(req, timeout=timeout) as resp:
		body = resp.read().decode("utf-8")
		return resp.status, json.loads(body)


def main():
	parser = argparse.ArgumentParser(description="Smoke-test FastAPI churn endpoints")
	parser.add_argument(
		"--base-url",
		default=DEFAULT_BASE_URL,
		help="FastAPI base URL (default: http://127.0.0.1:8000)",
	)
	args = parser.parse_args()

	base_url = args.base_url.rstrip("/")
	health_url = f"{base_url}/"
	predict_url = f"{base_url}/predict"

	try:
		health_status, health_body = _get_json(health_url)
		print(f"[health] status={health_status} body={health_body}")

		predict_status, predict_body = _post_json(predict_url, SAMPLE_CUSTOMER)
		print(f"[predict] status={predict_status} body={predict_body}")

		if health_status == 200 and health_body.get("status") == "ok" and "prediction" in predict_body:
			print("PASS: FastAPI health and predict endpoints are working.")
			return 0

		print("FAIL: Endpoint response format/status was not as expected.")
		return 1

	except error.HTTPError as e:
		body = e.read().decode("utf-8", errors="replace")
		print(f"HTTP error: status={e.code} body={body}")
		return 1
	except Exception as e:
		print(f"Request failed: {e}")
		return 1


if __name__ == "__main__":
	sys.exit(main())

