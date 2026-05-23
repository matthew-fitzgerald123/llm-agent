serve:
	uvicorn app.main:app --reload --port 8083

test:
	pytest tests/ -v

demo:
	python notebooks/demo.py
