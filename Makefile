.PHONY: test run lint

test:
	pytest tests/ -v --tb=short

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	python -m py_compile app/config.py app/models.py app/main.py
