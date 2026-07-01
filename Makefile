run:
	docker compose up --build

dev:
	uvicorn app.main:app --reload

lint:
	ruff check app/

format:
	ruff format app/

test:
	pytest tests/
