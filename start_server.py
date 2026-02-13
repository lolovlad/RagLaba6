from uvicorn import run
from main import app


run(app, host="localhost", port=8000)