from fastapi import FastAPI
from romapi.fast import app as api_app

app = FastAPI()

app.mount("/api", api_app)  # Mount API endpoints

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
