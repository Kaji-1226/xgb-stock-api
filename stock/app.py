# ✅ 文件名：app.py
from fastapi import FastAPI, Response
from data import run_model
import os

app = FastAPI()

@app.get("/signal")
def get_signal():
    try:
        run_model()
        if not os.path.exists("strategy_signals_300750.csv"):
            return Response(content="CSV 文件未找到", status_code=404)

        with open("strategy_signals_300750.csv", "r", encoding="utf-8") as f:
            csv_data = f.read()

        return Response(content=csv_data, media_type="text/csv")
    except Exception as e:
        return Response(content=str(e), status_code=500)
