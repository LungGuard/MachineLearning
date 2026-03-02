from fastapi import FastAPI


app = FastAPI() 

@app.post("/api/analyze")
def analyze():
    pass