from fastapi import FastAPI, File, UploadFile
from model_helper import give_prediction
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        prediction, probability = give_prediction(image_path)["class"], give_prediction(image_path)["probability"]
        return {"Prediction": prediction, "Probability":probability}
    except Exception as e:
        e = str(e)
        return {"error": e}