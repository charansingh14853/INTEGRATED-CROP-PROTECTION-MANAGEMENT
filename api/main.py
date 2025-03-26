from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL1 = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES1 = ["Early Blight", "Healthy", "Late Blight"]

MODEL2 = tf.keras.models.load_model("../models/2")
CLASS_NAMES2 = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]

MODEL3 = tf.keras.models.load_model("../models/3")
CLASS_NAMES3 = ['aphids','armyworm','beetle','bollworm','earthworm','grasshopper','mites','mosquito','sawfly','stem_borer']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def read_file_as_image2(data) -> np.ndarray:
    image2 = np.array(Image.open(BytesIO(data)))
    return image2


def read_file_as_image3(data) -> np.ndarray:
    image3 = np.array(Image.open(BytesIO(data)))
    return image3


@app.post("/Potato")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL1.predict(img_batch)
    predicted_class = CLASS_NAMES1[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.post("/Apple")
async def predict2(
        file2: UploadFile = File(...)
):
    image2 = read_file_as_image2(await file2.read())
    img_batch2 = np.expand_dims(image2, 0)

    predictions2 = MODEL2.predict(img_batch2)
    predicted_class2 = CLASS_NAMES2[np.argmax(predictions2[0])]
    confidence2 = np.max(predictions2[0])
    return {
        'class': predicted_class2,
        'confidence': float(confidence2)
    }


@app.post("/Pests")
async def predict3(
        file3: UploadFile = File(...)
):
    image3 = read_file_as_image2(await file3.read())
    img_batch3 = np.expand_dims(image3, 0)

    predictions3 = MODEL3.predict(img_batch3)
    predicted_class3 = CLASS_NAMES3[np.argmax(predictions3[0])]
    confidence3 = np.max(predictions3[0])
    return {
        'class': predicted_class3,
        'confidence': float(confidence3)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
