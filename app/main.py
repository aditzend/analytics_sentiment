from fastapi import FastAPI
from pydantic import BaseModel
from pysentimiento import create_analyzer
import numpy as np


sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
emotion_analyzer = create_analyzer(task="emotion", lang="es")
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")
ner_analyzer = create_analyzer(task="ner", lang="es")
pos_analyzer = create_analyzer(task="pos", lang="es")

# s1 = sentiment_analyzer.predict("Me encanta la vida")
# print(s1)


class Message(BaseModel):
    text: str
    description: str | None = None


# NER type:
# <class 'list' >
# [{'score': 0.99935895, 'word': 'alexander', 'start': 0, 'end': 9, 'text': 'Alexander', 'type': 'PER'}, {'score': 0.9988066, 'word': 'montevideo', 'start': 18,
#                                                                                                         'end': 29, 'text': 'Montevideo', 'type': 'LOC'}, {'score': 0.8645547, 'word': 'susana,', 'start': 50, 'end': 58, 'text': 'Susana,', 'type': 'PER'}]

class NER(BaseModel):
    score: float
    word: str
    start: int
    end: int
    text: str
    type: str


app = FastAPI()


@app.post("/ner/")
# Just NER
async def analyze_ner(message: Message) -> list[NER]:
    ner: list[NER] = ner_analyzer.predict(message.text)
    return ner


@app.post("/full/")
# Includes all processes
async def analyze_full(message: Message):
    sentiment = sentiment_analyzer.predict(message.text)
    emotion = emotion_analyzer.predict(message.text)
    hate_speech = hate_speech_analyzer.predict(message.text)
    return {
        "sentiment": sentiment,
        "emotion": emotion,
        "hate_speech": hate_speech,
    }

# POS type:
# <class 'list' >
# [{'score': 0.9999269, 'word': 'Alexander', 'start': 0, 'end': 9, 'text': 'Alexander', 'type': 'PROPN'}, {'score': 0.9996958, 'word': 'se', 'start': 9, 'end': 12, 'text': 'se', 'type': 'PRON'}, {'score': 0.99991393, 'word': 'fue', 'start': 12, 'end': 16, 'text': 'fue', 'type': 'VERB'}, {'score': 0.99981767, 'word': 'a', 'start': 16, 'end': 18, 'text': 'a', 'type': 'ADP'}, {'score': 0.9999155, 'word': 'Montevideo', 'start': 18, 'end': 29, 'text': 'Montevideo', 'type': 'PROPN'}, {'score': 0.9998603, 'word': 'y', 'start': 29, 'end': 31, 'text': 'y', 'type': 'CONJ'}, {'score': 0.99975497, 'word': 'me', 'start': 31, 'end': 34, 'text': 'me', 'type': 'PRON'}, {'score': 0.99982446, 'word': 'aten', 'start': 34, 'end': 39, 'text': 'aten', 'type': 'VERB'},
#     {'score': 0.99270684, 'word': 'dio', 'start': 39, 'end': 42, 'text': 'dio', 'type': 'VERB'}, {'score': 0.90868753, 'word': 'una', 'start': 42, 'end': 46, 'text': 'una', 'type': 'DET'}, {'score': 0.93186647, 'word': 'tal', 'start': 46, 'end': 50, 'text': 'tal', 'type': 'ADJ'}, {'score': 0.9999362, 'word': 'Sus', 'start': 50, 'end': 54, 'text': 'Sus', 'type': 'PROPN'}, {'score': 0.2752841, 'word': 'ana,', 'start': 54, 'end': 58, 'text': 'ana,', 'type': 'NOUN'}, {'score': 0.9128537, 'word': 'barba', 'start': 58, 'end': 64, 'text': 'barba', 'type': 'NOUN'}, {'score': 0.9996092, 'word': 'la', 'start': 64, 'end': 67, 'text': 'la', 'type': 'DET'}, {'score': 0.99407214, 'word': 'atencion', 'start': 67, 'end': 76, 'text': 'atencion', 'type': 'NOUN'}]


class POS(BaseModel):
    score: float
    word: str
    start: int
    end: int
    text: str
    type: str


@app.post("/pos/")
# Just POS
async def analyze_pos(message: Message) -> list[POS]:
    pos: list[POS] = pos_analyzer.predict(message.text)
    return pos
