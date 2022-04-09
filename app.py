from typing import List, Optional

from fastapi import FastAPI, Body, HTTPException, status
import uvicorn
from pydantic import BaseModel
from fastapi_camelcase import CamelModel

from source.spell_checker import SpellChecker

app = FastAPI()

spell_checker_handler = None


class Sentence(BaseModel):
    sentences: List[str]


class SpellCheckerInnerResponse(CamelModel):
    sentence: str
    correction: str


class SpellCheckerResponse(CamelModel):
    response: Optional[List[SpellCheckerInnerResponse]] = None


@app.post("/spell-checker/", response_model=SpellCheckerResponse)
async def spell_checker(sentences: Sentence):
    response = []
    try:
        for sentence in sentences.sentences:
            corrected_sentence = spell_checker_handler.prediction(sentence=sentence.split(" "), k=200,
                                                                  levenshtein_ratio_threshold=0.5)
            response.append({
                "sentence": sentence,
                "correction": " ".join(corrected_sentence)
            })
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,
                            detail="Any Image not uploaded")
    return {
        "response": response
    }
    # return \
    #     {
    #         "response":
    #             [
    #                 {
    #                     "sentence": "তাূরা দেখেন ঢাকার দূই সিটি করপোরেশনে মশা মারতে যে ওষধূ ছিটানো হয় তা অকাযংকর",
    #                     "correction": "তারা দেখেন ঢাকার দুই সিটি করপোরেশনে মশা মারতে যে ওষুধ ছিটানো হয় তা অকাযংকর"
    #                 }
    #             ]
    #     }


@app.on_event("startup")
async def startup_event():
    global spell_checker_handler
    spell_checker_handler = SpellChecker()


if __name__ == '__main__':
    uvicorn.run(app='app:app', reload=True, port=7003, host="127.0.0.1")
