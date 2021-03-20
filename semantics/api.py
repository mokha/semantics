from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
import argparse
import ujson as json
from .embeddings import Embeddings


def server(*args, **kwargs):
    models = {}
    app = FastAPI()

    for lang in kwargs.get('languages'):
        models[lang] = Embeddings(lang)

    def filter_words(lang, words):
        if words is None:
            return
        return [word for word in words if word in models[lang].L.vocab]

    @app.get("/n_similarity/")
    def n_similarity(lang: str, ws1: List[str] = Query([]), ws2: List[str] = Query([])):
        try:
            ws1 = filter_words(lang, ws1)
            ws2 = filter_words(lang, ws2)
            return models[lang].L.n_similarity(ws1, ws2)
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/similarity/")
    def similarity(lang: str, w1: str, w2: str):
        try:
            res = {'w1': w1, 'w2': w2, 'score': models[lang].L.similarity(w1, w2)}
            return JSONResponse(json.dumps(res))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/most_similar/")
    def most_similar(lang: str, positive: List[str] = Query([]), negative: List[str] = Query([]), topn: int = 10):
        try:
            positive = filter_words(lang, positive)
            negative = filter_words(lang, negative)
            res = models[lang].L.most_similar_cosmul(positive=positive, negative=negative, topn=topn)
            return JSONResponse(json.dumps(res))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/neighbours/")
    def neighbours(lang: str, word: str, threshold: float = 0.8):
        try:
            return JSONResponse(models[lang].neighbours_threshold(word, threshold))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/model_word_set/")
    def model_word_set(lang: str):
        try:
            res = models[lang].L.index2word
            return JSONResponse(json.dumps(res))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/vector/")
    def vector(lang: str, tokens: List[str] = Query([])):
        try:
            res = models[lang].to_vector(tokens)
            return JSONResponse(json.dumps(res.tolist()))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/theme/")
    def theme(lang: str, words: List[str] = Query([])):
        try:
            words = filter_words(lang, words)
            res = models[lang].theme(words)
            return JSONResponse(json.dumps(res))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/centroid/")
    def centroid(lang: str, words: List[str] = Query([])):
        try:
            words = filter_words(lang, words)
            res = models[lang].centroid(words)
            return JSONResponse(json.dumps(res.tolist()))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/analogy/")
    def analogy(lang: str, a: str, b: str, c: str, topn: int = 10):
        try:
            return JSONResponse(models[lang].analogy(a, b, c, topn))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/model/")
    def model(lang: str, word: str):
        try:
            res = models[lang].L[word]
            return JSONResponse(json.dumps(res.tolist()))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    @app.get("/align/")
    def align(lang1: str, lang2: str, word: str, topn: int = 10):
        try:
            res = Embeddings.align(models[lang1], models[lang2], word=word, topn=topn)
            return JSONResponse(json.dumps(res))
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-l', '--languages', nargs='+', help='Languages to load!', required=True)
    p.add_argument("--host", default=None, help="Hostname (default: localhost)")
    p.add_argument("--port", default=None, help="Port (default: 1337)")
    args = p.parse_args()

    uvicorn.run(server(**args.__dict__),
                host=args.host if args.host else "localhost",
                port=int(args.port) if args.port else 1337)
