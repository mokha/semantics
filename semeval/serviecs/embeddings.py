from typing import Optional, List
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import ujson as json
from semeval.embeddings import Embeddings
import json
import requests
from semeval.common import *


def EmbeddingsServer(*args, **kwargs):
    models = {}
    app = FastAPI()

    for lang in kwargs.get('languages', []):
        models[lang] = Embeddings(lang)

    def filter_words(lang, words):
        if words is None:
            return
        return [word for word in words if word in models[lang].L.vocab]

    def check_language(lang):
        if lang not in models and lang in supported_languages():
            models[lang] = Embeddings(lang)

    @app.middleware("http")
    async def load_language(request: Request, call_next):
        try:
            for _lang_k in ['lang', 'lang1', 'lang2']:
                if _lang_k in request.query_params:
                    check_language(request.query_params[_lang_k])
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

        response = await call_next(request)
        return response

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
            res = {'w1': w1, 'w2': w2, 'score': models[lang].L.similarity(w1, w2).item()}
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

    @app.get("/to_vector/")
    def to_vector(lang: str, tokens: List[str] = Query([])):
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

    @app.get("/vector/")
    def vector(lang: str, word: str):
        try:
            res = models[lang].vector(word)
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

    @app.get("/vocabulary/")
    def vocabulary(lang: str):
        try:
            return JSONResponse(models[lang].L.index2word)
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    return app


class EmbeddingsAPI:
    def __init__(self, host='http://localhost', port=1337, path=''):
        self.host = host
        self.port = port
        self.path = path
        self.baseurl = host + ':' + str(port) + path

    def n_similarity(self, ws1, ws2, lang='eng'):
        url = self.baseurl + '/n_similarity'
        content = self.get_response(url, {
            'ws1': ws1,
            'ws2': ws2,
            'lang': lang
        })
        return json.loads(content)

    def similarity(self, w1, w2, lang='eng'):
        url = self.baseurl + '/similarity'
        content = self.get_response(url, {
            'w1': w1,
            'w2': w2,
            'lang': lang
        })
        if content:
            return json.loads(content)
        return None

    def most_similar(self, positive, negative=[], topn=10, lang='eng'):
        url = self.baseurl + '/most_similar'
        content = self.get_response(url, {
            'positive': positive,
            'negative': negative,
            'topn': topn,
            'lang': lang
        })
        return json.loads(content)

    def vector(self, word, lang='eng'):
        url = self.baseurl + '/vector'
        content = self.get_response(url, {
            'word': word,
            'lang': lang
        })

        return json.loads(content)

    def model_word_set(self, lang='eng'):
        url = self.baseurl + '/model_word_set'
        content = self.get_response(url, {'lang': lang})
        return json.loads(content)

    def theme(self, words, lang='eng'):
        url = self.baseurl + '/theme'
        content = self.get_response(url, {'words': words, 'lang': lang})
        return json.loads(content)

    def neighbours(self, word, threshold=0.5, lang='eng'):
        url = self.baseurl + '/neighbours'
        content = self.get_response(url, {
            'word': word,
            'threshold': threshold,
            'lang': lang
        })

        return json.loads(content)

    def align(self, word, lang1='eng', lang2='fin'):
        url = self.baseurl + '/align'
        content = self.get_response(url, {
            'word': word,
            'lang1': lang1,
            'lang2': lang2
        })
        return json.loads(content)

    def centroid(self, words, lang='eng'):
        url = self.baseurl + '/centroid'
        content = self.get_response(url, {'words': words, 'lang': lang})
        return json.loads(content)

    def to_vector(self, tokens, lang='eng'):
        url = self.baseurl + '/to_vector'
        content = self.get_response(url, {'tokens': tokens, 'lang': lang})

        return json.loads(content)

    def analogy(self, a, b, c, topn=10, lang='eng'):
        url = self.baseurl + '/analogy'
        content = self.get_response(url, {
            'a': a,
            'b': b,
            'c': c,
            'topn': topn,
            'lang': lang
        })

        return json.loads(content)

    def vocabulary(self, lang='eng'):
        url = self.baseurl + '/vocabulary'
        content = self.get_response(url, {'lang': lang})
        return json.loads(content)

    def get_response(self, url, params={}):
        response = requests.get(url, params)
        if response.status_code != 200:
            content = json.loads(response.content)
            if 'error' in content:
                raise Exception(content['error'])
            else:
                raise Exception("Error... Response received: {}".format(content))
        return response.content


def test():
    api = EmbeddingsAPI()
    print(api.theme(words=['shoe', 'clothes'], lang='eng'))
    print(api.neighbours(word='hi', threshold=0.4, lang='eng'))
    print(api.analogy('man', 'king', 'woman', topn=10, lang='fin'))
    print(api.centroid(words=['hi', 'hello'], lang='eng'))
    print(api.to_vector(tokens='this is a great api !'.split(' '), lang='eng'))
    print(api.align(word='king', lang1='eng', lang2='fin'))
    print(api.similarity(w1='hi', w2='bye', lang='eng'))
    print(api.most_similar(positive=['hello', 'world'], negative=['king'], topn=10, lang='eng'))
    print(api.vector(word='king', lang='eng'))
    print(api.vocabulary(lang='eng'))
    # print(api.model_word_set(lang='eng'))


if __name__ == '__main__':
    test()
