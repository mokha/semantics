import json
import requests


class ApiModel:
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
            result = json.loads(content)
            result['score'] = float(result['score'])
            return result
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

    def model(self, word, lang='eng'):
        url = self.baseurl + '/model'
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

    def vector(self, tokens, lang='eng'):
        url = self.baseurl + '/vector'
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

    def get_response(self, url, params={}):
        response = requests.get(url, params)

        if response.status_code == 200:
            return response.content
        else:
            return None


def test():
    api = ApiModel()
    print(api.theme(words=['shoe', 'shoes', 'clothes'], lang='eng'))
    print(api.neighbours(word='hi', threshold=0.4, lang='eng'))
    print(api.analogy('man', 'king', 'woman', topn=10, lang='eng'))
    print(api.centroid(words=['hi', 'hello'], lang='eng'))
    print(api.vector(tokens='this is a great api !'.split(' '), lang='eng'))
    print(api.align(word='king', lang1='eng', lang2='fin'))
    print(api.similarity(w1='hi', w2='bye', lang='eng'))
    print(api.most_similar(positive=['hello', 'world'], negative=['king'], topn=10, lang='eng'))
    print(api.model(word='king', lang='eng'))
    # print(api.model_word_set(lang='eng'))


if __name__ == '__main__':
    test()
