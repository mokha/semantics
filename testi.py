from semantics import Embeddings

e = Embeddings("eng")
print(e.theme(words=['shoe', 'shoes', 'clothes'], lang='eng'))
print(e.neighbours(word='hi', threshold=0.4, lang='eng'))
print(e.analogy('man', 'king', 'woman', topn=10, lang='eng'))
print(e.centroid(words=['hi', 'hello'], lang='eng'))
print(e.vector(tokens='this is a great api !'.split(' '), lang='eng'))
print(e.align(word='king', lang1='eng', lang2='fin'))
print(e.similarity(w1='hi', w2='bye', lang='eng'))
print(e.most_similar(positive=['hello', 'world'], negative=['king'], topn=10, lang='eng'))
print(e.model(word='king', lang='eng'))