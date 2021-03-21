from semantics import Embeddings

e = Embeddings("eng")
"""
print(e.theme(['shoe', 'clothes']))
print(e.neighbours('hi'))
print(e.analogy('man', 'king', 'woman'))
print(e.centroid(['hi', 'hello']))
print(e.to_vector('this is a great api !'.split(' ')))


e2 = Embeddings("fin")
print(e.project('king', e2))

print(e.similarity('hi', 'bye'))

print(e.most_similar(positive=['hello', 'world'], negative=['king'], topn=10))
print(e.vector("king"))
"""
print(e.vocabulary())
