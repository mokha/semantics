# Semeval (Semantic Evaluation)

Semeval is a Python package for modelling various aspects of meaning in languages.

# Installation

The package can be installed via PIP: `pip install -U semeval`

## Download models

The following command downloads word vector embeddings for English and Finnish

	python3 -m semeval.download -l eng fin -m embeddings

We currently suppor eng, fin, kpv, myv, mdf, rus, liv and sms.

Relatedness models can be downloaded using the command:

    python3 -m semeval.download -l eng fin -m relatedness
    
Currently, only English and Finnish are supported in the relatedness model.

# Usage

## Word embeddings

Load embeddings of a language by running:

	from semeval import Embeddings
	e = Embeddings("eng")

After this, you can find related words

	e.theme(['shoe', 'clothes']) #outputs a word describing all input words
	>> ('clothing', 0.8700831532478333)

	e.neighbours('hi') #outputs the nearest neighbors
	>> [('hey', 0.6976003050804138), ('jeez', 0.6230848431587219), ('ya', 0.6213312149047852), ('hello', 0.6144036650657654) ...]

	e.analogy('man', 'king', 'woman') #analogous words
	>> [('monarch', 0.6457855105400085), ('regnant', 0.6354122161865234)...]

	e.most_similar(positive=['hello', 'world'], negative=['king'], topn=10)
	>> [('Tamana', 1.1119599342346191), ('Ibaraki', 1.087041974067688), ('Makuhari', 1.0793628692626953),... ]

Get vector representations

	e.centroid(['hi', 'hello']) #the centroid vector of the input words
	>> [0. 0.10486 -0.23701501 0.1293 0.052645 -0.027155...]

	e.to_vector('this is a great api !'.split(' ')) # a vector representing the sentence
	>> [1.1630000e-02 -2.2303334e-02  8.4316663e-02 -2.1063333e-02...]

	e.vector("king")
	>> [0.05574 -0.16716 0.10282 -0.10851 0.08783 -0.09499 0.16031...]

Get similar words in another language

	e2 = Embeddings("fin")
	e.project('king', e2)
	>>[('Ahasveros', 0.5551087856292725), ('kuningas', 0.5522325038909912), ('kuninkas', 0.5242758393287659)..]

Word similarity

	e.similarity('hi', 'bye')
	>> 0.28805155

Vocabulary

	e.vocabulary()
	>> {'astrologically': <Vocab object>, 'spinto': <Vocab object>, 'NortelNet': <Vocab object>...}

### Server-mode
Server-mode is optimal for some cases such debugging or not wanting to wait for multiple models to load. To start word 
embeddings server, run the below command in the terminal: `python -m semeval.server --service embeddings`

Once the server is loaded, the service is accessible through `EmbeddingsAPI` class. 
Note that the language/s must be passed every call, otherwise the server cannot know which model to use. 
Here is an example of accessing the service from Python.

```python
from semeval import EmbeddingsAPI

api = EmbeddingsAPI()
api.theme(words=['shoe', 'clothes'], lang='eng')
api.neighbours(word='hi', threshold=0.4, lang='eng')
api.analogy('man', 'king', 'woman', topn=10, lang='fin')
api.centroid(words=['hi', 'hello'], lang='eng')
api.to_vector(tokens='this is a great api !'.split(' '), lang='eng')
api.align(word='king', lang1='eng', lang2='fin')
api.similarity(w1='hi', w2='bye', lang='eng')
api.most_similar(positive=['hello', 'world'], negative=['king'], topn=10, lang='eng')
api.vector(word='king', lang='eng')
api.vocabulary(lang='eng')
```

## Relatedness
Load the relatedness model:

    from semeval import Relatedness
    m = Relatedness(lang='eng')

Get most 5 related word to the word *car*:

    m.get_sorted_rel('car')[:5]
    >> [('park', 0.20570222), ('insurance', 0.085593514), ('parking', 0.06783264), ('hire', 0.036158927), ('cheap', 0.028782822)]
    
Get relatedness score between two words:

    m.get_rel('car')['insurance']
    >> 0.085593514

Top 10 interpretations for the metaphor "*Alcohol* is a *Crutch*". NOTE: unlike the original paper, there is no filtering (e.g., POS 
filtering) applied in this function. Read the paper for further details and post-processing steps to improve the results.

    m.interpret('alcohol', 'crutch')[:10]
    >> [('use', 0), ('cane', 0), ('smoking', 1), ('psychological', 2), ('dependence', 2), ('emotional', 3), ('cigarette', 3), ('drug', 4), ('bandage', 4), ('week', 5)]

Metaphoricity scores for the tenor *computer*, vehicle *creative* and expression 'The algorithm for painting.'

    m.metaphoricity('computer', 'creative', ['the', 'algorithm', 'for', 'painting'], 300) # 0 to select all
    >> (3.2977940826411467e-07, 0.000510800164192915, 0.00025556497180058955) # (magnitude score, difference score, avg if both positive)
    
# Business solutions

<img src="https://rootroo.com/cropped-logo-01-png/" alt="Rootroo logo" width="128px" height="128px">

When your NLP needs grow out of what UralicNLP can provide, we have your back! [Rootroo offers consulting related to a variety of NLP tasks](https://rootroo.com/). We have a strong academic background in the state-of-the-art AI solutions for every NLP need. Just contact us, we won't bite.
