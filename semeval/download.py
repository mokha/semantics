import argparse
from .common import *

ZENODO_BASEURL = "https://zenodo.org/record/5156189/files/"


def _file_name(lang, model):
    if model == "sentiment":
        return "sentiment_model.pt"
    elif model == "embeddings":
        if lang not in supported_languages():
            raise (BaseException("Language not supported!"))
        return "vectors-{}.txt".format(lang)
    else:
        raise (BaseException("Unknown model type " + model))


def download_relatedness_model(language):
    relatedness_models = {
        'eng': {
            'rows': 'https://zenodo.org/record/4633293/files/rows.txt',
            'cols': 'https://zenodo.org/record/4633293/files/cols.txt',
            'model': 'https://zenodo.org/record/4633293/files/model_gzip.hkl'
        },
        'fin': {
            'rows': 'https://zenodo.org/record/3473456/files/unigrams_sorted_5k.txt',
            'cols': 'https://zenodo.org/record/3473456/files/unigrams_sorted_5k.txt',
            'model': 'https://zenodo.org/record/3473456/files/rel_matrix_n_csr.hkl'
        }
    }
    if language not in relatedness_models:
        raise (BaseException("Language not supported!"))

    for _f_name, _link in relatedness_models[language].items():
        _ext = _link.split('.')[-1]
        download_file("{}?download=1".format(_link),
                      "{}{}-relatedness-{}.{}".format(download_path(), language, _f_name, _ext),
                      show_progress=True)


def main(languages, models):
    os.makedirs(script_path("data"), exist_ok=True)
    for language in languages:
        for model in models:
            if model == 'relatedness':
                download_relatedness_model(language)
            else:
                file = _file_name(language, model)
                print("Downloading", model, "for", language)
                download_file(ZENODO_BASEURL + file + "?download=1", download_path() + file,
                              show_progress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='semeval download models')
    parser.add_argument('-l', '--languages', nargs='+', help='<Required> languages to download', required=True)
    parser.add_argument('-m', '--models', nargs='+', help='<Required> models to download', required=True)
    args = parser.parse_args()

    main(args.languages, args.models)
