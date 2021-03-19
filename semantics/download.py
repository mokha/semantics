from mikatools import *
import os
import argparse

ZENODO_BASEURL = "https://zenodo.org/record/4624114/files/"


def _file_name(lang, model):
    if model == "sentiment":
        return "sentiment_model.pt"
    elif model == "vectors":
        return "vectors-" + lang + ".txt"
    else:
        raise (BaseException("Unknown model type " + model))


def main(languages, models):
    os.makedirs(script_path("data"), exist_ok=True)
    for language in languages:
        for model in models:
            file = _file_name(language, model)
            print("Downloading", model, "for", language)
            download_file(ZENODO_BASEURL + file + "?download=1", script_path("data/" + file), show_progress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='semantics download models')
    parser.add_argument('-l', '--languages', nargs='+', help='<Required> languages to download', required=True)
    parser.add_argument('-m', '--models', nargs='+', help='<Required> models to download', required=True)
    args = parser.parse_args()

    main(args.languages, args.models)
