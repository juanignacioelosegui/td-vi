import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import logging
import gensim
import numpy as np
import random
import os
import pickle
import tqdm

# Descarga de stopwords para español si no están descargadas ya
# nltk.download('stopwords')

STOP_WORDS_SP = set(stopwords.words('spanish'))

def iterate_LN_corpus(path):
    """
    Genera un iterador para recorrer los archivos de texto en un directorio.

    Args:
        path (str): Ruta al directorio que contiene los archivos.

    Yields:
        str: Texto contenido en cada archivo.
    """
    articles = os.listdir(path)
    random.shuffle(articles)
    for art in articles:
        try:
            with open(os.path.join(path, art), encoding="utf-8") as f:
                raw_text = f.read()
            yield raw_text
        except (FileNotFoundError, IOError):
            pass  # Ignora los archivos que no se pueden abrir


def tokenizer(raw_text):
    """
    Tokeniza y preprocesa un texto.

    Args:
        raw_text (str): Texto sin procesar.

    Returns:
        list: Lista de oraciones, donde cada oración es una lista de palabras.
    """
    sentences = sent_tokenize(raw_text)
    sentences = [word_tokenize(e) for e in sentences]
    sentences = [[e2 for e2 in e1 if re.compile("[A-Za-z]").search(e2[0])] for e1 in sentences]
    sentences = [[e2.lower() for e2 in e1] for e1 in sentences]
    return(sentences)


def gen_sentences(path):
    """
    Genera una lista de oraciones a partir de archivos de texto en un directorio.

    Args:
        path (str): Ruta al directorio que contiene los archivos de texto.

    Returns:
        list: Lista de oraciones.
    """
    sentences = []
    n_arts = len(os.listdir(path))
    for i, art in tqdm.tqdm(enumerate(iterate_LN_corpus(path)), total=n_arts):
        sentences.extend(tokenizer(art))
    return(sentences)


def average_vectors(title_tokens, model, stopwords=None):
    """
    Calcula el vector promedio de un conjunto de tokens utilizando un modelo Word2Vec.

    Args:
        title_tokens (list): Lista de tokens.
        model (gensim.models.Word2Vec): Modelo Word2Vec.
        stopwords (set, optional): Conjunto de palabras stopwords. Defaults to None.

    Returns:
        numpy.ndarray: Vector promedio.
    """
    title_tokens = [e2 for e1 in title_tokens for e2 in e1]
    title_tokens = [e for e in title_tokens if e in model.wv]
    if stopwords is not None:
        title_tokens = [e for e in title_tokens if e not in stopwords]
    if len(title_tokens) == 0:
        output = np.zeros(model.wv.vector_size)
    else:
        output = np.array([model.wv.get_vector(e) for e in title_tokens]).mean(0)
    return output


def dummy_tokenizer(text_tokens):
    """
    Tokenizador dummy que simplemente devuelve los tokens de texto sin procesar.

    Args:
        text_tokens (list): Lista de tokens.

    Returns:
        list: Misma lista de tokens de entrada.
    """
    return text_tokens

# Configuración básica del logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt= '%H:%M:%S', level=logging.INFO)

#~ Análisis con datos de La Nación ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

RECALCULATE_SENTENCES = False
if RECALCULATE_SENTENCES:
    sentences = gen_sentences("./scraped_data/")
    with open("ln_tokens_tr.p", "wb") as f:
        pickle.dump(sentences, f)
else:
    with open("ln_tokens_tr.p", "rb") as f:
        sentences = pickle.load(f)

# Veamos qué tiene sentences
sentences[0]
sentences[1]
sentences[-1]

RETRAIN_W2VEC = False
if RETRAIN_W2VEC:
    # Defino los parámetros del modelo
    w2v_ln = gensim.models.Word2Vec(vector_size=50,
                                    window=5,
                                    min_count=5,
                                    negative=5,
                                    sample=0.01,
                                    workers=4,
                                    sg=1)
    
    # Se hace una pasada por el corpus y se crea el vocabulario
    w2v_ln.build_vocab(sentences,
                       progress_per=10000)

    # Se entrena el modelo
    w2v_ln.train(sentences,
                 total_examples=w2v_ln.corpus_count,
                 epochs=20, report_delay=1)

    # Se guarda en disco el modelo
    w2v_ln.save("ln_w2c.model")
else:
    # Se carga el modelo entrenado
    w2v_ln = gensim.models.Word2Vec.load("ln_w2c.model")

# Vectores
w2v_ln.wv.get_vector("gato", norm=True)
w2v_ln.wv.get_vector("banco", norm=True)

# Similitud
w2v_ln.wv.similarity("blanco", 'negro')
w2v_ln.wv.similarity("blanco", 'diario')

# Más similares
w2v_ln.wv.most_similar(positive=["blanco"], topn=5)
w2v_ln.wv.most_similar(positive=["clarín"], topn=5)
w2v_ln.wv.most_similar(positive=["banco"], topn=20)
w2v_ln.wv.most_similar(positive=["incertidumbre"], topn=50)

# Analogías
w2v_ln.wv.most_similar(positive=["abuelo", "mujer"], negative=["hombre"], topn=5)
w2v_ln.wv.most_similar(positive=["abuela", "hombre"], negative=["mujer"], topn=5)
w2v_ln.wv.most_similar(positive=["menem", "radicalismo"], negative=["peronismo"], topn=5)

# Palabra fuera de lugar
w2v_ln.wv.doesnt_match(['galicia', 'bbva', 'romario'])
