import os
import json
import torch
import transformers
import scipy as sp
import numpy as np
import enchant
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_json(d, out_path):
    with open(out_path, "wt") as f:
        json.dump(d, f)


def tokenize_and_embed(words, layer):
    global tokenizer_time
    global inference_time
    global use_pooler
    global device

    start = time.time()
    tokens = tokenizer(words, return_tensors="pt", padding=True).to(device)
    end = time.time()
    tokenizer_time += end - start

    start = time.time()
    # shape is [batch size, sequence length, model dim]
    outs = model(**tokens,
                 output_hidden_states=True)  # all hidden layers and initial embedding layer. outs["hidden_states"][-1] == outs["last_hidden_state"]
    if use_pooler:
        embeddings = outs["pooler_output"]
        out = embeddings.cpu().numpy()  # pooler is always correct dim, don't need to do anything else
    else:
        embeddings = outs["hidden_states"][layer][:, 1:-1]  # choose hidden state, and then drop start and end tokens
        out = (torch.mean(embeddings, 1)).squeeze().cpu().numpy()
    end = time.time()
    inference_time += end - start
    return out


def make_vocab(out_path, max_vocab_size, only_alpha=True):
    d = enchant.Dict("en_US")
    vocab = {}
    n_words = 0
    with open("vocab.txt", "rt", errors="replace") as f:
        for line in f:
            if n_words == max_vocab_size:
                break
            else:
                split = line.split()
                word, count = split[0], split[1]
                if only_alpha:
                    if word.isalpha() and d.check(word):
                        vocab[word] = count
                        n_words += 1
                else:
                    vocab[word] = count
                    n_words += 1
    return vocab


# take vocab dict and return dict with dim reduced embeddings
def make_embeddings(vocab, embedder, layer, batch_size=1):
    global sklearn_time
    n_words = len(vocab)
    embeddings = np.zeros((n_words, embedding_dim))
    words = ["" for _ in range(batch_size)]
    for i, k in enumerate(vocab.keys()):
        words[i % batch_size] = k
        nth_word = i + 1
        if (nth_word) % batch_size == 0:  # for now just assume batch_size divides vocab
            e = embedder(words, layer)  # e is numpy array with embeddings for words
            embeddings[nth_word - batch_size: nth_word] = e

    start = time.time()
    pca_embeddings = PCA(n_components=pca_components).fit_transform(embeddings)
    tsne_embeddings = TSNE(n_components=tsne_components).fit_transform(embeddings)
    end = time.time()
    sklearn_time += end - start

    d = {}
    for i, w in enumerate(vocab.keys()):
        x = tsne_embeddings[i][0]
        y = tsne_embeddings[i][1]
        d[w] = {"idx": int(i % 142), "x": float(x),
                "y": float(y)}  # 142 is because there are 142 colors in the json file, will have to address later
    return d


tokenizer_time = 0
inference_time = 0
sklearn_time = 0

if __name__ == "__main__":
    # model = transformers.BertModel.from_pretrained("bert-base-uncased")
    # model.save_pretrained("bert_model")
    # tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer.save_pretrained("bert_tokenizer")
    # max_vocab_size = 96000
    # vocab_path = "vocab_eng_" + str(int(max_vocab_size)) + ".json"
    # vocab_json = make_vocab(vocab_path, max_vocab_size)
    # save_json(vocab_json, vocab_path)

    batch_size = 32
    max_vocab_size = batch_size * 3000
    pca_components = 50
    tsne_components = 2
    embedding_dim = 768
    use_pooler = False
    model_path = "bert_model"
    tokenizer_path = "bert_tokenizer"
    vocab_path = "vocab_eng_" + str(int(max_vocab_size)) + ".json"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layers_of_interest = [0, -1, -2, -10]
    for layer in layers_of_interest:
            suffix = "pooler" if use_pooler else str(layer)
            embedding_path = "embeddings_" + str(int(max_vocab_size)) + "_" + suffix + ".json"
            with torch.no_grad():
                model = transformers.AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
                tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                with open(vocab_path, "rt") as f:
                    vocab = json.load(f)
                    start = time.time()
                    embeddings = make_embeddings(vocab, tokenize_and_embed, layer, batch_size=batch_size)
                    end = time.time()
                    save_json(embeddings, embedding_path)
                    print("Total embedding time was " + str(end - start))
                    print("sklearn time was" + str(sklearn_time))
                    print("Inference time was" + str(inference_time))
                    print("tokenization time was" + str(tokenizer_time))
