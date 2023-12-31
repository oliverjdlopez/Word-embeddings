This is project is a part of my work with Professor Paul Ginsparg. The code extracts word embeddings from the hidden layers of transformer models. The embeddings are then dimension-reduced with PCA and t-sne so that they can be projected onto the 2D plane.

I wrote the code in make_embeddings.py (everything that deals with the AI models and word embeddings), while credit for map-maker and wvec.html (the visualization and word-search code) goes to Alex Alemi. His website is here: https://www.alexalemi.com

This particular code uses BERT (downloaded from the huggingface hub), but it is written to be able to use any encoder model. A visualization using the second-to-last hidden layer is hosted at https://www.cs.cornell.edu/~ginsparg/embd/