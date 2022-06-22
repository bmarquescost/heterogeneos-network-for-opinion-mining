import networkx as nx
import pickle

import tensorflow as tf
import numpy as np
import random as rn

from sentence_transformers import SentenceTransformer

import argparse

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding

from tensorflow import keras

from stellargraph import StellarGraph

seed = 42

def get_sentence_embedding(g, model):
    embedding_text = []
    for node in g.nodes():
        if g.nodes[node]['type'] in ['aspect_node', 'aspect_node:negative_sampling', 'sentence_node']:
            embedding_text.append(node)

    embeddings = model.encode(embedding_text)

    node_counter = 0
    for node in g.nodes():
        if g.nodes[node]['type'] in ['aspect_node', 'aspect_node:negative_sampling', 'sentence_node']:
            g.nodes[node]['embedding'] = embeddings[node_counter]
            node_counter += 1

def gcn(G, opt):
    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(G)

    # Randomly sample a fraction p=0.2 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(p=0.2, method="global", keep_connected=True,seed=seed)

    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)

    # Randomly sample a fraction p=0.2 of all positive links, and same number of negative links, from G_test, and obtain the
    # reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(p=0.2, method="global", keep_connected=True,seed=seed)

    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(layer_sizes=[opt.gcn_embedding_size, opt.gcn_embedding_size], activations=["relu", "relu"], generator=train_gen, dropout=opt.dropout)

    x_inp, x_out = gcn.in_out_tensors()
    
    prediction = LinkEmbedding(activation="relu", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=opt.lr),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    history = model.fit(train_flow, epochs=opt.n_epochs, validation_data=test_flow, verbose=2, shuffle=False)

    sg.utils.plot_history(history)

    return model, get_gcn_embeddings(model, test_flow)


def get_gcn_embeddings(model, test_flow):
    embedding_gen = keras.Model(inputs=model.input, outputs=model.layers[7].output)
    return embedding_gen.predict(test_flow)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', default='multiple_models_graph.p', type=str)
    parser.add_argument('--gcn_embedding_size', default=64, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    opt = parser.parse_args()

    with open(f'{opt.graph_path}', 'rb') as f:
            graph = pickle.load(f)

    print("Obtaining sentence embeddings")
    sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    get_sentence_embedding(graph, sentence_model)

    G = StellarGraph.from_networkx(graph,node_features='embedding')

    print(f"GCN training and embeddings from GCN model with size {opt.gcn_embedding_size}")
    model, gcn_embeddings = gcn(G, opt)

    # Further Work: analysis on gcn_embeddings 

if __name__ == '__main__':
    main()