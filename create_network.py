from socket import create_connection
import nltk 
nltk.download('stopwords')

from tqdm import tqdm

import random
import numpy as np
import networkx as nx
import argparse

import pickle

random.seed(42)

APPS = ['eBay', 'WhatsApp', 'Facebook', 'Evernote', 'Twitter', 'Netflix', 'PhotoEditor', 'Spotify']
SW = nltk.corpus.stopwords.words('english')
TOKEN='$t$'

def get_sequence_to_token(sequence_to_tokens, lines):
    for line_idx in range(0, len(lines), 3):
        
        sentence  = lines[line_idx].strip().lower()
        token     = lines[line_idx + 1].strip().lower()
        iob_class = int(lines[line_idx + 2])

        full_sentence = sentence.replace(TOKEN, token)

        if full_sentence in sequence_to_tokens:
            sequence_to_tokens[full_sentence].append((token, iob_class))
        else:
            sequence_to_tokens[full_sentence] = [(token, iob_class)]        

def negative_sampling(Graph, sequence_to_tokens, n_positive_req):
    negatives = []
    
    for sentence, tokens_info in sequence_to_tokens.items():
        for token, iob_class in tokens_info:
            if iob_class == -1 and token not in SW:
                negatives.append((token, sentence))

    ns = random.sample(negatives, n_positive_req)
    for token, sentence in ns:
        Graph.add_node(token, iob_class=-1, type='aspect_node:negative_sampling', dataset='train')
        Graph.add_edge(sentence, token)

def populate_train_data(Graph, file_path='./dataset_train',):
    sequence_to_tokens = {}

    for app in APPS:
        lines = []
        with open(f"{file_path}/train_{app}.txt" , 'r') as f:
            lines = f.readlines()

        get_sequence_to_token(sequence_to_tokens, lines)
        
    positive_req = 0
    for sentence, tokens in sequence_to_tokens.items():
        Graph.add_node(sentence, type='sentence_node', dataset='train')
        for token, iob_class in tokens:
            if iob_class != -1 and token not in SW:
                positive_req += 1
                Graph.add_node(token, iob_class=1, type='aspect_node', dataset='train')
                Graph.add_edge(sentence, token)
    
    negative_sampling(Graph, sequence_to_tokens, positive_req)

def populate_test_layer(Graph, path_test, path_model_pred):
    for model_app in tqdm(APPS):
        APPS.remove(model_app)
        Graph.add_node(model_app, type='model_node')
        
        for app in APPS:
            with open(f"{path_model_pred}/{model_app}/{model_app}_model_on_{app}.txt") as model_pred_fp, open(f"{path_test}/test_data_{app}.txt") as test_fp:
                test_sentences = test_fp.readlines()    
                model_preds    = model_pred_fp.readlines()

            for sentence, extracted_data in zip(test_sentences, model_preds):
                sentence = sentence.strip().lower()
                if sentence not in Graph.nodes(): 
                    Graph.add_node(sentence, type='sentence_node', dataset='test')
                
                tmp = extracted_data.split(',')[0].split(';')
                extracted_requirements = [req.strip().lower() for req in tmp]
                if not (extracted_requirements[0] == '' and len(extracted_requirements) == 1):
                    for req in extracted_requirements:
                        if req not in Graph.nodes():
                            Graph.add_node(req, iob_class = 1, type='aspect_node', dataset='test')
                        Graph.add_edge(sentence.strip().lower(), req)
                        Graph.add_edge(model_app, req)
            
        APPS.insert(0, model_app)

def create_graph(opt):
    G = nx.Graph()
    
    # Create train dataset layer + negative sampling
    populate_train_data(G, opt.train_folder)

    # Create test dataset layer
    populate_test_layer(G, opt.test_folder, opt.models_pred_folder)

    return G

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', default='./dataset_train', type=str)
    parser.add_argument('--test_folder', default='./datasets_iob', type=str)
    parser.add_argument('--models_pred_folder', default='./models_predictions', type=str)
    parser.add_argument('--dump_graph_path', default='./multiple_models_graph.p')
    opt = parser.parse_args()

    graph = create_graph(opt)

    with open(opt.dump_graph_path, 'wb') as f:
        pickle.dump(graph, f) 
    
    g = None
    with open(opt.dump_graph_path, 'rb') as f:
        g = pickle.load(f)

    for n in g.nodes():
        print(n, g.nodes[n])

    print(g.number_of_nodes())


if __name__ == '__main__':
    main()