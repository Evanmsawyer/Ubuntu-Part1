import json
import argparse
import os
from typing import Dict, List, Tuple
import pyterrier as pt
if not pt.started():
    pt.init()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ranx
import time
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
import re

from clean_inputs import clean_text, clean_html, clean_json_file

warnings.filterwarnings("ignore", category=UserWarning)

class BM25FRetriever:
    def __init__(self, answers_file: str):
        self.answers = self.load_answers(answers_file)
        self.indexref = self.build_index()
        self.bm25f = self.create_bm25f()
        self.qe = pt.rewrite.Bo1QueryExpansion(self.indexref)
        self.retrieval_pipeline = self.bm25f >> self.qe >> self.bm25f
        self.word2vec_model = self.train_word2vec()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def load_answers(self, file_path: str) -> List[Dict]:
        if not self.validate_json(file_path):
            raise ValueError("Invalid JSON format in Answers.json")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Answers.json should contain a list of answer objects")

        processed_data = []
        for item in data:
            try:
                processed_item = {
                    'docno': str(item['Id']),
                    'text': self.preprocess_text(item.get('Text', '')),
                    'score': item.get('Score', 0)
                }
                if processed_item['text']:  # Only add if text is not empty
                    processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing item: {item.get('Id', 'Unknown ID')}")
                print(f"Error details: {str(e)}")
                print(f"Item content: {item}")
                continue
        
        if not processed_data:
            raise ValueError("No valid answers were processed. Check the format of your Answers.json file.")
        
        return processed_data

    def preprocess_text(self, text: str) -> str:
        text = clean_html(text)
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def build_index(self):
        home_dir = os.path.expanduser("~")
        index_path = os.path.join(home_dir, "askubuntu_index")
        os.makedirs(index_path, exist_ok=True)
        print(f"Creating index in: {index_path}")
        
        df = pd.DataFrame(self.answers)
        
        indexer = pt.index.IterDictIndexer(index_path, overwrite=True)
        indexref = indexer.index(df.to_dict('records'))
        
        return indexref

    def create_bm25f(self):
        return pt.BatchRetrieve(self.indexref, wmodel="BM25")

    def train_word2vec(self):
        sentences = [word_tokenize(answer['text']) for answer in self.answers]
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def expand_query(self, query: str, topn: int = 2) -> str:
        tokens = word_tokenize(self.preprocess_text(query))
        expanded_terms = tokens.copy()
        for token in tokens[:5]:  # Limit expansion to first 5 terms
            try:
                similar_words = self.word2vec_model.wv.most_similar(token, topn=topn)
                expanded_terms.extend([word for word, _ in similar_words])
            except KeyError:
                continue  # Word not in vocabulary
        return ' '.join(list(set(expanded_terms)))[:1000]  # Limit query length to 1000 characters

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        expanded_query = self.expand_query(query)
        results = self.retrieval_pipeline.search(expanded_query)
        return [(str(row['docno']), float(row['score'])) for _, row in results.iterrows()][:k]
    
    def validate_json(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except json.JSONDecodeError as e:
            print(f"JSON validation failed: {str(e)}")
            print(f"Error occurred at line {e.lineno}, column {e.colno}")
            return False



def load_topics(file_path: str) -> Dict[str, Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = {}
    for item in data:
        cleaned_item = {
            'Id': item['Id'],
            'Title': clean_text(item['Title']),
            'Body': clean_text(item['Body']),
            'Tags': [clean_text(tag) for tag in item['Tags']]
        }
        cleaned_data[item['Id']] = cleaned_item
    
    return cleaned_data
    

def write_results(results: Dict[str, List[Tuple[str, float]]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores, 1):
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score}\tAdvancedBM25\n")

def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    qrels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            topic_id, _, doc_id, relevance = line.strip().split()
            if topic_id not in qrels:
                qrels[topic_id] = {}
            qrels[topic_id][doc_id] = int(relevance)
    return qrels

def evaluate_results(qrels, run_results):
    ranx_qrels = ranx.Qrels.from_dict(qrels)
    ranx_run = ranx.Run.from_dict(run_results)
    metrics = ["map", "mrr", "ndcg@5", "ndcg@10", "precision@1", "precision@5", "precision@10", "recall@100"]
    eval_results = ranx.evaluate(ranx_qrels, ranx_run, metrics)
    return eval_results

def plot_evaluation_results(eval_results: Dict[str, float], topics_file: str):
    metrics = list(eval_results.keys())
    scores = list(eval_results.values())

    plt.figure(figsize=(12, 6))
    plt.bar(metrics, scores, color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title(f'Evaluation Metrics for {topics_file} (Advanced BM25)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.4f}", ha='center', va='bottom', rotation=45)
    plt.tight_layout()
    plt.savefig(f"evaluation_plot_advanced_bm25_{os.path.splitext(os.path.basename(topics_file))[0]}.png")
    plt.close()

def plot_precision_at_5_bar_plot(qrels, run_results, topics_file: str):
    topic_precision = {}
    for topic_id in run_results:
        if topic_id in qrels:
            relevant_docs = set(doc_id for doc_id, rel in qrels[topic_id].items() if rel > 0)
            retrieved_docs = list(run_results[topic_id].keys())[:5]
            relevant_retrieved = len([doc for doc in retrieved_docs if doc in relevant_docs])
            topic_precision[topic_id] = relevant_retrieved / 5
    
    sorted_precision = sorted(topic_precision.items(), key=lambda x: x[1], reverse=True)
    num_topics = min(70, len(sorted_precision))
    
    segment_size = len(sorted_precision) // 7
    selected_topics = []
    
    for i in range(7):
        start = i * segment_size
        end = (i + 1) * segment_size if i < 6 else len(sorted_precision)
        segment = sorted_precision[start:end]

        indices = np.linspace(0, len(segment) - 1, 10, dtype=int)
        selected_topics.extend([segment[j] for j in indices])
    
    selected_topics = selected_topics[:70]
    
    topic_ids, precisions = zip(*selected_topics)

    plt.figure(figsize=(20, 10))
    bars = plt.bar(range(1, len(topic_ids) + 1), precisions, width=0.8)
    
    for i, (topic_id, precision) in enumerate(selected_topics):
        plt.text(i + 1, precision, topic_id, ha='center', va='bottom', fontsize=8, rotation=90)

    plt.xlabel('Topics (Ranked by Precision@5)')
    plt.ylabel('Precision@5')
    plt.title(f'Precision@5 Bar Plot for {topics_file} (Advanced BM25 - Selected Topics)')
    plt.ylim(0, 1)
    plt.xlim(0, len(topic_ids) + 1)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(f"bar_plot_precision_at_5_advanced_bm25_{os.path.splitext(os.path.basename(topics_file))[0]}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Number of topics selected: {num_topics}")
    print(f"Best performing topic: {selected_topics[0][0]} (P@5: {selected_topics[0][1]:.4f})")
    print(f"Worst performing topic: {selected_topics[-1][0]} (P@5: {selected_topics[-1][1]:.4f})")
    print(f"Median performing topic: {selected_topics[num_topics//2][0]} (P@5: {selected_topics[num_topics//2][1]:.4f})")

def main():
    parser = argparse.ArgumentParser(description='BM25F Retrieval System')
    parser.add_argument('--answers_file', default='Answers.json', help='Path to Answers.json')
    parser.add_argument('--topics_files', nargs='+', default=['topics_1.json', 'topics_2.json'], help='List of topics files')
    parser.add_argument('--qrels_files', nargs='+', default=['qrel_1.tsv', 'qrel_2.tsv'], help='List of qrels files')
    args = parser.parse_args()

    start_time = time.time()

    try:
        print(f"Loading answers from {args.answers_file}")
        retriever = BM25FRetriever(args.answers_file)
    except Exception as e:
        print(f"Error initializing BM25FRetriever: {str(e)}")
        print("Please check your Answers.json file and ensure it contains the required fields (Id, Text, Score).")
        return

    for topics_file, qrels_file in zip(args.topics_files, args.qrels_files):
        print(f"Processing {topics_file} with {qrels_file}...")
        
        topics = load_topics(topics_file)
        qrels = load_qrels(qrels_file)
        
        results = {}
        for topic_id, topic in topics.items():
            query = f"{topic['Title']} {topic['Body']}"
            results[topic_id] = retriever.search(query)

        output_file = f"result_advanced_bm25f_{os.path.splitext(os.path.basename(topics_file))[0][-1]}.tsv"
        write_results(results, output_file)
        print(f"Results for {topics_file} written to {output_file}")

        print(f"Evaluating results using {qrels_file}...")
        run_results = {}
        for topic_id, docs in results.items():
            run_results[topic_id] = {doc_id: score for doc_id, score in docs[:100]}

        qrels_filtered = {topic_id: qrels[topic_id] for topic_id in run_results if topic_id in qrels}
        if not qrels_filtered:
            print("No matching topics found in qrels for evaluation.")
            continue

        eval_results = evaluate_results(qrels_filtered, run_results)
        print("Evaluation Results:")
        for metric, value in eval_results.items():
            print(f"{metric}: {value:.4f}")

        plot_evaluation_results(eval_results, topics_file)
        print(f"Evaluation plot saved as evaluation_plot_advanced_bm25f_{os.path.splitext(os.path.basename(topics_file))[0]}.png")

        plot_precision_at_5_bar_plot(qrels_filtered, run_results, topics_file)
        print(f"Bar plot saved as bar_plot_precision_at_5_advanced_bm25f_{os.path.splitext(os.path.basename(topics_file))[0]}.png")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()