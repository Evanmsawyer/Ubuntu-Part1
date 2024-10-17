import os
import pyterrier as pt
if not pt.started():
    pt.init()

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

from clean_inputs import clean_text

class InteractiveSearcher:
    def __init__(self, index_path: str, answers_file: str):
        self.index_path = index_path
        self.answers_file = answers_file
        self.indexref = pt.IndexRef.of(index_path)
        self.bm25 = pt.BatchRetrieve(self.indexref, wmodel="BM25")
        self.qe = pt.rewrite.Bo1QueryExpansion(self.indexref)
        self.bo1_pipeline = self.bm25 >> self.qe >> self.bm25
        self.word2vec_model = self.load_word2vec()
        self.answers = self.load_answers()

    def load_word2vec(self):
        model_path = os.path.join(os.path.dirname(self.index_path), "word2vec.model")
        if os.path.exists(model_path):
            return Word2Vec.load(model_path)
        else:
            print("Word2Vec model not found. Query expansion will be limited.")
            return None

    def load_answers(self):
        return pd.read_json(self.answers_file)

    def expand_query(self, query: str, topn: int = 2) -> str:
        tokens = word_tokenize(query)
        expanded_terms = tokens.copy()
        if self.word2vec_model:
            for token in tokens[:5]:
                try:
                    similar_words = self.word2vec_model.wv.most_similar(token, topn=topn)
                    expanded_terms.extend([word for word, _ in similar_words])
                except KeyError:
                    continue  
        return ' '.join(list(set(expanded_terms)))[:1000]  

    def search(self, query: str, k: int = 10):
        cleaned_query = clean_text(query)
        expanded_query = self.expand_query(cleaned_query)
        results = self.bo1_pipeline.search(expanded_query)
        
        for _, row in results.iterrows()[:k]:
            doc_id = row['docno']
            score = row['score']
            answer = self.answers[self.answers['Id'] == doc_id].iloc[0]
            print(f"Document ID: {doc_id}")
            print(f"Score: {score}")
            print(f"Title: {answer['Title']}")
            print(f"Text: {answer['Text'][:200]}...")  
            print("-" * 50)

def main():
    home_dir = os.path.expanduser("~")
    index_path = os.path.join(home_dir, "askubuntu_index")
    answers_file = "cleaned_answers.json"  

    if not os.path.exists(index_path):
        print(f"Index not found at {index_path}. Please run the main BM25 script first.")
        return

    searcher = InteractiveSearcher(index_path, answers_file)

    print("Welcome to the AskUbuntu Interactive Search")
    print("Enter your query or type 'quit' to exit")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'quit':
            break
        searcher.search(query)

if __name__ == "__main__":
    main()