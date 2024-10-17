# BM25 and TF-IDF Retrieval Systems

This project implements a TF-IDF and BM25 retrieval systems on a Ubuntu dataset using pyterrier

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Usage

Install the required packages:
 ```
pip install -r requirements.txt
 ```

Download the necessary NLTK data:
```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

To run the TF-IDF retrieval system:

```
python tfidf.py
```

To run the BM25 retrieval system:

```
python BM25.py
```

You can specify custom file paths using the following arguments:

- `--answers_file`: Path to the answers JSON file (default: 'answers.json')
- `--topics_files`: List of topics JSON files (default: ['topics_1.json', 'topics_2.json'])
- `--qrels_file`: Path to the qrels TSV file (default: 'qrel_1.tsv', 'qrel_2.tsv')

## Output

The script generates the following outputs:

1. Result files: `result_bm25_1.tsv` and `result_bm25_2.tsv`
2. Evaluation plots: `evaluation_plot_bm25_1.png`, `evaluation_plot_bm25_2.png` `evaluation_plot_tf-idf_1.png`, and `evaluation_plot_tf-idf_2.png`
3. Precision@5 bar plots

## Assumptions

1. The input files (answers, topics, and qrels) are in the correct format, encoding (UTF-8), and are in the same directory of the scripts.
2. The system has sufficient memory to load all answers and build the inverted index.
3. The evaluation metrics (P@1, P@5, nDCG@5, MRR, MAP) are calculated using the ranx library.
4. You have the helper module clean_inputs.py in the same directory.

## Note

This implementation is a simple version of the BM25 algorithm.
