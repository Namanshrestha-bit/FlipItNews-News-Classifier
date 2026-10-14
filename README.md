# FlipItNews News Classification

FlipItNews is an end-to-end news categorisation project. It explores the BBC News topic dataset, cleans and tokenises article text, compares neural models using Bag of Words, TF-IDF, and Word2Vec features, and deploys the selected recurrent neural network through an interactive Streamlit dashboard.

## What is included

- `FlipItNews_News_Classification_Ready.ipynb` — reproducible EDA, preprocessing, feature extraction, model training, comparison, error analysis, and export.
- `rnn_model.keras` — saved BiLSTM model selected for deployment.
- `model_artifacts/` — vocabulary, labels, stop words, sample stories, benchmark metrics, and the confusion matrix used by the dashboard.
- `build_artifacts.py` — rebuilds the deployment bundle from the BBC-format dataset and validates the saved model.
- `app.py` — Streamlit dashboard for single-story and batch classification.

## EDA findings

The dataset contains 2,225 articles across five categories: sport (511), business (510), politics (417), tech (401), and entertainment (386). There are no missing values. The text contains 133 duplicate article bodies; after removing those duplicates, 2,092 articles remain.

Preprocessing converts text to lowercase, removes punctuation, digits, and symbols, tokenises on whitespace, removes 198 English stop words, and drops one-character tokens. The deployed sequence model uses a 15,063-token vocabulary and a 353-token input window.

## Model comparison

| Model | Features | Test accuracy | Test F1 |
| --- | --- | ---: | ---: |
| NN + BoW | Bag of Words | 97.37% | 97.39% |
| NN + TF-IDF | TF-IDF | 97.61% | 97.62% |
| NN + Word2Vec | Average Word2Vec document vector | 96.42% | 96.44% |
| RNN (BiLSTM) + Word2Vec | Word2Vec sequence embeddings | 95.70% | 95.70% |
| RNN (BiGRU) + learned embeddings | Learned sequence embeddings | 94.51% | 94.50% |

The dashboard deploys the BiLSTM because the requirement is to host an RNN and it is the strongest RNN in the comparison. On the held-out set, it reaches 95.70% accuracy and 95.70% weighted F1. The weakest category is business at 94.85% F1, which is a useful target for future data collection and model improvement.

## Run locally

Use Python 3.12 for the TensorFlow environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If the deployment metadata needs to be regenerated after changing the dataset:

```powershell
python -m pip install -r requirements-training.txt
python build_artifacts.py
streamlit run app.py
```

To rerun the full experiment, open the notebook after installing `requirements-training.txt`. The notebook can use `bbc-news-data.tsv` or the included `bbc-news-data.csv` mirror; the dataset is read as tab separated because that is the canonical BBC file format.

## Deployment and FlipItNews impact

The dashboard supports:

1. Single-article classification with per-category probabilities.
2. Confidence-aware routing: predictions below the 70% threshold are flagged for editorial review.
3. Batch CSV/TSV classification with downloadable predictions.
4. Model comparison, category distribution, per-class metrics, and a confusion matrix.

Automated tags can reduce manual filing, improve search and archive facets, and provide category signals for follows, newsletters, and personalised feeds. The confidence threshold keeps the system human-in-the-loop: clear predictions can be auto-tagged while ambiguous stories remain visible to editors. In the benchmark simulation, roughly 96.9% of test stories cleared the 70% threshold, while the remaining 3.1% were routed for review.

The model should be monitored for topic drift, publisher/source changes, and new categories before it is used as the sole publishing decision.
