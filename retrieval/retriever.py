from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
 
def load_embedder():
    return SentenceTransformer('all-mpnet-base-v2')

def embed_documents(embedder, doc_df):
    return embedder.encode(doc_df['text'].tolist(), show_progress_bar=True)

def filter_candidates(doc_df, topic, care_setting, population):
    candidates = doc_df[
        (doc_df['topic'] == topic) &
        (doc_df['care_setting'] == care_setting) &
        (doc_df['population'] == population)
    ]
    if len(candidates) == 0:
        candidates = doc_df[doc_df['topic'] == topic]
    if len(candidates) == 0:
        candidates = doc_df
    return candidates

def retrieve_document(embedder, doc_embeddings, doc_df, question, topic, care_setting, population):
    candidates = filter_candidates(doc_df, topic, care_setting, population)
    candidate_indices = candidates.index.tolist()
    candidate_embeddings = doc_embeddings[candidate_indices]
    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(question_embedding, candidate_embeddings)[0]
    best_local_index = np.argmax(similarities)
    best_global_index = candidate_indices[best_local_index]
    return doc_df.loc[best_global_index, 'document_id'], similarities[best_local_index]
