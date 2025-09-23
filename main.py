"""
auto_personas.py
Автоматическое выделение целевой аудитории + извлечение болевых точек и генерация персон.
Ожидаемые колонки в input CSV:
  user_id, age, country, job, monthly_budget, visits_per_month, pain_text
"""

import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import os
import shutil

# ML & NLP
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

# NLP utilities
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Visualization
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------------------
# 1) Загрузка и простая валидация
# ---------------------------
def load_data(path_csv):
    df = pd.read_csv(path_csv)
    # минимальная валидация
    required = {'user_id','login','location','public_repos','followers','bio'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"В CSV не хватает колонок: {missing}")
    df = df.dropna(subset=['user_id'])  # drop rows without id
    return df

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r'http\S+',' ', s)
    s = re.sub(r'[^a-zа-яё0-9\s]', ' ', s)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(s.lower())
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ---------------------------
# 3) Векторизация и извлечение тем (NMF)
# ---------------------------
def extract_topics(texts, n_topics=6, max_features=2000):
    """Возвращает модель NMF, vectorizer и топ-слова для каждой темы"""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, comp in enumerate(H):
        terms_idx = np.argsort(comp)[-12:][::-1]
        terms = [feature_names[i] for i in terms_idx]
        topics.append(terms)
    return {
        'nmf': nmf,
        'vectorizer': vectorizer,
        'topic_terms': topics,
        'topic_matrix': W
    }

# ---------------------------
# 4) Кластеризация пользователей по числовым признакам
# ---------------------------
def cluster_users(df, num_clusters=3, features=['public_repos','followers']):
    X = df[features].fillna(0).astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(Xs)
    score = silhouette_score(Xs, labels) if len(set(labels)) > 1 else None
    return {
        'kmeans': kmeans,
        'labels': labels,
        'scaler': scaler,
        'silhouette': score
    }

# ---------------------------
# 5) Генерация описаний персон на основе кластера + тем
# ---------------------------
def synthesize_personas(df, cluster_labels, topic_matrix, topic_terms, top_n_topics=2):
    df = df.copy()
    df['cluster'] = cluster_labels
    personas = []
    for c in sorted(df['cluster'].unique()):
        subset = df[df['cluster']==c]
        count = len(subset)
        countries = subset['location'].value_counts().head(3).to_dict()
        avg_repos = float(subset['public_repos'].dropna().mean()) if count>0 else None
        avg_followers = float(subset['followers'].dropna().mean()) if count>0 else None

        # aggregate topic scores for cluster (topic_matrix rows align with rows in df in same order)
        cluster_indices = subset.index.tolist()
        if len(cluster_indices) == 0:
            top_topics = []
        else:
            cluster_topic_scores = topic_matrix[cluster_indices].mean(axis=0)
            top_topic_idxs = list(np.argsort(cluster_topic_scores)[-top_n_topics:][::-1])
            top_topics = [topic_terms[i] for i in top_topic_idxs]

        persona = {
            'cluster_id': int(c),
            'size': int(count),
            'top_countries': countries,
            'avg_public_repos': avg_repos,
            'avg_followers': avg_followers,
            'top_topics_terms': top_topics
        }

        # generate natural language summary
        persona['summary'] = generate_persona_text(persona)
        personas.append(persona)
    return personas

def generate_persona_text(p):
    lines = []
    lines.append(f"Кластер #{p['cluster_id']} — ~{p['size']} пользователей.")
    if p['top_countries']:
        countries = ", ".join([f"{k} ({v})" for k,v in p['top_countries'].items()])
        lines.append(f"Основные страны: {countries}.")
    if p['avg_public_repos'] is not None:
        lines.append(f"Сколько публичных репозиториев: {p['avg_public_repos']:.0f}.")
    if p['avg_followers'] is not None:
        lines.append(f"Средний уровень подписок: {p['avg_followers']:.0f}.")
    if p['top_topics_terms']:
        topic_snippets = ["; ".join(t[:6]) for t in p['top_topics_terms']]
        lines.append("Основные болевые точки / интересы (темы): " + " | ".join(topic_snippets))
    return " ".join(lines)

# ---------------------------
# 6) Визуализация кластеров (2D проекция)
# ---------------------------
from sklearn.decomposition import PCA

def plot_clusters(df, labels, features=['public_repos','followers'], out_path='clusters.png'):
    X = df[features].fillna(0).astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    plt.figure(figsize=(8,6))
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps['tab10'].resampled(len(unique_labels))
    for i,l in enumerate(unique_labels):
        mask = np.array(labels)==l
        plt.scatter(coords[mask,0], coords[mask,1], label=f'cluster {l}', alpha=0.7)
    plt.legend()
    plt.title('Кластеры пользователей (PCA 2D)')
    plt.xlabel('PC1'), plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    return out_path

# ---------------------------
# 7) Основной pipeline
# ---------------------------
def build_personas_pipeline(csv_path, out_json='personas.json', num_clusters=3, n_topics=6):
    df = load_data(csv_path)
    # clean text
    df['clean_pain'] = df['bio'].apply(clean_text)
    # topic extraction
    topics_res = extract_topics(df['clean_pain'].astype(str).tolist(), n_topics=n_topics)
    # clustering
    cluster_res = cluster_users(df, num_clusters=num_clusters)
    labels = cluster_res['labels']
    # synthesize personas
    personas = synthesize_personas(df, labels, topics_res['topic_matrix'], topics_res['topic_terms'], top_n_topics=2)
    # save
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'personas': personas,
            'silhouette': cluster_res['silhouette'],
            'topic_terms': topics_res['topic_terms']
        }, f, ensure_ascii=False, indent=2)
    # plot
    plot_path = plot_clusters(df, labels)
    return out_json, plot_path

# ---------------------------
# 8) Пример вызова при запуске
# ---------------------------
if __name__ == '__main__':
    import argparse

    # Папка для локальных данных NLTK в виртуальном окружении
    nltk_data_dir = os.path.join(os.path.dirname(__file__), ".venv", "nltk_data")
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    # Скачиваем только стандартные пакеты
    for pkg in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg, download_dir=nltk_data_dir)

    # Форсируем использование обычного Punkt вместо punkt_tab
    from nltk.tokenize import punkt

    punkt.PunktLanguageVars = punkt.PunktLanguageVars

    stop_words = set(stopwords.words('english')) | set(stopwords.words('russian'))  # bilingual
    lemmatizer = WordNetLemmatizer()

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=False, default='github_users.csv', help='Путь к CSV')
    parser.add_argument('--clusters', type=int, default=3)
    parser.add_argument('--topics', type=int, default=49)
    args = parser.parse_args()
    out_json, plot_path = build_personas_pipeline(args.csv, num_clusters=args.clusters, n_topics=args.topics)
    print("Personas saved to:", out_json)
    print("Cluster plot saved to:", plot_path)
