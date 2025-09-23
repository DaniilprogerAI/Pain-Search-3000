import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
nltk.download("punkt_tab")

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('russian'))

# ---------------------------
# 1) Сбор текста с сайтов конкурентов
# ---------------------------
def get_site_text(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        # удаляем скрипты и стили
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text)  # убираем лишние пробелы
        return text
    except Exception as e:
        print(f"Ошибка при получении {url}: {e}")
        return ""

# ---------------------------
# 2) Очистка текста и токенизация
# ---------------------------
def clean_and_tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# ---------------------------
# 3) Извлечение ключевых слов и фраз пакета
# ---------------------------
def extract_features(text):
    # простая эвристика: ищем ключевые слова "включает, пакет, услуги, feature, offer"
    sentences = sent_tokenize(text)
    package_sentences = [s for s in sentences if re.search(r'(include|пакет|offer|feature|services|содержит)', s)]
    weaknesses_sentences = [s for s in sentences if re.search(r'(но|ограничен|только|не включает|except|limited)', s)]
    return package_sentences, weaknesses_sentences

# ---------------------------
# 4) Анализ нескольких конкурентов
# ---------------------------
competitor_urls = [
    "https://www.imaginarycloud.com/services/python-development-services",
    "https://www.intecfy.com/hire-freelance-python-developers/",
    "https://applicants.bairesdev.com/job/71/279637/apply?utm_source=linkedinjobposting&utm_medium=jobposting&utm_campaign=Remote-20250813&urlHash=PS0d&lang=en",
    "https://www.caktusgroup.com/",
    "https://startups.epam.com/services/python-consulting?utm_source=chatgpt.com",
    "https://dashdevs.com/python-development-services/?utm_source=chatgpt.com",
    "https://www.iflexion.com/developers/python",
    "https://uvik.net/services/python-consulting/?utm_source=chatgpt.com",
    "https://www.netguru.com/services/python-consulting?utm_source=chatgpt.com",
    "https://www.cloverdynamics.com/expertise/technologies/python-consulting?utm_source=chatgpt.com"
]

results = []

for url in competitor_urls:
    text = get_site_text(url)
    package_sentences, weaknesses_sentences = extract_features(text)
    results.append({
        "url": url,
        "package_features": package_sentences,
        "weaknesses": weaknesses_sentences
    })

# ---------------------------
# 5) Сохраняем результаты
# ---------------------------
df = pd.DataFrame(results)
df.to_csv("competitors_analysis.csv", index=False, encoding='utf-8')
print("Анализ конкурентов сохранен в competitors_analysis.csv")