import requests
import datetime
import re


def get_stackoverflow_questions(tag="django", pagesize=20):
    """
    Получаем последние вопросы с Stack Overflow по тегу
    """
    url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "order": "desc",
        "sort": "creation",  # сортируем по дате создания
        "tagged": tag,  # фильтр по тегу
        "site": "stackoverflow",
        "pagesize": pagesize,
        "filter": "withbody"  # чтобы получить текст вопроса (body)
    }

    response = requests.get(url, params=params)
    data = response.json()

    questions = []
    for q in data.get("items", []):
        questions.append({
            "title": q["title"],
            "link": q["link"],
            "creation_date": datetime.datetime.fromtimestamp(q["creation_date"]),
            "is_answered": q["is_answered"],
            "body": q.get("body", "")
        })

    return questions


def extract_pain_points(text):
    """
    Простая эвристика для поиска 'болей' в тексте вопроса
    """
    pains = []
    markers = {
        "error": "Ошибка / баг",
        "not working": "Не работает",
        "fail": "Сбой",
        "slow": "Медленно",
        "deploy": "Проблемы с деплоем",
        "urgent": "Срочная задача",
        "crash": "Краш приложения",
    }

    text_lower = text.lower()
    for key, value in markers.items():
        if re.search(rf"\b{key}\b", text_lower):
            pains.append(value)
    return pains


if __name__ == "__main__":
    tag_pain = input()
    results = get_stackoverflow_questions(tag_pain, 10)
    for r in results:
        pains = extract_pain_points(r["title"] + " " + r["body"])
        print(f"📌 {r['title']}")
        print(f"🔗 {r['link']}")
        print(f"🕒 {r['creation_date']}, Ответ есть: {r['is_answered']}")
        if pains:
            print(f"❗ Обнаруженные боли: {', '.join(pains)}")
        else:
            print("✅ Боли не найдены")
        print("-" * 80)
