import requests
import datetime
import re
import sqlite3
import pypandoc

DB_NAME = "stackoverflow.db"
RTF_FILE = "stackoverflow_report.rtf"

def init_db():
    """Создаём таблицу, если её нет"""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        link TEXT,
        creation_date TEXT,
        is_answered INTEGER,
        pains TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_stackoverflow_questions(tag="django", pagesize=10):
    url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "order": "desc",
        "sort": "creation",
        "tagged": tag,
        "site": "stackoverflow",
        "pagesize": pagesize,
        "filter": "withbody"
    }
    response = requests.get(url, params=params)
    data = response.json()
    questions = []
    for q in data.get("items", []):
        questions.append({
            "title": q["title"],
            "link": q["link"],
            "creation_date": datetime.datetime.fromtimestamp(q["creation_date"]),
            "is_answered": int(q["is_answered"])
        })
    return questions

def extract_pain_points(text):
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
    pains = [value for key, value in markers.items() if re.search(rf"\b{key}\b", text_lower)]
    return pains

def save_to_db(questions):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    for q in questions:
        cur.execute("""
        INSERT INTO questions (title, link, creation_date, is_answered, pains)
        VALUES (?, ?, ?, ?, ?)
        """, (q["title"], q["link"], str(q["creation_date"]), q["is_answered"], ", ".join(q["pains"])))
    conn.commit()
    conn.close()

def export_to_rtf():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT title, link, creation_date, is_answered, pains FROM questions ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()

    # Формируем Markdown-документ
    md_content = "# Отчёт Stack Overflow\n\n"
    for r in rows:
        md_content += f"## {r[0]}\n"
        md_content += f"- 🔗 {r[1]}\n"
        md_content += f"- 🕒 {r[2]}\n"
        md_content += f"- Ответ есть: {'Да' if r[3] else 'Нет'}\n"
        md_content += f"- ❗ Боли: {r[4] if r[4] else 'Не обнаружено'}\n\n"

    # Конвертируем в RTF
    pypandoc.convert_text(md_content, 'rtf', format='md', outputfile=RTF_FILE, extra_args=['--standalone'])
    print(f"✅ Отчёт сохранён в {RTF_FILE}")

if __name__ == "__main__":
    init_db()
    tag = "python"  # можно менять на react, api, python
    results = get_stackoverflow_questions(tag, 10)

    # добавляем анализ болей
    for r in results:
        r["pains"] = extract_pain_points(r["title"])

    save_to_db(results)
    export_to_rtf()

