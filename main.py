import requests
import datetime
import re
import sqlite3
import os

DB_NAME = "github_issues.db"
RTF_FILE = "github_report.rtf"
GITHUB_TOKEN = ""  # если есть токен, вставьте для увеличения лимитов API

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


# ================== DATABASE ==================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
                CREATE TABLE IF NOT EXISTS issues
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    repo
                    TEXT,
                    title
                    TEXT,
                    link
                    TEXT,
                    creation_date
                    TEXT,
                    state
                    TEXT,
                    pains
                    TEXT
                )
                """)
    conn.commit()
    conn.close()


def save_to_db(issues):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    for i in issues:
        cur.execute("""
                    INSERT INTO issues (repo, title, link, creation_date, state, pains)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (i["repo"], i["title"], i["link"], str(i["creation_date"]), i["state"], ", ".join(i["pains"])))
    conn.commit()
    conn.close()


# ================== GITHUB SEARCH ==================
def search_repositories(query="python", per_page=5):
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page
    }
    response = requests.get(url, params=params, headers=HEADERS)
    data = response.json()
    repos = []
    for item in data.get("items", []):
        repos.append(item["full_name"])  # owner/repo
    return repos


def get_github_issues(repo, state="open", per_page=10):
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": state, "per_page": per_page}
    response = requests.get(url, params=params, headers=HEADERS)
    data = response.json()
    issues = []
    for item in data:
        if "pull_request" in item:
            continue
        issues.append({
            "repo": repo,
            "title": item["title"],
            "link": item["html_url"],
            "creation_date": datetime.datetime.strptime(item["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
            "state": item["state"]
        })
    return issues


# ================== ANALYSIS ==================
def extract_pain_points(text):
    markers = {
        "error": "Ошибка / баг",
        "fail": "Сбой",
        "slow": "Медленно",
        "performance": "Проблемы с производительностью",
        "crash": "Краш приложения",
        "memory": "Проблемы с памятью",
        "feature request": "Запрос новой функции",
    }
    text_lower = text.lower()
    pains = [v for k, v in markers.items() if re.search(rf"\b{k}\b", text_lower)]
    return pains


# ================== EXPORT RTF ==================
def export_to_rtf():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT repo, title, link, creation_date, state, pains FROM issues ORDER BY id DESC LIMIT 50")
    rows = cur.fetchall()
    conn.close()

    file_exists = os.path.exists(RTF_FILE)

    if not file_exists:
        rtf_content = "{\\rtf1\\ansi\\deff0\n"
        rtf_content += "{\\b GitHub Issues Report}\\par\n\n"
    else:
        with open(RTF_FILE, "r", encoding="utf-8") as f:
            rtf_content = f.read()
        if rtf_content.endswith("}"):
            rtf_content = rtf_content[:-1]

    for r in rows:
        rtf_content += f"\\b Repo:\\b0 {r[0]}\\par\n"
        rtf_content += f"\\b Title:\\b0 {r[1]}\\par\n"
        rtf_content += f"Link: {r[2]}\\par\n"
        rtf_content += f"Date: {r[3]} | State: {r[4]}\\par\n"
        rtf_content += f"Pains: {r[5]}\\par\n"
        rtf_content += "\\par\n"

    rtf_content += "}"
    with open(RTF_FILE, "w", encoding="utf-8") as f:
        f.write(rtf_content)

    print(f"✅ Данные добавлены в {RTF_FILE}")


# ================== MAIN ==================
if __name__ == "__main__":
    init_db()
    # Шаг 1: Автоматический поиск репозиториев по тегу python
    repos = search_repositories(query="python", per_page=5)
    print("Найдены репозитории:", repos)

    # Шаг 2: Сбор issues и анализ болей
    all_issues = []
    for repo in repos:
        issues = get_github_issues(repo, per_page=10)
        for i in issues:
            i["pains"] = extract_pain_points(i["title"])
        all_issues.extend(issues)

    save_to_db(all_issues)
    export_to_rtf()
