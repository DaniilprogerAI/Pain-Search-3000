import json
import csv

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def client_score(profile_match1, budget1, deadline1, rating1, problem_match1,
                 wP=0.3, wB=0.2, wD=0.1, wR=0.2, wS=0.2):
    # Все значения должны быть в диапазоне [0,1]
    return wP*profile_match1 + wB*budget1 + wD*deadline1 + wR*rating1 + wS*problem_match1

def create_profile_match():
    with open('config.json', 'r', encoding='utf-8') as f:
        my_data = json.load(f)
    my_skills = my_data['skills']
    client_skills = []
    while True:
        print("Введите 'выход' для выхода")
        skill = input("Введите навыки, которые нужны клиенту: ")
        if skill == 'выход':
            break
        client_skills.append(skill.capitalize())

    # Преобразуем списки в множества
    my_skills_set = set(my_skills)
    client_skills_set = set(client_skills)

    matching_skills = my_skills_set.intersection(client_skills_set)
    score_skills = len(matching_skills) / len(client_skills_set)

    print(f"Совпадение навыков: {score_skills:.2f}")
    return score_skills, client_skills

def budget_score(hourly_rate, hours_needed, client_budget):
    ideal_budget = hourly_rate * hours_needed
    score = min(client_budget / ideal_budget, 1)
    return max(score, 0)


def can_meet_deadline(available_time, required_time, client_deadline):
    # U - успеваемость по собственному времени
    U = available_time / required_time
    # Ud - успеваемость по дедлайну клиента
    Ud = client_deadline / required_time

    return {
        "score_personal": U,
        "score_deadline": Ud,
        "can_do": Ud >= 1
    }

def reliability_score(rating, min_rating=1, max_rating=5):
    return (rating - min_rating) / (max_rating - min_rating)

def problem_match(project_words, my_skills):
    return len(my_skills & project_words) / len(my_skills)

def target_audience():
    with open('config.json', 'r', encoding='utf-8') as f:
        my_data = json.load(f)

    profile_match, skills = create_profile_match()

    # Пример
    hourly_rate1 = int(input("Введите вашу почасовую ставку: "))  # твоя ставка $/час
    hours_needed1 = int(input("Введите сколько займёт часов проект: "))  # время проекта
    client_budget1 = int(input("Введите бюджет клиента: "))  # бюджет клиента

    budget = budget_score(hourly_rate1, hours_needed1, client_budget1)
    print(f"Соответствие бюджета: {budget:.2f}")  # 0.83 → чуть меньше идеала

    # Пример данных
    available_time = int(input("Введите сколько часов в неделю вы выделяете на задачу: "))  # часов в неделю
    client_deadline = int(input("Введите часы до дедлайна: "))  # часов до дедлайна

    deadline = can_meet_deadline(available_time, hours_needed1, client_deadline)
    print(deadline)

    rating_score = float(input("Введите сколько звёзд у клиента: "))
    rating = reliability_score(rating_score)
    print(f"Надёжность клиента: {rating:.2f}")  # Вывод: 0.65

    title = input("Введите заголовок задачи: ")
    S_c = set(title.lower().split())
    problem_match_var = 0
    for service in my_data['services']:
        S_s = set(service.lower().split())

        problem_match_var += problem_match(S_c, S_s)
    print(f"Problem match: {problem_match_var:.2f}")

    normalize_profile = normalize(profile_match, 0, 1)
    normalize_budget = normalize(budget, 0, 1)
    normalize_deadline = normalize(deadline["score_deadline"], 0, 1)
    normalize_rating = normalize(rating, 0, 1)
    normalize_problem = normalize(problem_match_var, 0, 1)

    score = client_score(
        normalize_profile,
        normalize_budget,
        normalize_deadline,
        normalize_rating,
        normalize_problem
    )
    print(f"Идеальность клиента: {score:.2f}")

    if score > 0.75:
        # Данные клиента
        client_data = {
            "required skills": skills,
            "budget": client_budget1,
            "deadline": client_deadline,
            "rating": rating_score,
            "title": title
        }

        # Имя файла
        filename = "clients.csv"

        # Проверяем, есть ли файл, если нет - добавляем заголовки
        try:
            with open(filename, "x", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=client_data.keys())
                writer.writeheader()
        except FileExistsError:
            pass  # Файл уже есть, заголовки писать не нужно

        # Добавляем данные клиента
        with open(filename, "a", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=client_data.keys())
            writer.writerow(client_data)

        print("Данные клиента сохранены!")

if __name__ == '__main__':
    target_audience()
