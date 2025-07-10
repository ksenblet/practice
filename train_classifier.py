# python train_classifier.py --results results.txt --grades classJS
import os
import json
import joblib
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASK_RANGE = range(22, 29)
MODEL_FILE = "classification_model.joblib"


def load_grades(grades_dir):
    grades = {}
    for filename in os.listdir(grades_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(grades_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            work_id = data['id']
            mask = data['mask']
            scores = []

            parts = mask.split(')')[:-1]
            for part in parts:
                score = part.split('(')[0]
                scores.append(int(score))

            grades[work_id] = scores

    return grades


def parse_results(results_file):
    works = {}
    current_work = None
    current_base_id = None

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if 'Задание' not in line:
                current_work = line
                current_base_id = line.rsplit('_', 1)[0]

                if current_base_id not in works:
                    works[current_base_id] = {}
            else:
                if not current_base_id:
                    continue

                parts = line.split(':', 1)
                if len(parts) < 2:
                    continue

                task_num = parts[0].replace('Задание', '').strip()
                text = parts[1].strip().strip('"')

                if task_num.isdigit() and int(task_num) in TASK_RANGE:
                    if task_num in works[current_base_id]:
                        works[current_base_id][task_num] += " " + text
                    else:
                        works[current_base_id][task_num] = text

    return works


def prepare_data(results_data, grades_data):
    text = []  # Тексты заданий
    gr = []  # Оценки

    for work_id, tasks in results_data.items():
        if work_id not in grades_data:
            continue

        scores = grades_data[work_id]

        for i, task_num in enumerate(TASK_RANGE):
            task_key = str(task_num)
            if task_key in tasks:
                text.append(tasks[task_key])
                gr.append(scores[i])

    return text, gr


def train_model(text, gr):
    text_train, text_test, gr_train, gr_test = train_test_split(
        text, gr, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42
        ))
    ])

    model.fit(text_train, gr_train)

    gr_pred = model.predict(text_test)
    print("\nОтчет о классификации:")
    print(classification_report(gr_test, gr_pred))

    return model


def main(results_file, grades_dir):
    grades_data = load_grades(grades_dir)
    results_data = parse_results(results_file)

    text, gr = prepare_data(results_data, grades_data)

    if not text:
        print("Ошибка: Нет данных для обучения!")
        return

    print(f"Найдено {len(text)} примеров для обучения")

    print("Обучение модели оценки...")
    model = train_model(text, gr)

    joblib.dump(model, MODEL_FILE)
    print(f"\nМодель сохранена в {MODEL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели для оценки работ.')
    parser.add_argument('--results', type=str, required=True, help='Файл с результатами распознавания')
    parser.add_argument('--grades', type=str, required=True, help='Папка с JSON-файлами оценок')

    args = parser.parse_args()

    main(
        results_file=args.results,
        grades_dir=args.grades
    )