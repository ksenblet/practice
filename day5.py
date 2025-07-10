# python day5.py --results results.txt --output class.json
import argparse
import json
import os

import joblib
import numpy as np

TASK_RANGE = range(22, 29)
MODEL_FILE = "classification_model.joblib"


def parse_results(results_file):
    works = {}
    current_base_id = None

    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if 'Задание' not in line:
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


def predict_grades(results_file, output_file="classifier.json"):
    if not os.path.exists(MODEL_FILE):
        print(f"Ошибка: Модель {MODEL_FILE} не найдена")
        print("Сначала обучите модель")
        return

    model = joblib.load(MODEL_FILE)
    print(f"Модель оценки загружена из {MODEL_FILE}")

    results_data = parse_results(results_file)

    predicted_grades = {}

    for work_id, tasks in results_data.items():
        work_grades = {}

        for task_num in TASK_RANGE:
            task_key = str(task_num)
            if task_key in tasks:
                text = tasks[task_key]
                try:
                    score = model.predict([text])[0]
                    work_grades[task_key] = score
                except Exception as e:
                    print(f"Ошибка при оценке задания {task_key} работы {work_id}: {str(e)}")
                    work_grades[task_key] = f"Ошибка оценки: {str(e)}"
            else:
                work_grades[task_key] = "Текст задания не найден"

            # Преобразуем все numpy int32 в обычные int
            for task_key, score in work_grades.items():
                if isinstance(score, np.integer):
                    work_grades[task_key] = int(score)

            predicted_grades[work_id] = work_grades

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predicted_grades, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты оценки сохранены в {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Файл с распознанными текстами')
    parser.add_argument('--output', default='classifier.json', help='Файл для сохранения оценок')

    args = parser.parse_args()
    predict_grades(args.results, args.output)
