import json
import os
from symspellpy import SymSpell, Verbosity
from ultralytics import YOLO
from tqdm import tqdm
# import day5 для полного пайплайна

webres_dir = "school"
images_dir = "photoDay4"
output_file = "results.txt"
grades_dir = "classJS"


sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
sym_spell.load_dictionary('ru_full.txt', term_index=0, count_index=1, encoding='utf-8')
model = YOLO('best.pt')


def correct_text(text):
    suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else text


def extract_text_from_webres(webres_path):
    with open(webres_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_boxes = []

    def parse_element(element):
        if isinstance(element, dict):
            if 'languages' in element:
                for lang in element['languages']:
                    if lang.get('lang') == 'rus':
                        for text_item in lang.get('texts', []):
                            text = text_item.get('text', '').strip()
                            if text:
                                corrected = correct_text(text)
                                box = {
                                    'x': element.get('x', 0),
                                    'y': element.get('y', 0),
                                    'w': element.get('w', 0),
                                    'h': element.get('h', 0),
                                    'text': corrected
                                }
                                text_boxes.append(box)
            for value in element.values():
                parse_element(value)
        elif isinstance(element, list):
            for item in element:
                parse_element(item)

    parse_element(data)

    return text_boxes


def get_yolo_boxes(image_path):
    results = model(image_path)
    yolo_boxes = []
    image_name = os.path.basename(image_path)
    print(f"\nРезультаты обработки для {image_name}:")

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls)
            conf = box.conf.item()
            print(
                f"  Класс: {cls_id}, Метка: {model.names[cls_id]}, Conf: {conf:.2f}, BBox: {xyxy}")

            yolo_boxes.append({
                'x1': xyxy[0],
                'y1': xyxy[1],
                'x2': xyxy[2],
                'y2': xyxy[3],
                'label': cls_id
            })
    print(f"\nРезультат обработки {image_path} записан в файл")
    return yolo_boxes


def match_text_to_tasks(webres_boxes, yolo_boxes):
    task_texts = {}

    for yolo_box in yolo_boxes:
        task_num = yolo_box['label']
        matched_texts = []

        for webres_box in webres_boxes:
            # находится ли центр текста внутри YOLO-бокса
            text_center_x = webres_box['x'] + webres_box['w'] / 2
            text_center_y = webres_box['y'] + webres_box['h'] / 2

            if (yolo_box['x1'] <= text_center_x <= yolo_box['x2'] and
                    yolo_box['y1'] <= text_center_y <= yolo_box['y2']):
                matched_texts.append(webres_box['text'])

        if matched_texts:
            task_label = model.names[task_num].split()[-1]
            task_texts[task_label] = " ".join(matched_texts)

    return task_texts


def process_pair(webres_path, image_path):
    webres_boxes = extract_text_from_webres(webres_path)
    yolo_boxes = get_yolo_boxes(image_path)
    return match_text_to_tasks(webres_boxes, yolo_boxes)


def find_pairs(webres_dir, images_dir):
    pairs = []
    webres_files = [f for f in os.listdir(webres_dir) if f.endswith('.webRes')]

    for webres_file in webres_files:
        base_name = webres_file.split('__')[0]
        image_file = f"{base_name}.png"

        if os.path.exists(os.path.join(images_dir, image_file)):
            pairs.append((
                os.path.join(webres_dir, webres_file),
                os.path.join(images_dir, image_file)
            ))

    return pairs


def main(webres_dir, images_dir, output_file):
    print('===== НАЧАЛО ОБРАБОТКИ =====')
    pairs = find_pairs(webres_dir, images_dir)
    grouped_results = {}

    if not pairs:
        print("Не найдено пар для обработки")
        return


    # для прогресс-бара
    with tqdm(total=len(pairs), desc="Обработка файлов", unit="pair") as pbar:
        for webres_path, image_path in pairs:
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]

            pbar.set_description(f"Обработка: {image_name}")

            try:
                task_texts = process_pair(webres_path, image_path)

                if base_name not in grouped_results:
                    grouped_results[base_name] = {}

                for task_label, text in task_texts.items():
                    if task_label in grouped_results[base_name]:
                        grouped_results[base_name][task_label] += " " + text
                    else:
                        grouped_results[base_name][task_label] = text

                tasks_found = list(task_texts.keys())
                pbar.write(f"Сопоставлено: {image_name} | Задания: {', '.join(tasks_found)}")

            except Exception as e:
                pbar.write(f"Ошибка: {image_name} | {str(e)}")

            pbar.update(1)

    with open(output_file, 'w', encoding='utf-8') as f:
        for base_name in sorted(grouped_results.keys()):
            f.write(f"{base_name}\n")

            tasks = grouped_results[base_name]
            for task_label in sorted(tasks.keys(), key=lambda x: int(x)):
                f.write(f"Задание {task_label}: \"{tasks[task_label]}\"\n")

            f.write("\n")

    print(f"\nРезультаты сохранены в: {output_file}")
    print("===== ОБРАБОТКА ЗАВЕРШЕНА =====")

    # Запуск модуля оценки для полного пайплайна
    # print("\n===== ЗАПУСК ОЦЕНКИ РАБОТ =====")
    # day5.evaluate_works(output_file, grades_dir)


if __name__ == "__main__":
    main(webres_dir, images_dir, output_file)

