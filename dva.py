import json
import re
import time
from collections import defaultdict
from tqdm import tqdm


def levenshtein(s1, s2):
    if s1 == s2:
        return 0

    len1, len2 = len(s1), len(s2)
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    current = list(range(len1 + 1))
    for i in range(1, len2 + 1):
        previous, current = current, [i] + [0] * len1
        for j in range(1, len1 + 1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1]
            if s1[j - 1] != s2[i - 1]:
                change += 1
            current[j] = min(add, delete, change)

    return current[len1]


def correct_words(original_words, dictionary_set, length_dict):
    corrected = []
    seen_words = set()  # уже обработаны

    for word in tqdm(original_words, desc="Прогресс:"):
        lower_word = word.lower()
        if lower_word in seen_words:
            continue

        seen_words.add(lower_word)

        if lower_word in dictionary_set:
            corrected.append(lower_word)
            continue

        candidates = []
        target_len = len(lower_word)
        for length in [target_len - 2, target_len - 1, target_len, target_len + 1, target_len + 2]:
            if length in length_dict:
                candidates.extend(length_dict[length])

        if not candidates:
            corrected.append(lower_word)
            continue

        closest = min(candidates, key=lambda w: levenshtein(lower_word, w))
        corrected.append(closest)

    return corrected


def process_text(text):
    words = re.findall(r'\b[а-яёa-z]+\b', text, flags=re.IGNORECASE)
    unique_words = []
    seen = set()

    for word in words:
        lower_word = word.lower()
        if lower_word not in seen:
            seen.add(lower_word)
            unique_words.append(lower_word)

    return unique_words


def main():
    output_name = 'result.txt'
    start_total = time.time()

    print("Загрузка словаря...")
    dict_set, len_dict = load_dictionary('russian.utf-8')

    print("Обработка входного файла...")
    with open('input.txt.webRes', 'r', encoding='utf-8') as f:
        data = json.load(f)

    text = data['data']['text']
    ordered_words = process_text(text)
    print(f"Найдено уникальных слов: {len(ordered_words)}")

    corrected_words = correct_words(ordered_words, dict_set, len_dict)

    with open(output_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrected_words))

    print(f"\nГотово! Общее время: {time.time() - start_total:.2f} сек")
    print(f"Результат сохранен в {output_name}")


def load_dictionary(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        dictionary = [line.strip().lower() for line in f if line.strip()]

    dict_set = set(dictionary)
    len_dict = defaultdict(list)
    for word in dictionary:
        len_dict[len(word)].append(word)

    return dict_set, len_dict


if __name__ == "__main__":
    main()
