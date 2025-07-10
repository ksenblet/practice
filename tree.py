import re
import tarfile
import os
import time
from pathlib import Path
from symspellpy import SymSpell, Verbosity

ARCHIVE_PATH = 'school.tar.gz'
OUTPUT_DIR = 'results'
DICTIONARY_PATH = 'russian.utf-8'


def initialize_symspell(dict_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2)

    temp_freq_path = "temp_frequency_dict.txt"
    with open(dict_path, 'r', encoding='utf-8') as f_in, \
            open(temp_freq_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            word = line.strip()
            if word:
                f_out.write(f"{word}\t1\n")

    with open(temp_freq_path, 'r', encoding='utf-8') as temp_file:
        sym_spell.load_dictionary_stream(temp_file, term_index=0, count_index=1)
    os.remove(temp_freq_path)

    return sym_spell


def correct_text(text, sym_spell):
    words = re.findall(r'\b[а-яё-]+\b', text.lower())
    unique_words = []
    seen = set()

    for word in words:
        lower_word = word.lower()
        if lower_word not in seen:
            seen.add(lower_word)
            unique_words.append(lower_word)

    corrected = []
    for word in unique_words:
        suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)

    return corrected


def process_tar_archive(archive_path, output_dir, dict_path):
    sym_spell = initialize_symspell(dict_path)
    os.makedirs(output_dir, exist_ok=True)

    print("Обработка архива...")

    with tarfile.open(archive_path, 'r:gz') as tar:
        members = [m for m in tar if m.isfile() and m.name.endswith('.txt.webRes')]

        for member in members:
            try:
                content = tar.extractfile(member).read().decode('utf-8')
                corrected_words = correct_text(content, sym_spell)

                input_stem = Path(member.name).stem.replace('.txt', '')
                output_path = os.path.join(output_dir, f"{input_stem}_corrected.txt")

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(corrected_words))

            except Exception as e:
                print(f"Ошибка в файле {member.name}: {str(e)}")


if __name__ == "__main__":
    print("Начало обработки архива")

    start_time = time.time()
    process_tar_archive(ARCHIVE_PATH, OUTPUT_DIR, DICTIONARY_PATH)

    print("\n" + "=" * 50)
    print(f"Готово. Время обработки: {time.time() - start_time:.2f} сек")
    print(f"Результаты в: {os.path.abspath(OUTPUT_DIR)}")