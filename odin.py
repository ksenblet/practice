def levenshtein(s1, s2):
    len1, len2 = len(s1), len(s2)
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    current = range(len1 + 1)
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


with open('russian.utf-8', 'r', encoding='utf-8') as f:
    dictionary = [line.strip() for line in f if line.strip()]

word = input("Введите слово: ").lower()

if dictionary:
    res = min(dictionary, key=lambda w: levenshtein(word, w.lower()))
    distance = levenshtein(word, res.lower())

    print(f"Самое близкое слово: {res}")
    print(f"Расстояние Левенштейна: {distance}")
else:
    print("Словарь пуст")
