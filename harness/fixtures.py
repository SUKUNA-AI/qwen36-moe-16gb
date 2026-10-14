"""Versioned public synthetic holdout; independent of KV scale calibration."""
import json
from bench import ROOT, document

RU = [
 ('Почему двоичный поиск работает только на упорядоченном массиве?', 'Двоичный поиск сравнивает искомое значение со средним элементом. Упорядоченность позволяет отбросить половину диапазона, поскольку все её элементы заведомо меньше или больше искомого значения. На каждом шаге границы поиска сближаются, поэтому сложность составляет O(log n).'),
 ('Объясни отличие очереди от стека.', 'Очередь извлекает элементы в порядке поступления: первым выходит тот, кто пришёл первым. Стек извлекает последним добавленный элемент. Очередь подходит для обработки заданий, а стек применяется при обходе в глубину и хранении вызовов функций.'),
 ('Что означает идемпотентная операция?', 'Повторное выполнение идемпотентной операции с одинаковыми аргументами имеет тот же результат, что и однократное выполнение. Например, установка конкретного значения идемпотентна, а увеличение счётчика на единицу обычно не является идемпотентным.'),
 ('Почему нельзя сравнивать скорость программы только по одному запуску?', 'Один запуск зависит от прогрева кэшей, фоновой нагрузки, частоты процессора и случайных задержек. Нужны повторные измерения в одинаковых условиях. Следует сравнивать распределения времени, учитывать разброс и проверять, что результат вычислений остаётся правильным.'),
 ('Как обнаружить утечку памяти в сервисе?', 'Нужно измерять память после завершения одинаковых запросов и сравнивать её со стабильным исходным уровнем. Рост рабочего набора сам по себе не доказывает утечку: память может удерживаться кэшем. Полезно учитывать живые объекты, резерв аллокатора и файловый кэш.'),
 ('Зачем нужен таймаут сетевого запроса?', 'Таймаут ограничивает время ожидания ответа и помогает освободить занятые ресурсы. Его выбирают с учётом нормальной длительности операции. После истечения срока запрос отменяют, а повтор разрешают только тогда, когда это безопасно для данной операции.')]
CODE = [
 ('binary_search(a, x)', 'def binary_search(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo if lo < len(a) and a[lo] == x else -1\n'),
 ('merge_sorted(a, b)', 'def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result + a[i:] + b[j:]\n'),
 ('prefix_sums(a)', 'def prefix_sums(a):\n    out = [0]\n    for value in a:\n        out.append(out[-1] + value)\n    return out\n'),
 ('deduplicate(a)', 'def deduplicate(a):\n    seen = set()\n    out = []\n    for value in a:\n        if value not in seen:\n            seen.add(value)\n            out.append(value)\n    return out\n'),
 ('count_words(text)', 'def count_words(text):\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n'),
 ('rotate(a, k)', 'def rotate(a, k):\n    if not a:\n        return []\n    k %= len(a)\n    return a[-k:] + a[:-k] if k else list(a)\n')]


def teacher_cases(tokenizer):
    cases=[]
    for i,(question,answer) in enumerate(RU):
        cases.append(dict(case=f'ru-{i}',prompt=question,target=answer))
    for i,(name,answer) in enumerate(CODE):
        cases.append(dict(case=f'code-{i}',prompt=f'Напиши только код Python для {name}.',target=answer))
    for i in range(4):
        target=json.dumps({'city':['Казань','Томск','Пермь','Омск'][i],'temperature_c':12+i,'unit':'celsius'},ensure_ascii=False)
        cases.append(dict(case=f'json-{i}',prompt=f'Верни JSON: город {json.loads(target)["city"]}, температура {12+i}, единица celsius.',target=target))
    for i,length in enumerate([4096,16384,65536,120000]):
        body=document(tokenizer,length,881009+i)
        target='Код проекта: СЕВЕР-741. Число насосов: 17. Ответственный: Мария. Итоговая сумма: 816 рублей.'
        a=len(body)//3;b=2*len(body)//3
        prompt='Проект имеет код СЕВЕР-741.\n'+body[:a]+'\nУ проекта 17 насосов.\n'+body[a:b]+'\nОтветственный за проект — Мария.\n'+body[b:]+'\nЦена за один насос: 48 рублей. Укажи код, число насосов, ответственного и итоговую стоимость.'
        cases.append(dict(case=f'long-{length}',prompt=prompt,target=target))
    for i,case in enumerate(cases):
        case['seed']=900000+i
        case['ids']=tokenizer.encode(case['target'],add_special_tokens=False)
    return cases


def prepare(tokenizer, *, create=False):
    from freetoken.quality_evidence import strict_json, digest
    cases=teacher_cases(tokenizer)
    payload={'fixture_version':1,'targets':{str(c['seed']):{k:c[k] for k in ('case','ids')} for c in cases}}
    path=ROOT/'config/teacher-fixture.json'
    if path.exists():
        if digest(strict_json(path.read_text())) != digest(payload):
            raise RuntimeError('existing teacher fixture differs; refusing to replace reference targets')
    elif create:
        with path.open('x') as out:out.write(json.dumps(payload))
        path.chmod(0o600)
    else:
        raise RuntimeError('teacher fixture missing; explicit --prepare is required, no implicit generation')
    return cases
