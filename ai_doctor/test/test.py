import os.path
import string
from string import Template


class NoteTemplate(Template):
    delimiter = '$'
    idpattern = r'[^{}]+'


template = NoteTemplate('Hello, ${K1 F (D)}! You have $$10 in your $$.')
result = template.substitute({'K1 F (D)': '12'})
print(result)

a = os.path.join('/aaaa/bb', 'ccc')
print(a)

d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
d_ = sorted(sorted(d.items(), key=lambda item: item[1], reverse=True))
print(d_[:2])

def mmm(d):
    for k, v in d.items():
        d[k] += 1

mmm(d)
print(d)

aa = []
if aa:
    print('aa')
else:
    print('no aa')