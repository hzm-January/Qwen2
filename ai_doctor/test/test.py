import os.path
import string
from string import Template

import numpy as np


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

label = 2  # 假设当前的label值为2
numbers = {0, 1, 2, 3}
print(list(numbers - {label}))
print(list({0, 1, 2, 3} - {label}))

cm = np.array([[86, 2, 0, 0],
               [10, 7, 1, 0],
               [0, 7, 3, 6],
               [0, 0, 0, 21]])
# [[86  2  0  0]
#  [10  7  1  0]
#  [ 0  7  3  6]
#  [ 0  0  0 21]]
print(cm)
# Specificity for each class
specificity = []
for i in range(len(cm)):
    tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - np.diag(cm)[i])
    fp = np.sum(cm[:, i]) - np.diag(cm)[i]
    specificity.append(tn / (tn + fp))
specificity = np.array(specificity)
print(specificity)

la = [1, 2, 34, 55, 6, 7, 55, 1, 2, 3, 4]
print(la.count(1))

set1 = set([1, 23, 3]) - {3}
print(set1)

lb = [1, 2, 3]
aaa = [(la.index(id),id) for id in lb if id in la]
# aaa = {la.index(id):id for id in lb if id in la}
print(aaa)
print(aaa[0][0])
print(aaa[0][1])

if not lb:
    print('lb not')
else:
    print('lb ok')

test_str = "adasfas{aa}asfasf"
aa = 2
print(test_str)


# 示例数据
strarra = ["abbr1", "abbr2", "abbr3", "abbr4", "abbr5"]  # 简称数组 (假设长度为100)
strarrb = ["full1", "full2", "full3", "full4", "full5"]  # 全称数组 (假设长度为10)
# 假设这个json是简称与全称的对应关系
json_data = {
    "abbr1": "full1",
    "abbr2": "full2",
    "abbr6": "full6",
    "abbr3": "full3"
}
if "abbr1" in json_data:
    print(json_data["abbr1"])

# 找出strarra中所有有对应全称的简称
matched_abbr = [abbr for abbr in strarra if abbr in json_data]

# 输出有对应全称的简称
print(matched_abbr)
print("==========")
print(str(json_data))
