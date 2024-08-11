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