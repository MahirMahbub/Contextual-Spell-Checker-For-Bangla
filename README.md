# Contextual spell checker for bangla

## Automatic Context-Sensitive Spelling Correction for Bangla Text Using BERT and Levenshtein Distance.

- Bert Masked Model (Added), Other model support(For example, LSTM/GRU based Masked Prediction model) will be added. 

- Bert NER Model (Added)
- Levenshtein Distance (Added)
- Dictionary Look up (Yet to add)

## Instruction
- Download a Bert Masked Model in "model/bangla-bert-base" (Recommeded https://huggingface.co/sagorsarker/bangla-bert-base)
- Download a Bert NER Model in "model/mbert-bengali-ner" (Recommended https://huggingface.co/sagorsarker/mbert-bengali-ner)
- Specify the Bert Masked Model and Bert NER Model controller class name in "config.json" 

**Example**:

```
from source.spell_checker import SpellChecker
sentence = "নাম প্রকা না করার শর্তে এক প্রত্যক্ষদর্শী ওই ঘটনার বর্ণন দেন".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=100, levenshtein_ratio=0.5)))
>>> ['নাম', 'প্রকাশ', 'না', 'করার', 'শর্তে', 'এক', 'প্রত্যক্ষদর্শী', 'ওই', 'ঘটনার', 'বর্ণনা', 'দেন']

sentence = "পুলিশ আসা আগে ডাকাত পালিয়ে গোছে".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=100, levenshtein_ratio=0.5)))
>>> ['পুলিশ', 'আসার', 'আগে', 'ডাকাত', 'পালিয়ে', 'গেছে']

```
