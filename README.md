# Auto Progressive Contextual spell checker for bangla

## Automatic Progressive Context-Sensitive Spelling Correction for Bangla Text Using BERT and Levenshtein Distance.

- Bert Masked Model (Added), Other model support(For example, LSTM/GRU based Masked Prediction model) will be added. 

- Bert NER Model (Added)
- Levenshtein Distance (Added)
- Dictionary Look up (Yet to add, but added word piece vocab look up) 
- Progressive spell checking
- New constraints added while checking the spelling

## Instruction
- Download a Bert Masked Model in "model/bangla-bert-base" (Recommeded https://huggingface.co/sagorsarker/bangla-bert-base)
- Download a Bert NER Model in "model/mbert-bengali-ner" (Recommended https://huggingface.co/sagorsarker/mbert-bengali-ner)
- Specify the Bert Masked Model and Bert NER Model controller class name in "config.json" 

**Example**:

```
from source.spell_checker import SpellChecker
sentence = "আগুনে কমক্ষে ৩৮ জন দগ্ধ হয়েছ".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=50, levenshtein_ratio_threshold=0.5)))
>>> ['আগুনে', 'কমপক্ষে', '৩৮', 'জন', 'দগ্ধ', 'হয়েছে']

sentence = "এখান লাঠিয়াল, ঢুলি, পালকিবাহক বা পেয়াদাগিরির পেশ বেছে নিতে হয়".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=50, levenshtein_ratio_threshold=0.5)))
>>> ['এখানে', 'লাঠিয়াল,', 'ঢুলি,', 'পালকিবাহক', 'বা', 'পেয়াদাগিরির', 'পেশা', 'বেছে', 'নিন', 'হয়']

sentence = "এক এলাকা সোলতা আহমেদের ছে আব্দুর রহমান (৩০)".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=50, levenshtein_ratio_threshold=0.5)))
['একই', 'এলাকার', 'সোলতা', 'আহমেদের', 'ছেলে', 'আব্দুর', 'রহমান', '(৩০)']

```

## Result

Evaluation dataset in created from https://github.com/habibsifat/Algorithm-for-Bengali-Error-Dataset-Generation. 

TP: Did not change the correct word / total correct word

FN: Change the correct word incorrectly / total incorrect word

FP: Did not change the incorrect word(Mark incorrect as correct) / total incorrect word

TN: Change the incorrect word correctly / total correct word

TN_PLUS: Change the incorrect word incorrectly 
            
| Model      | TP | FN | FP | TN | TN_PLUS |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Sagor Sarkar | 0.98366 | 0.01634| 0.67857 | 0.3214 | 0.0 |
            
