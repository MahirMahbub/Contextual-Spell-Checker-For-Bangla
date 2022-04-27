# Auto Progressive Contextual Spell Checker For Bangla

## Automatic Progressive Context-Sensitive Spelling Correction for Bangla Text Using BERT and Levenshtein Distance.

- Bert Masked Model (Added), Other model support(For example, LSTM/GRU based Masked Prediction model) will be added. 

- Bert NER Model (Added)
- Levenshtein Distance (Added)
- Dictionary Look up (Added), 451742 unique words from Oscar 2019 dataset.
- Progressive spell checking with NER (Added)
- New constraints added while checking the spelling (Added)

## Instruction
- Download a Bert Masked Model in "model/bangla-bert-base" (Recommeded https://huggingface.co/sagorsarker/bangla-bert-base)
- Download a Bert NER Model in "model/mbert-bengali-ner" (Recommended https://huggingface.co/sagorsarker/mbert-bengali-ner)
- Specify the Bert Masked Model and Bert NER Model controller class name in "config.json" 
- [Download](https://drive.google.com/file/d/1Z98rG7CSvnHFUSOAZ0jtWCCAYf_nBde0/view?usp=sharing) dictionary and place in at /data/output/
- Run app.py for API(Based of Fastapi)

**Example**:

```
from source.spell_checker import SpellChecker

sentence = "আগুনে কমক্ষে ৩৮ জন দগ্ধ হয়েছ".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>> ['আগুনে', 'কমপক্ষে', '৩৮', 'জন', 'দগ্ধ', 'হয়েছে']

sentence = "এক এলাকা সোলতা আহমেদের ছে আব্দুর রহমান (৩০)".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>>['একই', 'এলাকার', 'সোলতা', 'আহমেদের', 'ছেলে', 'আব্দুর', 'রহমান', '(৩০)']

sentence = "পরে ডাকাতরা তাদসের উিপর হামলা করে এলোপাতাড়ি কুপাতে থাকেন"
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>>['পরে', 'ডাকাতরা', 'তাদের', 'উপর', 'হামলা', 'করে', 'এলোপাতাড়ি', 'কুপাতে', 'থাকেন' ]

sentence = "তাূরা দেখেন ঢাকার দূই সিটি করপোরেশনে মশা মারতে যে ওষধূ ছিটানো হয় তা অকাযংকর"
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>>['তারা', 'দেখেন', 'ঢাকার', 'দুই', 'সিটি', 'করপোরেশনে', 'মশা', 'মারতে', 'যে', 'ওষুধ', 'ছিটানো', 'হয়', 'তা', 'অকাযংকর']

```

## Result

Evaluation dataset in created from https://github.com/habibsifat/Algorithm-for-Bengali-Error-Dataset-Generation. 

**TP:** Did not change the correct word / total correct word.

**FN:** Change the correct word incorrectly / total correct word.

**FP:** Did not change the incorrect word (Mark incorrect as correct) / total incorrect word.

**TN:** Change the incorrect word correctly / total incorrect word.

**TN_PLUS:** Change the incorrect word incorrectly.
            
<!-- | Model      | Top N| TP | FN | FP | TN | TN_PLUS |
| :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :------------ |
| Sagor Sarkar | 100 | 0.9837 | 0.0163| 0.6786 | 0.3214 | 0.0000 | -->
            
| Model      | Top N| TP | FN | FP | TN | TN_PLUS |
| :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :------------ |
| Sagor Sarkar | 50 | 0.9782 | 0.0218 | 0.4150 | 0.5017 | 0.0833 | -->
| NWP(W2V Skipgram)| 50 | 0.9879 | 0.0121 | 0.6612 | 0.2825 | 0.0563 | -->



## API
We also added API. Here is the API response.

![api](https://github.com/MahirMahbub/Contextual-Spell-Checker-For-Bangla/blob/master/api.png)


