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

sentence = "পুলিশ আসা আগে ডাকাত পালিয়ে গোছে".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>> ['পুলিশ', 'আসার', 'আগে', 'ডাকাত', 'পালিয়ে', 'গেছে']

sentence = "এক এলাকা সোলতা আহমেদের ছে আব্দুর রহমান (৩০)".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>>['একই', 'এলাকার', 'সোলতা', 'আহমেদের', 'ছেলে', 'আব্দুর', 'রহমান', '(৩০)']

sentence = "২০১৫ সালের নভেম্বরে প্রান্সে জলবায়ূ সসেলনে বিশেবর ২০০ দেশ অংশগ্রহণ করে".split(" ")
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>>['২০১৫', 'সালের', 'নভেম্বরে', 'ফ্রান্সের', 'জলবায়ূ', 'সম্মেলনে', 'বিশ্বের', '২০০', 'দেশ', 'অংশগ্রহণ', 'করে']

sentence = "পরে তাদসের উিপর হামলা করে এলোপাতাড়ি কুপাতে থাকে"
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>>['পরে', 'তাদের', 'উপর', 'হামলা', 'করে', 'এলোপাতাড়ি', 'কুপাতে', 'থাকে']

sentence = "তাূরা দেখেন ঢাকার দূই সিটিতে মশা মারতে যে ওষধূ ছিটানো হয় তা অকার্যকর"
print(SpellChecker().prediction(sentence=sentence, k=100)))
>>>['তারা', 'দেখে', 'ঢাকার', 'দুই', 'সিটিতে', 'মশা', 'মারতে', 'যে', 'ওষুধ', 'ছিটানো', 'হত', 'তা', 'অকার্যকর']

```

## Result

Evaluation dataset in created from https://github.com/habibsifat/Algorithm-for-Bengali-Error-Dataset-Generation. 

**TP:** Did not change the correct word / total correct word.

**FN:** Change the correct word incorrectly / total correct word.

**FP:** Did not change the incorrect word (Mark incorrect as correct) / total incorrect word.

**TN:** Change the incorrect word correctly / total incorrect word.

**TN_PLUS:** Change the incorrect word incorrectly.

### Result of bangla bert for different language models
            
| Model | Top N| TP | FN | FP | TN | TN_PLUS |
| :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :------------ |
| Sagor Sarkar | 50 | 0.9782 | 0.0218 | 0.4150 | 0.5017 | 0.0833 | -->
| NWP(W2V Skipgram)| 50 | 0.9879 | 0.0121 | 0.6612 | 0.2825 | 0.0563 | -->



### The result of spell checker based on bangla bert for different conditions

We conducted the experiment on different value of maximum edit distance (ml). The conditions are given below:

• **C1:** ml = Probable misspell word(mw)’s length//2.

• **C2:** ml = mw’s length//2 if mw’s length > 4 else ml = 2.

• **C3:** ml = mw’s length//2 if mw’s length > 6 else ml = 2.

• **C4:** ml = mw’s length//2 if mw’s length > 6 else ml = 3

| Condition | TP | FN | FP | TN | TN_PLUS |
| :----------- | :----------- | :----------- | :----------- | :----------- | :----------- |
| C1 | 0.9837 | 0.0163 | 0.6779 | 0.3209 | 0.0012 |
| C2 | 0.9782 | 0.0218 | 0.4150 | 0.5017 | 0.0833 |
| C3 | 0.9776 | 0.0224 | 0.5534 | 0.4410 | 0.0056 |
| C4 | 0.9623 | 0.0377 | 0.6498 | 0.2010 | 0.1492 |



## API
We also added API.


