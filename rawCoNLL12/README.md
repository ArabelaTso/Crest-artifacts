## CoNLL-12

This directory stores the validation and test set of CoNLL-12. 
You may skip it and directly use the sampled data under `Output/`

TL;DR

This directory include intra-sentence coreference resolution annotations:

- `dev.english.v4_gold_conll.sen.json`: validation set 
- `test.english.v4_gold_conll.sen.json`: test set


### File Format

*sen.json
```
{
    "doc_id": "bc/cctv/00/cctv_0000", 
    "sentence": ["With", "their", "unique", "charm", ",", "these", "well", "-", "known", "cartoon", "images", "once", "again", "caused", "Hong", "Kong", "to", "be", "a", "focus", "of", "worldwide", "attention", "."], 
    "cluster": [[[5, 10], [1, 1]]]}

```

### Doc id

| ID  | Source                        |
| ------- |-------------------------------|
| bc | Broadcast Conversation |
| bn | Broadcast News         |
| mz | Magazine (Newswire)     |
| nw | Newswire           |
| pt | Pivot Corpus                  |
| tc | Telephone conversation  |
| wb | Web text               |


When sampling, we select 