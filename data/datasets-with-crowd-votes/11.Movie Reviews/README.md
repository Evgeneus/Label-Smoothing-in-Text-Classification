This is the transformed "MovieReviews" dataset from "Deep Learning from Crowds" paper (Filipe Rodrigues et. al.).

-->trainl.csv:
1500 crowdsourced movie reviews, answers are ratings for movies (based on the reviews) from 0 to 10 ==> we converted the ratings into three classes: “negative” [0-3], “middling”(3-7), and “positive”[7-10] reviews.

-->test.csv:
Test dataset of 3508 reviews with gold labels.

- 3 Classes
- Train size: 1200
- Val size: 300
- Test size: 3508
- Num. answers per instance (± stddev.): 4.96 ± 0.196
- Mean annotators accuracy (± stddev.): 0.565 ± 0.258
- MV acciracy: 0.795,
- DS accuracy: 0.782,
- GLAD accuracy: 0.796,
- LFC accuracy: 0.782

Where MV- Majority voting, DS - Dawid and Skene, LFC - learning from crowd (without data features).

#### Data cleaning
We performed different data preprocessing techniques:
-   "raw" - original textual data
-   "clean"- textual data were preprocessed with the following steps:

        - make lowercase
        - remove punctuation marks
        - replace english contractions by full words
        - substitute numbers
        - remove english stopwords
        - words lemmatization
        - words stemming
        - strip html
        - removing accented characters
        - substitute URLs
        - substitute usernames
- "tobert" - textual data were preprocessed to feed Transformer models (e.g., BERT):

        - ASCII letters are removed
        - URLs, @[NAME], HTML tags are substituted by tokens
// Source code of the data cleaner used: https://github.com/Evgeneus/NLP-classification-tools //


BERT tokenizer:
-   90th percentile: 780.10 tokens
- 95th percentile: 864.10 tokens