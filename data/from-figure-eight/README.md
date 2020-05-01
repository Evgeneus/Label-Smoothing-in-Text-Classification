### 10 textual datasets for Soft-Target classification from Figure-Eight

This repository contains 10 textual preprocessed datasets that were initially provided by Figure-Eight (https://appen.com/resources/datasets/).
We split each of the crowdsourced datasets into Train/Test/Validation sets.
As the original data does not have Gold Labels, our team manually reannotated all Test sets (the detailed report about this can be found in "relabeling-test-data-stat.pdf").

CSV files can contain the following columns:
"text" is textual features for the data
"gold_label" is the gold label that was assigned by our team (only presented in test sets)
"crowd_label" is the label aggregated by Figure-Eight from crowd votes
"conf0,.., confN" are probabilistic confidences for labels 0,...,N that obtained from the crowd votes and provided by Figure-Eight

The "balanced-test-data" folder represents datasets with balanced test sets, while "unbalanced-test-data" contains unbalanced test sets.

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


#### Description of the datasets
 1. First GOP debate sentiment analysis (GOP2-sentiment)
 
         Tweets about the early August GOP debate in Ohio are analyzed by workers to decide if the tweet was relevant.
2. Disasters on social media (Disaster-relevance)

        The dataset includes over 10,000 tweets culled with a variety of searches like “ablaze”, “quarantine”, and “pandemonium”. Contributers analyzed them to decide whether the tweet referred to a disaster event.
        
3. Do these chemicals contribute to a disease? (Chemicals\&Disease)
    
        The dataset includes sentences in which both a chemical and a disease were present. Contributors are asked to decide whether the chemical directly contributed to the disease or caused it. 

4. Economic News Article Tone and Relevance (News-relevance)

        The dataset includes snippets and news articles, and the contributors are asked whether they are relevant to the US economy and, if so, what is the tone of the article. Tone was judged on a 9 point scale (from 1 to 9, with 1 representing the most negativity).                
        
5. Corporate messaging (Corporate-messaging)

        A data categorization job concerning what corporations actually talk about on social media. Contributors were asked to classify statements as information (objective statements about the company or it’s activities), dialog (replies to users, etc.), or action (messages that ask for votes or ask users to click on links, etc.)
    
6. First GOP debate sentiment analysis-Sentiment (GOP3-sentiment)

        Same dataset with "First GOP debate sentiment analysis", but here contributors analyze the sentiment of the tweet; positive, negative, or neutral.
        
7. Twitter sentiment analysis: Self-driving cars (Self-driving-cars)

        The dataset contains tweets and the contributors are asked to decide whether their sentiment are positive, negative, or neutral.
        
8. Drug relation database (Drug-relation)

        The dataset includes color-coded sentences about the relationships of a drug, symptoms, and the diseases. Contributors determined whether the sentence is related to a personal experience, impersonal experience, or is a question.
        
9. Indian terrorism deaths database (Deaths-in-India)

        The dataset includes sentences from the South Asia Terrorism Portal. Contributors counted the deaths mentioned in a sentence and whether they were terrorists, civilians, or security forces.

10. First GOP debate sentiment analysis-Subject (GOP5-subject)

        Same dataset with "First GOP debate sentiment analysis", but here contributors analyze what subject was mentioned; religion, women's issues (not abortion though), foreign policy, racial issues, or abortion.
                                    
        