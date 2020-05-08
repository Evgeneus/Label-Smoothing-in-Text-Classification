This is the reshaped dataset from "Ex Machina: Personal Attacks Seen at Scale" paper (Wulczyn et al).

- 2 Classes:  {0, 1}, I.e., Does the comment contain a personal attack/harassment of not? (Yes:1, No:0)
- Train size: 69526
- Val size: 23160
- Test size:23178


** Properties for the original crowdsourced data (train, val, test- all together) **

    - Num of workers: 4053
    - Total num. answers: 1365217
    - Num. answers per worker (± stddev.): 336.841 ± 296.493
    - Num. answers per instance (± stddev.):  11.783 ± 4.879
    - Mean annotators accuracy (± stddev.):  None
    - Inter-annotator agreement: Krippendorf’s alpha score of 0.45


** Train data properties (ALL votes per instance) **
    
    - Num of workers: 4048
    - Total num. answers: 762046
    - Num. answers per worker (± stddev.): 188.2 ± 162.43
    - Num. answers per instance (± stddev.):  10.96 ± 3.808
    - Mean annotators accuracy (± stddev.):-
    - Maj. vot. accuracy: MV accuracy: None,
    - DS accuracy: None,
    - GLAD accuracy: None,
    - LFC accuracy: None


** Train data properties (5 votes per instance) **

    - Num of workers (± stddev.): 4016.333 ± 4.109
    - Total num. answers: 347630
    - Num. answers per worker (± stddev.): 86.669  ± 73.892
    - Num. answers per instance (± stddev.):  5.0 ± 0.0
    - Mean annotators accuracy (± stddev.): 0.90 ± 0.113
    - Maj. vot. accuracy: MV accuracy: 0.966,
    - DS accuracy: 0.951,
    - GLAD accuracy: 0.958,
    - LFC accuracy: 0.951


** Train data properties (3 votes per instance) **

    - Num of workers (± stddev.): 3992.0 ± 2.160
    - Total num. answers: 65534
    - Num. answers per worker (± stddev.): 52.275 ± 44.572
    - Num. answers per instance (± stddev.):  3.0 ± 0.0
    - Mean annotators accuracy (± stddev.): 90.0 ± 0.122
    - Maj. vot. accuracy: MV accuracy: 0.947,
    - DS accuracy: 0.942,
    - GLAD accuracy: 0.945,
    - LFC accuracy: 0.942 

// Source code of the data cleaner used: https://github.com/Evgeneus/NLP-classification-tools //
