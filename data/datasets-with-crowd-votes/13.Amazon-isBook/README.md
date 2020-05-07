In this dataset, we crowdsource 1000 amazon reviews and the task was to identify whether a  review was written on a book or other product.

To train classifiers, we used only "_golden"==False reviews as these reviews were used as test questions in crowdsourcing jobs.

- 2 Classes: isBook {0, 1}
- Train size: 1000
- Val size: 1000
- Test size:4000
- Train size crowdsourced: 1000
- Num of workers: 263
- Total num. answers: 4907
- Num. answers per worker (± stddev.): 18.6 ± 4.6
- Num. answers per instance (± stddev.): 4.9 ± 0.41
- Mean annotators accuracy (± stddev.): 0.946 ± 0.076
- Maj. vot. accuracy: MV acciracy: 0.964,
- DS accuracy: 0.961,
- GLAD accuracy: 0.964,
- LFC accuracy: 0.961 