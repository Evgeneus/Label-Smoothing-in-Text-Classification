# An Empirical Study on Soft Target and Label Smoothing in Text Classificationwith Crowd Annotations

### Abstract
Recent research on label smoothing has shown that using label distribution as the training target can enhance both model performance and probability calibration.  In the context of crowdsourcing, the empirical distribution over crowd labels, namely the soft target, offers a potentially better distribution than that obtained by label smoothing; however, the effect of soft targets on text classification remains unknown and how to best leverage crowd labels for creating label distributions remains largely unexplored. This paper introduces two generic soft target methods that can incorporate any label fusion methods for noise reduction and presents a systematic evaluation on 13 real-world datasets to understand the effect of both soft targets and label smoothing on text classification. We show that soft targets are a better approach than label smoothing especially to train well-calibrated models and that our proposed methods substantially improve model performance and probability calibration across datasets of different noise levels.

### /data/
This folder contains the datasets used in this paper:
1) datasets from Figure Eight (from-figure-eight) that contain already predefined label distribution and gold test sets that annotated by our team.
2) datasets with the actual crowd votes (datasets-with-crowd-votes).

More detailed description of the datasets could be found in corresponding folders in README.md


### /res/
This folder contains the details of our hyperparameter search, summary of results that includes ECE figures, F_1, F_01, F_10, ECE score, as well as Precision and Recall.

### /scr/transformers/
This folder contains the implementation of DistilBert model for Soft/Hard/sHard/Label smoothing training

### /scr/nnets/
This folder contains the implementation of simple neural network for Soft/Hard/sHard/Label smoothing training