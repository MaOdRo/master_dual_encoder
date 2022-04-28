Masterarbeit zur Wikipedia Image/Caption Competition
==============================

Als Teil meiner Masterarbeit wird hier die Bearbeitung der Masterarbeit zur Wikipedia Image/Caption Competition dargestellt. Multilingual Problem... Diese Arbeit...dual encoder (also known as two-tower) neural network models...

Orientiert an https://keras.io/examples/nlp/nl_image_search/
<p align="center">
<img src="/data/images/Modell Übersicht.png" width="600" height="300">
</p>

## Getting Started

### Prerequisites

First, make sure the following python libraries are installed.

```
A.
...
```
### Examples

Für die Visualization der Daten:
```
python src/visualization/visualize_training_data.py
python src/generation/visualize_test_data.py

```

Eininge Notebooks, die den Ablauf der Bearbeitung zeigen

- [Einleitung/Datainspection](https://colab.research.google.com/drive/1p0GIyOQP1hrQpwrephUh10zfPb5LvikB?hl=de#scrollTo=WR78qszh6mPA)
- [Benchmark/Zero Shot CLIP](https://colab.research.google.com/drive/1wLefrr7n329jjH4XGHPOtYW67-5T-Ufm?hl=de#scrollTo=lmP4P3IPshFC)
- [Multilingual Modell](https://colab.research.google.com/drive/1hb-9B_D8eXfI7U8YCjC7xenMvu_TvUKh?hl=de#scrollTo=D37L7HrR4W3Z)
- [Prompt Engineering](https://colab.research.google.com/drive/1R6q1L_9rx54mTGAOMBpw2JzsquL4woGe#scrollTo=Nj0N5f-cgOPF)

Für die Erstellung und Evaluation des Projekts:
train_model.py - 5 Epochen

```
python src/models/train_model.py
python src/models/show_training_logs.py

python src/evaluation/multilingual_model_evaluation.py
python src/evaluation/zero_shot_evaluation.py
python src/evaluation/prompt_engineering_evaluation.py
