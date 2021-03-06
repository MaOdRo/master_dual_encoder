Masterarbeit zur Wikipedia Image/Caption Competition
==============================

Als Teil meiner Masterarbeit wird hier die Bearbeitung der Wikipedia Image/Caption Competition dargestellt. In dieser Arbeit wird ein multilinguales Dual Encoder System auf Basis von [CLIP](https://github.com/openai/CLIP) erstellt.


<p align="center">
<img src="/data/images/Modell Übersicht.png" width="600" height="300">
</p>

Ebenfalls wird die Auswirkung von multilingualem Prompt Engineering auf die Performanz des Modells untersucht.

<p align="center">
<img src="/data/images/PromptEngineering.png" width="600" height="300">
</p>

## Getting Started

### Prerequisites

Folgende Python Libraries müssen zunächst installiert werden.

```
...
...
```
### Examples

Für die Visualisation der Daten:
```
python src/visualization/....py
python src/generation/....py
...
```

Zusammenfassende Notebooks, die den Ablauf der Bearbeitung zeigen

- [Einleitung/Datainspection](https://colab.research.google.com/drive/1p0GIyOQP1hrQpwrephUh10zfPb5LvikB?hl=de#scrollTo=WR78qszh6mPA)
- [Benchmark/Zero Shot CLIP](https://colab.research.google.com/drive/1wLefrr7n329jjH4XGHPOtYW67-5T-Ufm?hl=de#scrollTo=lmP4P3IPshFC)
- [Multilingual Modell](https://colab.research.google.com/drive/1hb-9B_D8eXfI7U8YCjC7xenMvu_TvUKh?hl=de#scrollTo=D37L7HrR4W3Z)
- [Prompt Engineering](https://colab.research.google.com/drive/1R6q1L_9rx54mTGAOMBpw2JzsquL4woGe#scrollTo=Nj0N5f-cgOPF)

Für die Erstellung und Evaluation des Projekts:

```
python src/models/..._model.py
python src/models/..training_logs.py

python src/evaluation/multilingual_model_....py
python src/evaluation/zero_shot_....py
python src/evaluation/prompt_engineering_....py
