# Deep Learning Projekt – LIME Explainability auf CNN mit LFW

Dieses Projekt zeigt, wie man mit einem selbst trainierten CNN-Modell auf dem LFW-Datensatz Gesichter klassifiziert und mithilfe von **LIME (Local Interpretable Model-agnostic Explanations)** visuelle Erklärungen für einzelne Vorhersagen erzeugt.

## Projektstruktur

```
lime_cnn_final_project/
├── scripts/              # Python-Skripte
│   ├── train.py
│   ├── lime_explain.py
│   ├── dataset_split.py
│   └── model.py
├── slurm/                # SLURM-Batch-Skripte für das HPC-Cluster
│   ├── train.sh
│   └── lime.sh
├── data/                 # Datenstruktur (nach dem Split)
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── models/               # Trainierte Modelle (.pth)
├── outputs/lime_heatmaps # LIME-Erklärungen als Bilder
└── logs/                 # SLURM-Ausgaben
```

## Setup Cluster
------------------------------------
### Umgebung vorbereiten:
```bash Code
module load Miniforge3/24.11.3-0
conda create -n limefaces python=3.11
conda activate limefaces
pip install torch torchvision lime matplotlib scikit-learn
```

## Training starten
```bash Code
sbatch slurm/train.sh
```

## LIME-Visualisierungen erzeugen
```bash Code
sbatch slurm/lime.sh
```
------------------------------------

Achtung: In `lime.sh` müssen `--model_path` und `--num_classes` an dein Modell angepasst werden.

## Datensplit anpassen
Verwende `dataset_split.py`, um `train/val/test` neu zu erzeugen. Der Datensatz (z. B. `lfw_funneled`) muss vorher entpackt unter `data/raw/` liegen.

## Ergebnis
- Modell gespeichert in `results/<JOB_ID>/models/cnn_lime.pth`
- Heatmaps gespeichert in `results/<JOB_ID>/lime_heatmaps/`

## Autor
Anonymisiert zur öffentlichen Weitergabe.
