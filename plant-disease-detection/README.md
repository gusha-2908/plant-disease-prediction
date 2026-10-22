# Plant Disease Detection Using Deep Learning

Image-based plant disease classification with transfer learning,
Grad-CAM explainability and a Streamlit demo.

## Structure
```
data/        train/ val/ test/  (one sub-folder per class)
src/         dataset, model, train, evaluate, gradcam
app/         streamlit_app.py
models/      saved checkpoints
outputs/     metrics, confusion matrices, training history
notebooks/   exploration and experiments
```

## Setup
```bash
pip install -r requirements.txt
```

## Data
Download PlantVillage / New Plant Diseases Dataset (Kaggle) and arrange as
`data/train`, `data/val`, `data/test`, each containing one folder per class.

## Train (run from project root)
```bash
python -m src.train --model mobilenet_v2 --epochs 10
python -m src.train --model efficientnet_b0 --epochs 10
python -m src.train --model resnet50 --epochs 10
```

## Evaluate
```bash
python -m src.evaluate --checkpoint models/mobilenet_v2_best.pt --split test
```

## Demo
```bash
streamlit run app/streamlit_app.py
```

## Roadmap
- [ ] Dataset analysis (class balance, sample images)
- [ ] Train and compare 3 models
- [ ] Lab-to-field test (PlantVillage -> PlantDoc) and domain-gap discussion
- [ ] Grad-CAM analysis of failure cases
- [ ] Treatment recommendation module
- [ ] Deploy on Hugging Face Spaces
