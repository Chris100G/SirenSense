# Trained model goes here

The app loads `model.json` and its weight shard files from this directory. They are intentionally not included: model weights must be trained on the public UrbanSound8K dataset using the project script.

From the project root, run:

```bash
python training/train.py --dataset /path/to/UrbanSound8K --output model
```

The exported TensorFlow.js model returns two softmax probabilities in this order: `[not_siren, siren]`. The browser feature contract is 16 kHz mono, one-second windows, 64 log-mel bands by 64 time frames, per-window standardization. Do not substitute a model trained with different preprocessing or output ordering.

Until `model.json` and the referenced weights are present, live detection and demo inference remain disabled. The UI will not report sirens based on a placeholder or unvalidated heuristic.
