# SirenSense

SirenSense is a mobile-first, installable PWA demonstrating how TinyML could turn nearby siren sounds into visual and haptic awareness for deaf and hard-of-hearing drivers. Audio feature extraction and inference are designed to stay in the browser; this static app has no audio-upload service.

**Live demo:** [chris100g.github.io/SirenSense](https://chris100g.github.io/SirenSense/)  
**Source repository:** [github.com/Chris100G/SirenSense](https://github.com/Chris100G/SirenSense)

> **Safety notice:** This is an educational prototype, not a certified driver-assistance or emergency-warning system. It may miss sirens or produce false alerts. Never rely on it instead of mirrors, safe driving practices, or local road rules. Do not interact with a phone while driving.

## Run locally

Microphone access requires HTTPS or localhost. From the project root, start any static server, for example:

```bash
python -m http.server 8000
```

Open `http://localhost:8000` and allow microphone permission. Listening and local audio-file analysis work immediately using an experimental tone-sweep signal-processing fallback. This fallback is not a trained classifier and is not reliable for driving. To use the intended TensorFlow.js ML pipeline, train and install model weights as described below. The PWA service worker is available on localhost and HTTPS. A static HTTPS host is required for installation and microphone access on deployed sites.

## Train and install a model

1. Download and extract the public [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). Preserve its `audio/fold1` … `audio/fold10` folders and `metadata/UrbanSound8K.csv`. Review the dataset license/terms before redistribution.
2. Use Python 3.10 or 3.11 in a supported environment (TensorFlow and TensorFlow.js converter compatibility varies by platform), then install:

   ```bash
   python -m pip install -r training/requirements.txt
   ```

3. Train, evaluate on held-out fold 10, and export an 8-bit-weight TensorFlow.js model:

   ```bash
   python training/train.py --dataset /path/to/UrbanSound8K --output model
   ```

   On Windows, use a path such as `C:\datasets\UrbanSound8K`. Optional flags: `--epochs 25` and `--batch-size 64`.
4. Confirm `model/model.json` and the referenced `.bin` shard(s) exist. The app expects the two-output softmax order `[not_siren, siren]`. Deploy these files with the app.

The architecture is a small two-block CNN. Training uses UrbanSound8K folds 1–8 for training, fold 9 for validation, and fold 10 for held-out evaluation. One deterministic 1-second crop per clip is used to limit memory. Labels are binary: UrbanSound8K `classID == 8` is siren; all other classes are not-siren. The browser and training script share the same 16 kHz mono, one-second, 64 × 64 log-mel feature contract.

The converter quantizes model weights; browser features and activations remain floating point. The target model size is below 2–5 MB but depends on converter output; inspect generated artifacts. Quantization alone is not evidence that the model is accurate or safe. Review precision, recall, false alarms, class balance, and performance on siren recordings not represented by this dataset before drawing conclusions. Training on UrbanSound8K does not establish real-world driving performance.

Without a trained `model.json`, the app runs an experimental tone-sweep signal-processing fallback so the microphone and audio-demo flows can be exercised without model setup. It is not a trained siren classifier, may miss sirens or trigger on other tonal sounds, and its displayed pattern score is not a probability. It must not be used for driving decisions. The UI clearly identifies this fallback; an installed trained model takes precedence and uses TensorFlow.js, loaded only when needed.

## PWA and deployment

Deploy the project folder as static files on GitHub Pages, Netlify, or another HTTPS static host. Preserve relative paths, including `model/` and `assets/`. There is no build step. The service worker caches the app shell; trained model files are cached as requested. The no-model signal-analysis fallback works offline after the shell has been cached. Trained-model inference also requires TensorFlow.js and its weights to have been cached successfully.

This repository is configured to publish through [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) on every push to `main`. For a new repository, set its Pages source to **GitHub Actions**. The first deployment may take a few minutes. For Netlify, publish the project root with no build command. Verify microphone permission, install prompt, offline behavior, and model download on the deployed HTTPS origin.

## Browser implementation

- `app.js`: microphone lifecycle, Web Audio capture, resampling, aligned log-mel features, TensorFlow.js inference, thresholding, visual and vibration alerts, demo-file decoding, theme/settings persistence.
- `service-worker.js` and `manifest.json`: installable app shell and offline cache.
- `about.html`: TinyML concepts, project workflow, accessibility context, engineering details, and limitations.
- `training/train.py`: dataset loading, feature extraction, compact CNN training, held-out evaluation, and quantized TensorFlow.js conversion.

The inference loop analyzes one-second windows every 500 ms. Two consecutive above-threshold predictions trigger the alert; browser and device performance affect latency. Vibration requires browser/device support and may be unavailable in some iOS browser contexts. This project uses the Web Audio `ScriptProcessorNode` for broad browser compatibility; a production implementation should test current platform support and consider an AudioWorklet.
