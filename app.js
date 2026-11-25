(function () {
  'use strict';

  var SAMPLE_RATE = 16000;
  var WINDOW_SECONDS = 1;
  var WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS;
  var HOP_SAMPLES = 250;
  var FFT_SIZE = 512;
  var MEL_BANDS = 64;
  var FRAMES = 64;
  var MODEL_URL = './model/model.json';
  var TFJS_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
  var SIREN_FFT_SIZE = 1024;
  var SIREN_FRAME_HOP = 512;
  var model = null;
  var detectorMode = 'signal';
  var tensorflowPromise = null;
  var audioContext = null;
  var mediaStream = null;
  var sourceNode = null;
  var processorNode = null;
  var analyser = null;
  var muteNode = null;
  var rafId = null;
  var active = false;
  var lastPredictionAt = 0;
  var positiveWindows = 0;
  var audioBuffer = new Float32Array(0);
  var visualData = new Uint8Array(0);
  var alertActive = false;
  var inferenceBusy = false;
  var settings = { threshold: 0.75, vibration: true, pattern: 'triple', intensity: false };

  var $ = function (id) { return document.getElementById(id); };
  var canvas = $('visualizer');
  var canvasContext = canvas ? canvas.getContext('2d') : null;
  var listenButton = $('listen-button');
  var statusPill = $('status-pill');
  var statusText = $('status-text');
  var confidenceValue = $('confidence-value');
  var confidenceBar = $('confidence-bar');
  var confidenceCaption = $('confidence-caption');
  var modelMessage = $('model-message');
  var alertBox = $('alert-box');
  var fileInput = $('audio-file');

  function setStatus(kind, text) {
    statusPill.className = 'status-pill' + (kind ? ' is-' + kind : '');
    statusText.textContent = text;
  }
  function setConfidence(value, caption) {
    if (value === null || !Number.isFinite(value)) {
      confidenceValue.innerHTML = '—<small>%</small>';
      confidenceBar.style.width = '0%';
      confidenceCaption.textContent = caption || 'Waiting for audio';
      return;
    }
    var percent = Math.max(0, Math.min(100, Math.round(value * 100)));
    confidenceValue.innerHTML = percent + '<small>%</small>';
    confidenceBar.style.width = percent + '%';
    confidenceCaption.textContent = caption || 'Siren likelihood';
  }
  function setMessage(message) { modelMessage.textContent = message; }
  function enableSignalDetector(reason) {
    model = null;
    detectorMode = 'signal';
    setStatus('ready', 'Experimental detector ready');
    $('confidence-label').textContent = 'SIREN PATTERN SCORE';
    setMessage(reason || 'Experimental signal-pattern detector ready. This is not a trained ML model and is not for safety decisions.');
    $('visualizer-caption').textContent = 'Ready for local signal analysis';
    listenButton.disabled = false;
  }
  function loadSettings() {
    try {
      var saved = JSON.parse(localStorage.getItem('sirensense-settings') || '{}');
      if (typeof saved.threshold === 'number') settings.threshold = Math.max(0.5, Math.min(0.95, saved.threshold));
      settings.vibration = saved.vibration !== false;
      settings.pattern = saved.pattern === 'pulse' ? 'pulse' : 'triple';
      settings.intensity = saved.intensity === true;
      if (saved.theme === 'light') document.body.classList.add('light-theme');
    } catch (error) {
      console.warn('SirenSense settings could not be read.', error);
    }
    $('threshold').value = Math.round(settings.threshold * 100);
    $('threshold-value').textContent = Math.round(settings.threshold * 100) + '%';
    $('vibration-toggle').checked = settings.vibration;
    $('vibration-pattern').value = settings.pattern;
    $('intensity-toggle').checked = settings.intensity;
    document.body.classList.toggle('high-intensity', settings.intensity);
    updateThemeLabel();
  }
  function saveSettings() {
    try {
      localStorage.setItem('sirensense-settings', JSON.stringify({
        threshold: settings.threshold, vibration: settings.vibration, pattern: settings.pattern,
        intensity: settings.intensity, theme: document.body.classList.contains('light-theme') ? 'light' : 'dark'
      }));
    } catch (error) {
      console.warn('SirenSense settings could not be saved.', error);
    }
  }
  function updateThemeLabel() {
    var isLight = document.body.classList.contains('light-theme');
    $('theme-toggle').setAttribute('aria-label', isLight ? 'Switch to dark theme' : 'Switch to light theme');
  }
  function initializeAboutTheme() {
    try {
      var saved = JSON.parse(localStorage.getItem('sirensense-settings') || '{}');
      if (saved.theme === 'light') document.body.classList.add('light-theme');
    } catch (error) {
      console.warn('SirenSense theme preference could not be read.', error);
    }
    updateThemeLabel();
    $('theme-toggle').addEventListener('click', function () {
      document.body.classList.toggle('light-theme');
      updateThemeLabel();
      try {
        var savedSettings = JSON.parse(localStorage.getItem('sirensense-settings') || '{}');
        savedSettings.theme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
        localStorage.setItem('sirensense-settings', JSON.stringify(savedSettings));
      } catch (error) {
        console.warn('SirenSense theme preference could not be saved.', error);
      }
    });
  }

  function loadTensorFlow() {
    if (window.tf) return Promise.resolve(window.tf);
    if (tensorflowPromise) return tensorflowPromise;
    tensorflowPromise = new Promise(function (resolve, reject) {
      var script = document.createElement('script');
      script.src = TFJS_URL;
      script.onload = function () {
        if (window.tf) resolve(window.tf);
        else reject(new Error('TensorFlow.js loaded without initializing.'));
      };
      script.onerror = function () { reject(new Error('TensorFlow.js could not be downloaded.')); };
      document.head.appendChild(script);
    });
    return tensorflowPromise;
  }
  async function loadModel() {
    try {
      var response = await fetch(MODEL_URL);
      if (!response.ok) {
        enableSignalDetector('No trained model is installed. Using the experimental on-device tone-sweep detector instead; it is not ML and is not for safety decisions.');
        return;
      }
      var modelJson = await response.json();
      if (!modelJson.modelTopology || !Array.isArray(modelJson.weightsManifest)) {
        throw new Error('The model file is not a valid TensorFlow.js layers model.');
      }
      var tf = await loadTensorFlow();
      await tf.ready();
      model = await window.tf.loadLayersModel(MODEL_URL);
      detectorMode = 'model';
      $('confidence-label').textContent = 'SIREN CONFIDENCE';
      setStatus('ready', 'Model ready');
      setMessage('On-device model loaded. Start listening to begin.');
      listenButton.disabled = false;
    } catch (error) {
      console.warn('SirenSense could not load the trained model; using its signal-analysis fallback.', error);
      enableSignalDetector('The trained model could not be loaded. Using the experimental on-device tone-sweep detector; it is not ML and is not for safety decisions.');
    }
  }

  function resizeCanvas() {
    if (!canvas || !canvasContext) return;
    var rect = canvas.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvasContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function drawVisualizer() {
    if (!canvasContext || !canvas) return;
    var width = canvas.clientWidth, height = canvas.clientHeight;
    canvasContext.clearRect(0, 0, width, height);
    var bars = 56, gap = 3, barWidth = Math.max(2, (width - gap * (bars - 1) - 48) / bars);
    var total = bars * barWidth + (bars - 1) * gap, start = (width - total) / 2;
    var centerY = height * 0.57;
    for (var i = 0; i < bars; i++) {
      var sample = visualData.length ? visualData[Math.floor(i * visualData.length / bars)] / 255 : 0.5;
      var base = active ? Math.abs(sample - 0.5) * 2 : 0;
      var wave = active ? base * 90 + 3 : (Math.sin(i * 0.4) + 1) * 4 + 3;
      var barHeight = Math.max(4, Math.min(height * 0.65, wave));
      var gradient = canvasContext.createLinearGradient(0, centerY - barHeight / 2, 0, centerY + barHeight / 2);
      gradient.addColorStop(0, alertActive ? '#ff8588' : '#ddff8b');
      gradient.addColorStop(1, alertActive ? '#ff4d55' : '#82aa42');
      canvasContext.fillStyle = gradient;
      canvasContext.globalAlpha = active ? 0.42 + Math.min(base, 0.45) : 0.32;
      canvasContext.beginPath();
      canvasContext.roundRect(start + i * (barWidth + gap), centerY - barHeight / 2, barWidth, barHeight, 3);
      canvasContext.fill();
    }
    canvasContext.globalAlpha = 1;
    rafId = window.requestAnimationFrame(drawVisualizer);
  }
  function stopVisualizer() {
    if (rafId !== null) window.cancelAnimationFrame(rafId);
    rafId = null;
  }

  function resample(input, inputRate) {
    if (inputRate === SAMPLE_RATE) return input;
    var ratio = inputRate / SAMPLE_RATE;
    var output = new Float32Array(Math.floor(input.length / ratio));
    for (var i = 0; i < output.length; i++) {
      var position = i * ratio, left = Math.floor(position), fraction = position - left;
      var right = Math.min(left + 1, input.length - 1);
      output[i] = input[left] * (1 - fraction) + input[right] * fraction;
    }
    return output;
  }
  function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
  function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }
  function createMelFilters() {
    var lowMel = hzToMel(40), highMel = hzToMel(SAMPLE_RATE / 2), points = [];
    for (var i = 0; i < MEL_BANDS + 2; i++) points.push(melToHz(lowMel + (highMel - lowMel) * i / (MEL_BANDS + 1)));
    return points.map(function (hz) { return Math.floor((FFT_SIZE + 1) * hz / SAMPLE_RATE); });
  }
  var melFilters = createMelFilters();
  function fftPower(frame, transformSize) {
    var size = transformSize || FFT_SIZE;
    var real = new Float32Array(size), imag = new Float32Array(size);
    for (var i = 0; i < size; i++) {
      var sample = frame[i] || 0;
      real[i] = sample * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1)));
    }
    // In-place radix-2 Cooley-Tukey FFT with bit-reversed input ordering.
    for (var i = 1, j = 0; i < size; i++) {
      var bit = size >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        var tempReal = real[i]; real[i] = real[j]; real[j] = tempReal;
        var tempImag = imag[i]; imag[i] = imag[j]; imag[j] = tempImag;
      }
    }
    for (var stageSize = 2; stageSize <= size; stageSize <<= 1) {
      var half = stageSize >> 1, angleStep = -2 * Math.PI / stageSize;
      for (var base = 0; base < size; base += stageSize) {
        for (var offset = 0; offset < half; offset++) {
          var angle = angleStep * offset, evenIndex = base + offset, oddIndex = evenIndex + half;
          var oddReal = real[oddIndex] * Math.cos(angle) - imag[oddIndex] * Math.sin(angle);
          var oddImag = real[oddIndex] * Math.sin(angle) + imag[oddIndex] * Math.cos(angle);
          real[oddIndex] = real[evenIndex] - oddReal;
          imag[oddIndex] = imag[evenIndex] - oddImag;
          real[evenIndex] += oddReal;
          imag[evenIndex] += oddImag;
        }
      }
    }
    var power = new Float32Array(size / 2 + 1);
    for (var b = 0; b < power.length; b++) power[b] = (real[b] * real[b] + imag[b] * imag[b]) / size;
    return power;
  }
  function extractMel(samples) {
    var padded = new Float32Array(WINDOW_SAMPLES);
    padded.set(samples.subarray(Math.max(0, samples.length - WINDOW_SAMPLES)));
    var raw = new Float32Array(FRAMES * MEL_BANDS);
    var frame = new Float32Array(FFT_SIZE);
    for (var t = 0; t < FRAMES; t++) {
      var offset = t * HOP_SAMPLES;
      frame.fill(0);
      for (var n = 0; n < FFT_SIZE && offset + n < padded.length; n++) frame[n] = padded[offset + n];
      var power = fftPower(frame);
      for (var m = 0; m < MEL_BANDS; m++) {
        var left = melFilters[m], center = Math.max(left + 1, melFilters[m + 1]), right = Math.max(center + 1, melFilters[m + 2]);
        var sum = 0;
        for (var bin = left; bin < center && bin < power.length; bin++) sum += power[bin] * (bin - left) / (center - left);
        for (var bin2 = center; bin2 < right && bin2 < power.length; bin2++) sum += power[bin2] * (right - bin2) / (right - center);
        raw[t * MEL_BANDS + m] = Math.log(Math.max(sum, 1e-10));
      }
    }
    // Match training normalization: log-mel values are standardized over the window.
    var mean = 0;
    for (var x = 0; x < raw.length; x++) mean += raw[x];
    mean /= raw.length;
    var variance = 0;
    for (var y = 0; y < raw.length; y++) variance += Math.pow(raw[y] - mean, 2);
    var std = Math.sqrt(variance / raw.length) || 1;
    for (var z = 0; z < raw.length; z++) raw[z] = (raw[z] - mean) / std;
    return raw;
  }

  function analyzeSirenPattern(samples) {
    if (samples.length < SAMPLE_RATE / 2) return 0;
    var window = new Float32Array(SIREN_FFT_SIZE);
    var minBin = Math.ceil(400 * SIREN_FFT_SIZE / SAMPLE_RATE);
    var maxBin = Math.floor(1800 * SIREN_FFT_SIZE / SAMPLE_RATE);
    var frequencies = [], tonalities = [];
    for (var start = 0; start + SIREN_FFT_SIZE <= samples.length; start += SIREN_FRAME_HOP) {
      for (var i = 0; i < SIREN_FFT_SIZE; i++) window[i] = samples[start + i];
      var power = fftPower(window, SIREN_FFT_SIZE);
      var totalEnergy = 0, peakBin = minBin, peakEnergy = 0;
      for (var bin = minBin; bin <= maxBin; bin++) {
        totalEnergy += power[bin];
        if (power[bin] > peakEnergy) { peakEnergy = power[bin]; peakBin = bin; }
      }
      if (totalEnergy < 1e-9) continue;
      var localEnergy = power[peakBin];
      if (peakBin > minBin) localEnergy += power[peakBin - 1];
      if (peakBin < maxBin) localEnergy += power[peakBin + 1];
      frequencies.push(peakBin * SAMPLE_RATE / SIREN_FFT_SIZE);
      tonalities.push(localEnergy / totalEnergy);
    }
    if (frequencies.length < 6) return 0;

    var toneAverage = tonalities.reduce(function (sum, value) { return sum + value; }, 0) / tonalities.length;
    var minFrequency = Math.min.apply(null, frequencies);
    var maxFrequency = Math.max.apply(null, frequencies);
    var pitchSpan = maxFrequency - minFrequency;
    var direction = 0, travel = 0, reversals = 0;
    for (var frame = 1; frame < frequencies.length; frame++) {
      var change = frequencies[frame] - frequencies[frame - 1];
      if (Math.abs(change) < 24) continue;
      var nextDirection = change > 0 ? 1 : -1;
      if (direction && nextDirection !== direction && travel >= 55) reversals++;
      if (nextDirection !== direction) travel = 0;
      direction = nextDirection;
      travel += Math.abs(change);
    }
    var toneScore = Math.max(0, Math.min(1, (toneAverage - 0.10) / 0.55));
    var sweepScore = Math.max(0, Math.min(1, (pitchSpan - 90) / 360));
    var reversalScore = Math.min(1, reversals / 2);
    return toneScore * 0.35 + sweepScore * 0.45 + reversalScore * 0.20;
  }

  async function infer(samples) {
    if (inferenceBusy || !active) return;
    inferenceBusy = true;
    try {
      var score;
      if (model) {
        var features = extractMel(samples);
        var input = window.tf.tensor4d(features, [1, FRAMES, MEL_BANDS, 1]);
        var output = model.predict(input);
        var values = await output.data();
        input.dispose(); output.dispose();
        // Training exports [non-siren, siren] softmax probabilities in that order.
        score = values.length > 1 ? values[1] : values[0];
      } else {
        score = analyzeSirenPattern(samples);
      }
      setConfidence(score, detectorMode === 'model'
        ? (score >= settings.threshold ? 'Possible siren' : 'Below alert threshold')
        : (score >= settings.threshold ? 'Siren-like sweep' : 'No clear sweep'));
      if (score >= settings.threshold) {
        positiveWindows++;
        if (positiveWindows >= 2) triggerAlert();
      } else {
        positiveWindows = 0;
        if (alertActive && score < settings.threshold * 0.7) clearAlert();
      }
    } catch (error) {
      console.error('On-device inference failed.', error);
      stopListening();
      setStatus('error', 'Inference error');
      setMessage('On-device audio analysis stopped after an error. Reload the page to try again.');
    } finally {
      inferenceBusy = false;
    }
  }

  function appendSamples(samples) {
    var combined = new Float32Array(audioBuffer.length + samples.length);
    combined.set(audioBuffer); combined.set(samples, audioBuffer.length);
    audioBuffer = combined.length > WINDOW_SAMPLES * 2 ? combined.slice(-WINDOW_SAMPLES * 2) : combined;
    if (audioBuffer.length >= WINDOW_SAMPLES && Date.now() - lastPredictionAt >= 500) {
      lastPredictionAt = Date.now();
      infer(audioBuffer.slice(-WINDOW_SAMPLES));
    }
  }
  function onAudioProcess(event) {
    var input = event.inputBuffer.getChannelData(0);
    visualData = analyser ? new Uint8Array(analyser.frequencyBinCount) : visualData;
    if (analyser) analyser.getByteTimeDomainData(visualData);
    appendSamples(resample(input, audioContext.sampleRate));
  }

  async function startListening() {
    if (detectorMode !== 'model' && detectorMode !== 'signal') return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('error', 'Microphone unavailable');
      setMessage('This browser does not support microphone access. Try a current browser over HTTPS.');
      return;
    }
    listenButton.disabled = true;
    setMessage('Requesting microphone permission…');
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false
      });
      var AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error('Web Audio API is not supported in this browser.');
      audioContext = new AudioContextClass();
      if (audioContext.state === 'suspended') await audioContext.resume();
      sourceNode = audioContext.createMediaStreamSource(mediaStream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      processorNode.onaudioprocess = onAudioProcess;
      muteNode = audioContext.createGain();
      muteNode.gain.value = 0;
      sourceNode.connect(analyser);
      analyser.connect(processorNode);
      processorNode.connect(muteNode);
      muteNode.connect(audioContext.destination);
      active = true;
      audioBuffer = new Float32Array(0);
      positiveWindows = 0;
      lastPredictionAt = 0;
      listenButton.disabled = false;
      listenButton.classList.add('is-stop');
      $('listen-button-label').textContent = 'Stop listening';
      $('visualizer-caption').textContent = 'Listening for nearby sirens';
      setStatus('listening', 'Listening');
      setMessage(model
        ? 'Audio is analyzed locally by the trained model. Keep this page open while monitoring.'
        : 'Experimental tone-sweep analysis is running locally (not a trained ML model). Keep this page open; do not rely on it for driving.');
      stopVisualizer();
      drawVisualizer();
    } catch (error) {
      console.error('Microphone could not be started.', error);
      stopListening();
      setStatus('error', error.name === 'NotAllowedError' ? 'Permission denied' : 'Microphone error');
      if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
        setMessage('Microphone permission was denied or blocked. Allow microphone access in browser settings and use HTTPS.');
      } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        setMessage('No microphone was found. Connect a microphone or use the audio demo.');
      } else {
        setMessage(error.message || 'The microphone could not be started. Try again or use the audio demo.');
      }
    }
  }
  function stopListening() {
    active = false;
    if (processorNode) { processorNode.onaudioprocess = null; processorNode.disconnect(); processorNode = null; }
    if (muteNode) { muteNode.disconnect(); muteNode = null; }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (analyser) { analyser.disconnect(); analyser = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(function (track) { track.stop(); }); mediaStream = null; }
    if (audioContext) { audioContext.close().catch(function (error) { console.warn('Audio context close failed.', error); }); audioContext = null; }
    audioBuffer = new Float32Array(0);
    visualData = new Uint8Array(0);
    if (listenButton) {
      listenButton.disabled = false;
      listenButton.classList.remove('is-stop');
      $('listen-button-label').textContent = 'Start listening';
    }
    if (statusPill && statusPill.classList.contains('is-listening')) setStatus('ready', model ? 'Model ready' : 'Experimental detector ready');
    if ($('visualizer-caption')) $('visualizer-caption').textContent = model ? 'Ready when you are' : 'Ready for local signal analysis';
    if (model) setMessage('On-device model loaded. Start listening to begin.');
  }

  function vibrate() {
    if (!settings.vibration || !navigator.vibrate) return;
    var pattern = settings.pattern === 'pulse' ? [500, 250, 500, 250, 500] : [130, 110, 130, 170, 520];
    navigator.vibrate(pattern);
  }
  function triggerAlert() {
    if (alertActive) return;
    alertActive = true;
    alertBox.hidden = false;
    document.body.classList.toggle('high-intensity', settings.intensity);
    setStatus('alert', 'Siren detected');
    $('visualizer-caption').textContent = 'Possible siren — check surroundings';
    vibrate();
  }
  function clearAlert() {
    alertActive = false;
    if (alertBox) alertBox.hidden = true;
    if (navigator.vibrate) navigator.vibrate(0);
    document.body.classList.toggle('high-intensity', settings.intensity);
    if (active) {
      setStatus('listening', 'Listening');
    } else {
      setStatus('ready', detectorMode === 'model' ? 'Model ready' : 'Experimental detector ready');
      $('visualizer-caption').textContent = 'Ready when you are';
    }
  }

  async function processDemoFile(file) {
    if (!file) return;
    if (!file.type.startsWith('audio/')) {
      $('demo-message').textContent = 'Please choose an audio file (WAV or MP3).';
      return;
    }
    $('demo-message').textContent = model
      ? 'Decoding and analyzing audio locally with the trained model…'
      : 'Analyzing local audio with the experimental tone-sweep detector (not ML)…';
    var context = null;
    try {
      var AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error('Audio decoding is not supported in this browser.');
      context = new AudioContextClass();
      var decoded = await context.decodeAudioData(await file.arrayBuffer());
      var mono = new Float32Array(decoded.length);
      for (var ch = 0; ch < decoded.numberOfChannels; ch++) {
        var channel = decoded.getChannelData(ch);
        for (var i = 0; i < channel.length; i++) mono[i] += channel[i] / decoded.numberOfChannels;
      }
      var samples = resample(mono, decoded.sampleRate);
      var score = null;
      for (var start = 0; start < samples.length; start += Math.floor(WINDOW_SAMPLES / 2)) {
        var segment = samples.slice(start, Math.min(start + WINDOW_SAMPLES, samples.length));
        if (segment.length < WINDOW_SAMPLES / 2) continue;
        var padded = new Float32Array(WINDOW_SAMPLES); padded.set(segment);
        var segmentScore;
        if (model) {
          var input = window.tf.tensor4d(extractMel(padded), [1, FRAMES, MEL_BANDS, 1]);
          var output = model.predict(input);
          var values = await output.data();
          input.dispose(); output.dispose();
          segmentScore = values.length > 1 ? values[1] : values[0];
        } else {
          segmentScore = analyzeSirenPattern(padded);
        }
        score = Math.max(score === null ? 0 : score, segmentScore);
      }
      if (score === null) throw new Error('The audio is too short to analyze. Choose a clip at least half a second long.');
      setConfidence(score, detectorMode === 'model'
        ? (score >= settings.threshold ? 'Possible siren in clip' : 'Below alert threshold')
        : (score >= settings.threshold ? 'Siren-like sweep in clip' : 'No clear sweep in clip'));
      if (score >= settings.threshold) { positiveWindows = 2; triggerAlert(); }
      else if (alertActive && score < settings.threshold * 0.7) clearAlert();
      $('demo-message').textContent = 'Local analysis complete. ' + (model
        ? 'Model predictions are experimental; verify the sound yourself.'
        : 'This pattern score is not a siren probability or trained classifier. Verify the sound yourself.');
    } catch (error) {
      console.error('Audio demo could not be analyzed.', error);
      $('demo-message').textContent = error.message || 'The file could not be decoded. Try a WAV or MP3 file.';
    } finally {
      if (context) await context.close().catch(function (error) { console.warn('Demo audio context close failed.', error); });
    }
  }

  function initialize() {
    if (!listenButton) {
      initializeAboutTheme();
      return;
    }
    loadSettings();
    resizeCanvas();
    drawVisualizer();
    loadModel();
    listenButton.addEventListener('click', function () {
      if (active) { stopListening(); clearAlert(); }
      else startListening();
    });
    $('dismiss-alert').addEventListener('click', clearAlert);
    $('threshold').addEventListener('input', function (event) {
      settings.threshold = Number(event.target.value) / 100;
      $('threshold-value').textContent = event.target.value + '%';
      saveSettings();
    });
    $('vibration-toggle').addEventListener('change', function (event) { settings.vibration = event.target.checked; saveSettings(); });
    $('vibration-pattern').addEventListener('change', function (event) { settings.pattern = event.target.value; saveSettings(); });
    $('intensity-toggle').addEventListener('change', function (event) {
      settings.intensity = event.target.checked;
      document.body.classList.toggle('high-intensity', settings.intensity);
      saveSettings();
    });
    $('theme-toggle').addEventListener('click', function () {
      document.body.classList.toggle('light-theme');
      updateThemeLabel();
      saveSettings();
    });
    fileInput.addEventListener('change', function (event) { processDemoFile(event.target.files[0]); });
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('pagehide', function () { stopListening(); stopVisualizer(); });
  }
  document.addEventListener('DOMContentLoaded', initialize);
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('./service-worker.js').catch(function (error) {
        console.error('SirenSense could not register its offline service worker.', error);
      });
    });
  }
}());
