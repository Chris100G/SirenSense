'use strict';

var CACHE_NAME = 'sirensense-v1';
var APP_SHELL = [
  './',
  './index.html',
  './about.html',
  './styles.css',
  './app.js',
  './manifest.json',
  './assets/siren.svg'
];
var TFJS_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

self.addEventListener('install', function (event) {
  event.waitUntil(caches.open(CACHE_NAME).then(function (cache) {
    return Promise.all(APP_SHELL.concat([TFJS_URL]).map(function (url) {
      return cache.add(url).catch(function () { return undefined; });
    }));
  }).then(function () { return self.skipWaiting(); }));
});

self.addEventListener('activate', function (event) {
  event.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (key) { return key.indexOf('sirensense-') === 0 && key !== CACHE_NAME; })
      .map(function (key) { return caches.delete(key); }));
  }).then(function () { return self.clients.claim(); }));
});

self.addEventListener('fetch', function (event) {
  if (event.request.method !== 'GET') return;
  var requestUrl = new URL(event.request.url);
  if (requestUrl.origin !== self.location.origin && requestUrl.href.indexOf(TFJS_URL) !== 0) return;
  event.respondWith(caches.match(event.request).then(function (cached) {
    if (cached) return cached;
    return fetch(event.request).then(function (response) {
      if (response && (response.ok || response.type === 'opaque')) {
        var copy = response.clone();
        caches.open(CACHE_NAME).then(function (cache) { cache.put(event.request, copy); });
      }
      return response;
    });
  }));
});
