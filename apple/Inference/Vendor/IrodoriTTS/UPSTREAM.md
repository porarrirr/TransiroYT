Source: https://github.com/Corvelis/irodori-tts-coreml
Revision: 948f17da32e94fdb87eeb9420b76326312665d49 (v0.2.0)

Vendored library sources only. The upstream license, NOTICE and component licenses
are retained. TransiroYT changes:

- Use a separate `transiroyt-coreml-v1` compiled-model cache.
- On the iOS simulator, use the supplied standalone flexible stage-1 decoder
  and CPU execution: the iOS 27 simulator rejects multifunction descriptions.
- Load the pinned multifunction decoder through its default w64 function;
  named w64 loading failed on the first macOS/iOS simulator preparation.
- Add `prepareWatermark` to skip AudioSeal loading/warm-up when disabled. The
  shared TransiroYT engine passes false and always synthesizes with watermark nil.
- Add a sampling step count to `synthesizeText` and `IrodoriEngine.synthesize`
  (default 4, as upstream); the file-dubbing setting chooses it.
- Add an optional speaking length (`seconds`) to `synthesizeText` and
  `IrodoriEngine.synthesize` for one-utterance synthesis. When given, it sets the
  latent frame count (40 ms each) instead of the duration model; the model's
  prediction is still computed and the length limit still applies.
- Remove upstream migration code that deletes old global Irodori caches. Existing
  models, caches and references must survive application updates.

The original source and model conversion/rebuild instructions are available at the
revision above. Model files are separately downloaded through the versioned INT8
manifest with per-file SHA-256 validation; they are not embedded in this source.
