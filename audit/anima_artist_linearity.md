# Anima Artist Weight Linearity Verification

This is the saved feasibility result requested before implementing any cross-cell
artist-effect cache. No cache implementation is included in this change.

## Source-backed pipeline

The installed ComfyUI source (`aaabf342`) forces Qwen token weights to `1.0` in
`comfy/text_encoders/anima.py`, keeps T5 weights in `t5xxl_weights`, and runs:

```text
raw Qwen embedding + T5 ids -> LLMAdapter -> output * t5xxl_weights -> 512-row padding
```

The installed Mixer (`a1e61eb`) encodes each artist as
`<artist>\n<base_prompt>`, aligns each post-Adapter result, linearly combines the
results using its `::weight` injection weights, and applies the projected delta
to the base context. Its run cleanup clears the GPU-side embedding cache.

The Mixer README describes CLIP weighting as pre-Adapter, but the installed
ComfyUI execution path above is authoritative for this environment. The source
path was tested directly.

## Tested identity

The tested identity was:

```text
E(weighted_artist + prompt) - E(prompt)
    == weight * (E(unit_artist + prompt) - E(prompt))
```

The metrics below are relative L2 error between the two sides after the Mixer
token alignment. `1.0` is an exact match. Tests used the installed Qwen text
encoder and CPU Anima Adapter weights, not a mock encoder.

| Model / prompt | weight 0.4 | weight 0.8 | weight 1.2 | weight 1.6 |
| --- | ---: | ---: | ---: | ---: |
| anima-base, `1girl, solo, white dress` (raw delta) | 36.27% | 7.10% | 4.90% | 11.18% |
| Anima 2.9B, same prompt (projected delta) | 25.48% | 4.57% | 3.09% | 6.99% |
| Anima 2.9B, `masterpiece, dynamic pose, city at night` | 37.45% | 7.42% | 5.14% | 11.75% |
| Anima 2.9B, empty base prompt | 13.51% | 2.30% | 1.54% | 3.46% |

At weight `0.0`, the artist-only rows become zero, but the base-prompt rows
still contain the artist-prefix contextual change (L2 `4.6486` in the first
base-model case). That is a direct counterexample to scalar delta caching.
Artist-only rows were near-linear (about `3e-4` relative error, consistent with
FP16), but the full aligned effect is not.

## Safe conclusion

Unit-effect caching is not mathematically equivalent to existing
`(@artist:weight)` prompt weighting and must not be enabled silently.

The external Mixer's `::weight` path is linearly composable for a fixed model,
base prompt, token alignment, dtype, and device. It could support a separate
opt-in injection-weight mode, but that would change the meaning of the current
artist weight. The current task deliberately leaves that cache and semantic
change unimplemented.
