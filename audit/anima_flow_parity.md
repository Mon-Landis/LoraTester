# AnimaFlow XY / production parity audit

## Scope

The installed Comfyui-anima-sampler and Anima-Artist-Mixer Python implementations were compared with the XY adapter. Read-only execution-history inspection used the running local ComfyUI service; no production prompt was queued and no production workflow was edited.

## Observed execution differences

The latest recorded XY execution began on October 7, 2026 at 02:03:38 China Standard Time; the latest recorded production execution began at 02:08:28. They are useful concrete examples, not proof that every pair the user compared used these exact inputs.

| Input | XY execution | Production execution |
| --- | --- | --- |
| Seed / steps / CFG | 114514 / 20 / 6 | 114514 / 20 / 6 |
| Solver / schedule / shift | flow_euler / flow_cosmos_rho7 / 3 | same |
| CFG mode | const | ramp cfg |
| Latent dimensions / batch | 800 × 640 / 1 | same |
| External Settings node fields | connected | same fields |
| Mixer strength | no configuration input; legacy implicit 1.6 | explicit 1.0 |
| Negative prompt | short negative with nsfw | longer negative with different tags and additional weighted censor tags |
| Base prompt | quality prefix ends in a comma; no base artist tag | quality prefix without that comma; also includes a base artist tag |

The latest two-artist XY entry and production artist chain use the same two artist names. Earlier XY history contains a larger artist combination, so earlier and latest comparisons must not be conflated. An ordinary base-prompt artist is not extracted into the explicit Mixer chain by this plugin.

## Code findings and changes

- AnimaFlow XY forwards the nine public controls, conditioning, latent metadata, optional Settings and VAE into the registered external sampler's public `sample()` method. It does not remap solver/schedule choices or fall back to KSampler.
- The implicit Mixer strength mismatch was real. AnimaFlow XY now obtains implicit Mixer defaults from the installed external Mixer schema. Explicit configurations retain precedence; legacy tester/configuration-node defaults are unchanged for saved-workflow compatibility.
- Advanced external Mixer options were not forwarded. The Mixer Configuration node now accepts `ANIMA_OPTS`, snapshots it and forwards an independent copy to the external Mixer.
- Random Anchor-Q seeds are conditional on enabling Anchor-Q with an empty anchor list. The inspected executions did not enable that path. This is a reproduction boundary, not an established cause for those executions.
- Detail logs now expose the final positive/base prompt, negative prompt, artist chain, Mixer parameters, submitted Flow controls/Settings/latent metadata and upstream sampler log. Detail logging remains optional.
- `None` versus connected default Settings is intentionally preserved: the external sampler can select a different disconnected `final_clean_pass` default. CLIP artist weights remain CLIP weights; they are not rewritten into linear `::weight` injection.

## Validation

`tests/test_anima_flow_upstream.py` loads the installed external packages when both `LORA_TESTER_ANIMA_FLOW_ROOT` and `LORA_TESTER_ANIMA_MIXER_ROOT` are provided.

- Direct versus XY execution of the actual public sampler produces identical normalized backend arguments for all three CFG modes, with absent, empty and default Settings (nine combinations).
- A full unsigned 64-bit seed, conditioning tensors, latent samples, batch indexes and noise masks reach the same backend values.
- Direct versus XY artist Pack/Adapter Mixer paths produce identical artist embedding tensors, token IDs, base conditioning, user weights and Mixer strength/alignment state.
- An advanced-axis override matches a directly configured external Settings object.

These are CPU invocation-boundary tests. Only the final denoising backend is replaced with a deterministic tensor-returning probe; the real public sampler, normalization, Settings builder, artist Pack, Mixer patch construction and VAE decode path execute. This does **not** establish pixel-level GPU equivalence of the user's full production workflow.
