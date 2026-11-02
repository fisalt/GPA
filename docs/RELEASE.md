# Author release checklist

Source is published at the repository below. This guide describes subsequent
owner uploads and optional GitHub Release attachments; no release/tag is
created automatically by the build script.
The repository referenced by the manuscript is https://github.com/fisalt/GPA.
GitHub Release attachment publication is a separate action from creating a
source archive. No repository token or other credential is included.

For v0.3, include the [measured report](REPORT_V3.zh-CN.md),
[116-test validation record](VALIDATION_V3.md), and `docs/results/v3` rather
than advertising the historical v0.2 results as current. The selected pilot
has 93.36% four-token-ID EM versus 96.09% for its same-width/depth Transformer
on independent synthetic T256 data. Other three tasks score 100% in this
sample. It is not official RULER/NIAH or proven Transformer equivalence.
The dense GPA reader and optional counted alternative are distinct operators;
the counted alternative's poor results must remain visible.

1. The author has confirmed the pinned manuscript in `PROVENANCE.md` is the
   intended latest version. Retain that provenance when publishing this bundle.
2. Select an appropriate license and add `LICENSE`; review dependency licenses.
   The current archive deliberately grants no automatic open-source license.
   Add `LICENSE` to the build script's top-level whitelist after choosing one.
3. Review formulas/limitations and run `python -m pytest -q` in your environment.
   Optional CUDA tests must really pass, not just skip.
4. If code changes, update validation results and version, then rebuild:

   ```bash
   python scripts/build_release.py
   ```

5. Upload source to the repository using your authenticated GitHub session.
   For an initially empty repo, copy this directory excluding `dist`, caches,
   local environments and generated egg-info; then, in that clean directory:

   ```bash
   git init
   git add src tests examples scripts docs pyproject.toml README.md README.zh-CN.md .gitignore
   # Also git add LICENSE after the rights holder supplies it.
   git commit -m "Release dense multilevel GPA reference implementation"
   git branch -M main
   git remote add origin https://github.com/fisalt/GPA.git
   git push -u origin main
   ```

   If the repo is no longer empty, clone and integrate instead of force-pushing.

6. GitHub → Releases → Draft a new release → choose a reviewed tag (for example
   `v0.4.0`). Attach the ZIP and its `.sha256` file. Summarize the five reference
   modules, dense GDN implementation, actual validation, and approximation caveats.
   Publish only after the authors' review. No paper PDF/data/weights are in the ZIP.

## Distribution verification

`MANIFEST.sha256.json` inside the ZIP lists every bundled source/document hash.
The adjacent archive checksum protects the entire ZIP. The build uses sorted
members and fixed member timestamps, so an unchanged source tree produces the
same archive bytes. Both are integrity checks, not a cryptographic author signature.

Do not describe this release as reproducing the paper's model accuracy or as a
fully fused all-level kernel without running the corresponding experiments.
