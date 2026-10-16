# Software installation and platform scope

Working environment observed during preparation: **Python 3.11.17**, macOS arm64, Apple M4 MacBook Air, 16 GB unified memory, external SSD, PyTorch MPS float32. Exact direct package versions are in [environment.json](../provenance/environment.json) and [requirements.txt](../requirements.txt). No packages or environments were installed/replaced during preparation.

For count-only verification, choose an existing Python interpreter and install only the reviewer requirement if needed on your own machine:

```sh
PYTHON=python3
"$PYTHON" -m pip install -r requirements-reviewer.txt
"$PYTHON" release_tools/verify.py --output verification_outputs/reviewer
```

For actual source/tests/notebooks/research reproduction, use your chosen Python 3.11 interpreter:

```sh
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" release_tools/pipeline.py --help
"$PYTHON" release_tools/run_core_tests.py --output verification_outputs/core
"$PYTHON" release_tools/smoke.py --output verification_outputs/smoke
```

These are user installation instructions, not commands executed by the preparation agent. The pins report the actual working environment, not a fully resolved cross-platform lock or proof that every wheel is available from every package index. Use the appropriate official distribution for your platform; if a pinned build is unavailable, record the discrepancy and obtain the matching environment/build rather than silently changing versions and calling it an exact reproduction. Transitive/bundled native-library identities are not completely locked. This limitation remains explicit.

Portable commands import the unchanged `src` directly, so editable installation is optional. The export's `pyproject.toml` adds the existing PyTorch dependency to the original metadata; it does not replace model source. Historical environment paths in unchanged legacy source/tests are provenance defaults, not required paths for the `release_tools` commands.

CPU supports the lightweight verifier and synthetic core suite. Full real training and gated final inference explicitly require available MPS and never silently fall back to CPU. The existing trainer has CUDA state support, but this release has no validated CUDA full-pipeline route or CUDA reproduction claim. POSIX `fcntl` in experiment/checkpoint code means native Windows full research execution is unsupported. Count verification uses NumPy and standard Python and does not import POSIX/PyTorch research modules.

MPS training and checkpoint resume are not promised bitwise-identical across runs, OS/PyTorch/NumPy versions, workers or hardware. CPU synthetic equivalence tests establish the tested fixtures only. Record software versions, source hashes and device in every independent reproduction; its checkpoints have new identities.

Third-party license metadata is inventoried in [THIRD_PARTY.md](THIRD_PARTY.md). Dependencies are not vendored. The author approved MIT for authorized original code, separately from dependency and dataset terms; see [license scope](LICENSE_SCOPE.md).
