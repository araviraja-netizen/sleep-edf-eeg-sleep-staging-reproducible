# Third-party dependency and source-rights review

Dependencies are not vendored or redistributed as package binaries in this source release. The observed distribution license metadata and license-file paths are recorded in [dependency_licenses.json](../provenance/dependency_licenses.json); these are evidence, not a blanket legal compatibility determination.

| Distribution | Observed version | License metadata summary |
| --- | --- | --- |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.17.1 | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. All rights reserved. Redistribution and use in source and binary f |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| pandas | 3.0.6 | BSD 3-Clause License Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team All ri |
| matplotlib | 3.11.2 | License agreement for matplotlib versions 1.3.0 and later ========================================================= 1. This LICENS |
| mne | 1.13.2 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| nbformat | 5.11.1 | BSD 3-Clause License - Copyright (c) 2001-2015, IPython Development Team - Copyright (c) 2015-, Jupyter Development Team All right |
| nbclient | 0.11.0 | BSD 3-Clause License Copyright (c) 2020-, Jupyter Development Team All rights reserved. Redistribution and use in source and binar |
| ipykernel | 7.4.0 | BSD-3-Clause |
| jupyter-client | 8.10.0 | BSD 3-Clause License - Copyright (c) 2001-2015, IPython Development Team - Copyright (c) 2015-, Jupyter Development Team All right |
| Pillow | 12.3.0 | MIT-CMU |
| setuptools | 84.0.0 | MIT |

Inspect the full installed license files, including bundled native-library notices, before redistributing dependency binaries or building a container. A pip requirement is not a transfer of ownership in a third-party package. This proposal contains imports and independent project code, not copied upstream library source. The original research archive had no license grant at preparation. During finalization the author confirmed copyright ownership/contributor authorization and approved scoped MIT for original code; see [license scope](LICENSE_SCOPE.md). It does not license PhysioNet data, derived data, manuscripts or figures by assumption.

Original scientific source files are copied byte-for-byte and their source hashes are recorded. No automated license scanner establishes that every original line is solely owned by the authors; the author has confirmed provenance/contributor authorization for the scoped MIT grant. Any adapted third-party code later discovered needs its original notices and a compatible release decision. The author/license approvals are explicit; no ethics approval is inferred.
