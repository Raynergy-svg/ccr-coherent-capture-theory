# Reproduce the private CCT baseline check

Python 3.12.14. Dependencies are pinned in requirements-reanalysis.txt. The files under repo/ are read-only copies from public commit ff273145c6edf855a810992d6f8afbdf7916a7de. Root-era inputs come from 83f0468ca579c4e9baf690cc7fa94f289d7c3dda. No network is needed for these calculations.

Run from this directory:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reanalysis.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reanalysis.py strict_probability
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reanalysis.py strict_probability_fold_preserved
    OPENBLAS_NUM_THREADS=1 python verify_and_describe.py

The no-argument run is the preserved original algorithm, including its documented PM_PROB sentinel error. Use the corrected and fold-preserving results for the cautiously framed final inference. All original and corrective runs are shown, not hidden. Existing result files will be recomputed in place when the commands are run. TEST_SPEC.md and VALIDITY_ADDENDUM.md record the sequence. The version-2 report corrects interpretation and terminology following an independent check; source calculations did not change.

The outcome is [C/Fe]-[O/Fe]=[C/O], solar-relative in dex. Variable names log_CO are shorthand; an absolute log10(C/O) differs by a constant and gives the same prediction-error statistics. The selection is an analyst-chosen screen, not the published OCCAM membership rule and not posterior membership certainty.
