# YOLO Hat Model

Place a custom YOLO model at:

```text
models/hat_detector.pt
```

`python main.py --exam` loads this file only in exam mode. If the file is missing or cannot be loaded, the app falls back to the built-in FaceMesh/ROI hat heuristic.

Supported class names:

- Hat/cap: `cap`, `hat`, `hardhat`, `hard_hat`, `helmet`, `safety_helmet`
- No hat: `no_cap`, `no-cap`, `nocap`, `no_hat`, `no-hat`, `no hardhat`, `no-hardhat`, `no_helmet`, `without_cap`, `without_hat`

Notes:

- Keep model files out of git unless the license explicitly allows redistribution.
- The public `ravee360/Cap-detection` repository includes a YOLOv8 cap model, but no license is shown on the repository page, so it is not vendored here as an open-source dependency.
