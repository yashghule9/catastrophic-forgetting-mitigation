# Data setup — Office-Home

This project uses the [Office-Home dataset](https://www.hemanthdv.org/officeHomeDataset.html)
(Venkateswara et al., 2017): 4 domains (**Art**, **Clipart**, **Product**, **Real World**),
65 object categories, ~15,500 images total. There is no stable auto-download for it (the
official mirror and Kaggle mirrors both require manual download), so setup is manual:

1. Download the dataset zip from the [official page](https://www.hemanthdv.org/officeHomeDataset.html)
   or a Kaggle mirror (e.g. search "Office-Home Dataset" on Kaggle).
2. Extract it so the layout under this `data/` directory looks like:

   ```
   data/raw/OfficeHome/Art/<ClassName>/*.jpg
   data/raw/OfficeHome/Clipart/<ClassName>/*.jpg
   data/raw/OfficeHome/Product/<ClassName>/*.jpg
   data/raw/OfficeHome/Real World/<ClassName>/*.jpg
   ```

   Each domain folder should contain the same 65 class subfolders (see
   `src/cfm/data/office_home.py::CLASSES` for the exact expected class list).

3. Validate the layout:

   ```
   python scripts/validate_data.py --config configs/realworld_to_clipart.yaml
   ```

   This checks that all four domain folders exist, that each contains all 65 expected
   class folders, and warns (without failing) if image counts deviate noticeably from
   the published per-domain counts (Art 2427, Clipart 4365, Product 4439, Real World 4357),
   since mirrors sometimes differ slightly.

`data/raw/` and `data/cache/` are both gitignored — nothing under them is committed.
`data/cache/` is populated automatically by `scripts/extract_features.py` and
`scripts/init_prototypes.py` (cached embeddings, splits, and initial prototypes), so the
GPU-heavy backbone forward pass only ever needs to run once per domain.
