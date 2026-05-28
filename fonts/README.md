# fonts/

Place OCR-A and OCR-B TrueType fonts here for text-replacement augmentation
(`packages/trainer/src/meiban_ocr_trainer/data/text_replace.py`).

Recommended sources (Open Font License or public domain):

- OCR-A: <https://fonts.google.com/specimen/Inconsolata> (fallback) or any OFL OCR-A clone
- OCR-B: any OFL OCR-B clone

Expected filenames (referenced by config):

- `fonts/OCR-A.ttf`
- `fonts/OCR-B.ttf`

These files are gitignored; the project does not redistribute them.
