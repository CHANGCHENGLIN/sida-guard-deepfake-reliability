# Dataset Instructions

Raw videos are not included in this GitHub repository because deepfake datasets are usually large and may have access restrictions.

## Expected Input

The pipeline searches the input directory recursively for video files:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

On Kaggle, the default input path is:

```text
/kaggle/input
```

For local execution, use:

```bash
python src/sida_guard_pipeline.py --input_dir /path/to/dataset --output_dir ./SIDA_Guard
```

## Recommended Dataset Format

The script detects real/fake videos using folder or filename keywords.

Real keywords:

```text
real, original, authentic, youtube-real, celeb-real
```

Fake keywords:

```text
fake, deepfake, synthesis, manipulated, forged, celeb-synthesis
```

If your dataset uses different folder names, edit the keyword lists in `src/sida_guard_pipeline.py`.
