# SIDA-Guard

**SIDA-Guard: A No-Training Quality-Aware Reliability Layer for Deepfake Detection**

SIDA-Guard is a course final project that does **not** train a new deepfake detector. Instead, it analyzes whether visual quality degradation may make a deepfake detector's prediction unreliable.

The project generates controlled visual perturbations, extracts whole-frame and fixed face ROI quality metrics, and converts these metrics into an interpretable reliability risk score.

---

## 1. Project Idea

Most deepfake detectors output a binary prediction such as `real` or `fake`. However, real-world videos may include quality problems such as:

- lighting flicker
- motion blur
- ghosting / temporal artifacts
- background degradation
- unstable camera motion

These problems may reduce detector reliability. Therefore, this project asks a different question:

> The problem is not only **"Is it fake?"**, but also **"Can we trust the prediction?"**

SIDA-Guard is designed as a **quality-aware reliability layer** that can be used together with an existing deepfake detector.

---

## 2. Repository Structure

```text
SIDA-Guard/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── sida_guard_pipeline.py          # Main reproducible pipeline
├── report/
│   ├── final_report.md                 # Report content in Markdown
│   ├── ieee/
│   │   ├── main.tex                    # IEEE/Overleaf LaTeX report source
│   │   └── references.bib              # BibTeX references
│   └── figures/                        # Selected experiment figures
├── results/
│   ├── *.csv                           # Example CSV outputs
│   └── examples/                       # Selected preview images and charts
├── data/
│   └── README.md                       # Dataset instructions; raw data not included
├── docs/
│   ├── final_submission_checklist.md
│   └── report_guidelines_summary.md
└── notebooks/
    └── README.md                       # Kaggle/Colab notebook instructions
```

---

## 3. What is `main.tex`?

`main.tex` is the **LaTeX source file** for the final report.

The final report needs to follow a research-paper style format such as IEEE, Elsevier, or Springer. `main.tex` can be uploaded to **Overleaf** to generate the final report PDF in IEEE style.

In simple terms:

- `README.md` explains the project and how to run the code.
- `src/sida_guard_pipeline.py` contains the main source code.
- `report/final_report.md` is the report text in an easy-to-read Markdown format.
- `report/ieee/main.tex` is the formal IEEE report source for Overleaf.
- `report/ieee/references.bib` stores academic references.

---

## 4. Dataset Information

This project was tested using sampled real and fake videos from a deepfake dataset available in Kaggle input storage.

The script searches `/kaggle/input` recursively for video files. It detects real/fake files by path keywords such as `real`, `original`, `fake`, `synthesis`, or `deepfake`.

Raw datasets are **not included** in this repository because they may be large and may have their own licenses or access rules.

Recommended datasets:

- Celeb-DF v2
- FaceForensics++
- DFDC

---

## 5. Installation

### Option A: Kaggle Notebook

1. Create a Kaggle notebook.
2. Add your deepfake video dataset using **Add Data**.
3. Upload this repository or copy `src/sida_guard_pipeline.py` into the notebook.
4. Run:

```bash
python src/sida_guard_pipeline.py
```

### Option B: Local Environment

```bash
git clone <YOUR_GITHUB_REPOSITORY_LINK>
cd SIDA-Guard
pip install -r requirements.txt
python src/sida_guard_pipeline.py --input_dir /path/to/your/dataset --output_dir ./SIDA_Guard
```

---

## 6. Main Pipeline

The main script performs the following steps:

1. Create project folders
2. Sample real and fake videos
3. Generate controlled perturbation videos
4. Save perturbation metadata
5. Compute whole-frame quality metrics
6. Compute fixed face ROI metrics
7. Compute SIDA-Guard quality risk score
8. Save CSV outputs and figures
9. Generate a final status report

---

## 7. Perturbation Conditions

Each sampled video is converted into the following conditions:

| Condition | Description |
|---|---|
| `clean` | Original clean video |
| `light_flicker_whole` | Lighting flicker on the whole frame |
| `motion_blur_whole` | Motion blur on the whole frame |
| `ghosting_whole` | Frame ghosting / residual effect |
| `mixed_whole` | Flicker + motion blur + ghosting on the whole frame |
| `light_flicker_face` | Lighting flicker only on the face region |
| `motion_blur_face` | Motion blur only on the face region |
| `light_flicker_background` | Lighting flicker only on the background |
| `motion_blur_background` | Motion blur only on the background |

---

## 8. Output Files

Important output files include:

```text
SIDA_Guard/results/sampled_videos_metadata.csv
SIDA_Guard/results/processed_videos_metadata.csv
SIDA_Guard/results/perturbation_metadata.csv
SIDA_Guard/results/quality_metrics.csv
SIDA_Guard/results/quality_metrics_summary_by_condition.csv
SIDA_Guard/results/fixed_face_tracks.csv
SIDA_Guard/results/fixed_face_roi_quality_metrics.csv
SIDA_Guard/results/quality_metrics_with_fixed_face_roi.csv
SIDA_Guard/results/fixed_face_roi_quality_summary_by_condition.csv
SIDA_Guard/results/sida_guard_quality_only.csv
SIDA_Guard/results/sida_guard_quality_summary_by_condition.csv
```

Important figures include:

```text
SIDA_Guard/figures/whole_frame_flicker_score_by_condition.png
SIDA_Guard/figures/whole_frame_blur_score_by_condition.png
SIDA_Guard/figures/fixed_face_flicker_score_by_condition.png
SIDA_Guard/figures/fixed_face_blur_score_by_condition.png
SIDA_Guard/figures/sida_guard_quality_risk_by_condition.png
```

---

## 9. Main Result Summary

In the MVP experiment, `mixed_whole` produced the highest final quality risk because it combines flicker, motion blur, and ghosting. Clean videos produced the lowest risk.

Example results:

| Condition | Risk | Reliability |
|---|---:|---:|
| mixed_whole | 0.618 | 0.382 |
| light_flicker_whole | 0.484 | 0.516 |
| motion_blur_whole | 0.300 | 0.700 |
| clean | 0.000 | 1.000 |

---

## 10. How to Prepare Final Submission

1. Upload this repository to GitHub.
2. Add your generated report PDF.
3. Add your final presentation slides.
4. Add the GitHub link to your report in the Data Availability Statement.
5. Submit the report PDF, slides, source code, and GitHub repository link to the course portal.

---

## 11. GitHub Repository Link

Replace this line after creating your repository:

```text
GitHub repository: <PASTE YOUR GITHUB LINK HERE>
```
