# DermaLens — Data Cleaning & Preprocessing Pipeline

A reproducible, dataset-agnostic data engineering pipeline for preparing heterogeneous dermatological image datasets for the **DermaLens / FairDermNet** research project.

The pipeline is designed to combine and standardize data from multiple sources such as **ISIC, PAD-UFES-20, DDI**, and other compatible skin-lesion datasets while preserving the original information required for downstream research.

> **Current scope:** Data ingestion, auditing, cleaning, deduplication, label harmonization, metadata standardization, leakage analysis, image preprocessing, feature engineering, and normalization.
>
> **Out of scope:** Model architecture, model training, hyperparameter tuning, deployment, and TFLite conversion.

---

## 1. Objectives

The primary objectives of this pipeline are to:

- Build a clean and reproducible dermatological image dataset.
- Detect and remove corrupted or unusable images.
- Identify exact and near-duplicate images.
- Detect potential patient/lesion-level data leakage.
- Harmonize labels from heterogeneous datasets.
- Standardize metadata across different sources.
- Standardize image formats, dimensions, and color channels.
- Perform image-quality analysis.
- Apply appropriate image preprocessing and normalization.
- Extract useful image-level statistical features.
- Analyze class and dataset distributions.
- Preserve dataset provenance throughout the pipeline.
- Produce a model-ready dataset without coupling the pipeline to a specific model architecture.

---

# 2. Pipeline Overview

```text
                         RAW DATASETS
                              │
             ┌────────────────┼────────────────┐
             │                │                │
            ISIC          PAD-UFES-20         DDI
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                     DATA INGESTION
                              │
                              ▼
                       DATASET AUDIT
                              │
                              ▼
                         DATA CLEANING
                              │
                              ▼
                    DUPLICATE DETECTION
                              │
                              ▼
                    LABEL HARMONIZATION
                              │
                              ▼
                  METADATA STANDARDIZATION
                              │
                              ▼
                     LEAKAGE ANALYSIS
                              │
                              ▼
                   IMAGE PREPROCESSING
                              │
                              ▼
                    FEATURE ENGINEERING
                              │
                              ▼
                       NORMALIZATION
                              │
                              ▼
                  DISTRIBUTION ANALYSIS
                              │
                              ▼
                    FINAL DATASET
```

The pipeline is intentionally modular so that each stage can be executed, validated, and modified independently.

---

# 3. Supported Data Sources

The initial pipeline is designed around heterogeneous dermatological datasets.

### ISIC

Dermoscopic skin-lesion imagery with diagnostic labels and associated metadata depending on the specific ISIC release.
Source Link: https://www.kaggle.com/datasets/syedibrarhussain/isic-2019-skin-cancer-detection-dataset?resource=download

About:
This dataset contains dermoscopic images of skin lesions, specifically curated and processed for a multi-class classification task. This work is a derivative of the ISIC 2019 Challenge dataset, structured to support a deep learning pipeline for detecting 9 distinct types of skin conditions.
This dataset has 9 classes of skin cancer

### PAD-UFES-20

Clinical smartphone images with associated patient, lesion, demographic, and diagnostic metadata.
Source Link: https://data.mendeley.com/datasets/zr7vgbcyr2/1?utm_source=chatgpt.com

About:
- The PAD-UFES-20 dataset was collected along with the Dermatological and Surgical Assistance Program (in Portuguese: Programa de Assistência Dermatológica e Cirurgica - PAD) at the Federal University of Espírito Santo (UFES-Brazil), which is a nonprofit program that provides free skin lesion treatment, in particular, to low-income people who cannot afford private treatment.
- The dataset consists of 2,298 samples of six different types of skin lesions. Each sample consists of a clinical image and up to 22 clinical features including the patient's age, skin lesion location, Fitzpatrick skin type, and skin lesion diameter. 
- The skin lesions are: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratosis (ACK), Seborrheic Keratosis (SEK), Bowen’s disease (BOD), Melanoma (MEL), and Nevus (NEV). As the Bowen’s disease is considered SCC in situ, we clustered them together, which results in six skin lesions in the dataset, three skin cancers (BCC, MEL, and SCC) and three skin disease (ACK, NEV, and SEK)
- All BCC, SCC, and MEL are biopsy-proven. The remaining ones may have clinical diagnosis according to a consensus of a group of dermatologists. In total, approximately 58% of the samples in this dataset are biopsy-proven. This information is described in the metadata.
- The images present in the dataset have different sizes because they are collected using different smartphone devices. All images are available in .png format.
- The metadata associated with each skin lesion is composed of up to 26 features. All features are available in a CSV document in which each line represents a skin lesion and each column a metadata feature.
- In total, there are 1,373 patients, 1,641 skin lesions, and 2,298 images present in the dataset. Each image/sample has a reference to the patient and the skin lesion in the metadata.


### DDI

Diverse clinical dermatology images with demographic and skin-type information, particularly useful for evaluating model generalization across different skin tones.
Source Link: https://ddi-dataset.github.io/

About:
Artificial intelligence (AI) may aid in triaging skin diseases. However, most AI models have not been rigorously assessed on images of diverse skin tones or uncommon diseases. To ascertain potential biases in algorithm performance in this context, we curated the Diverse Dermatology Images (DDI) dataset—the first publicly available, deeply curated, and pathologically confirmed image dataset with diverse skin tones. The DDI was retrospectively selected from reviewing pathology reports in Stanford Clinics from 2010-2020 with further details provided in the methods. After filtering out images that were poor quality (see methods), there were a total of 656 images representing 570 unique patients (Supplemental Table 2 and 3). The dataset comprised a retrospective convenience sample across all images of FST I-VI but was also designed to allow direct comparison between FST I-II and FST V-VI by matching diagnostic category, age within 10 years, gender, and date of photograph within 3 years (online methods). There was no significant difference in photo quality scores between FST I-II photos and FST V-VI photos (Mann-Whitney U, p = 0.33). There were a total of 208 images of FST I-II (159 benign, 49 malignant), 241 images of FST III-IV (167 benign, 74 malignant), and 207 images of FST V-VI (159 benign and 48 malignant).

Additional compatible datasets can be added without changing the core pipeline.

---

# 4. Important Design Principle

The preprocessing pipeline **does not enforce a fixed final classification taxonomy**.

Different datasets may contain:

- Different numbers of classes.
- Different diagnostic terminology.
- Different levels of diagnostic specificity.
- Classes that exist in one dataset but not another.

Therefore, the pipeline preserves both:

```text
original_label
```

and

```text
canonical_label
```

rather than immediately discarding source-specific information.

The final model taxonomy will be decided separately during the modeling stage.

---

# 5. Project Structure

```text
data-pipeline/
│
├── data/
│   ├── raw/
│   │   ├── isic/
│   │   ├── pad_ufes/
│   │   └── ddi/
│   │
│   ├── interim/
│   │   ├── cleaned/
│   │   ├── deduplicated/
│   │   └── harmonized/
│   │
│   └── processed/
│       ├── images/
│       ├── metadata.csv
│       └── features.csv
│
├── src/
│   ├── ingestion/
│   │   └── loaders.py
│   │
│   ├── audit/
│   │   ├── dataset_audit.py
│   │   ├── image_quality.py
│   │   └── distribution.py
│   │
│   ├── cleaning/
│   │   ├── corrupt_images.py
│   │   ├── duplicates.py
│   │   └── leakage.py
│   │
│   ├── harmonization/
│   │   ├── labels.py
│   │   └── metadata.py
│   │
│   ├── preprocessing/
│   │   ├── image_standardization.py
│   │   ├── color_normalization.py
│   │   └── artifacts.py
│   │
│   ├── features/
│   │   └── extract_features.py
│   │
│   └── pipeline.py
│
├── configs/
│   └── pipeline.yaml
│
├── reports/
│   ├── audit/
│   ├── quality/
│   ├── duplicates/
│   └── distributions/
│
├── requirements.txt
└── README.md
```

---

# 6. Pipeline Stages

## 6.1 Data Ingestion

Each dataset is loaded through a dataset-specific adapter.

The ingestion stage converts different source formats into a common internal representation.

Example:

```text
Dataset-specific metadata
            │
            ▼
     Standard schema
```

Standard metadata fields may include:

```text
image_id
image_path
dataset
patient_id
lesion_id
original_label
canonical_label
age
sex
skin_type
image_type
width
height
```

Dataset-specific fields that cannot be standardized are retained where possible.

---

# 6.2 Dataset Audit

Before modifying any image, the pipeline generates an audit report.

The audit includes:

- Total image count.
- Number of patients.
- Number of lesions.
- Class distribution.
- Dataset distribution.
- Image dimensions.
- Image formats.
- Missing metadata.
- Corrupt images.
- Duplicate candidates.
- Skin-tone distribution where available.
- Demographic distribution where available.

Example output:

```text
Dataset: PAD-UFES-20

Images:              XXXX
Patients:            XXXX
Unique lesions:      XXXX
Classes:             X
Corrupt images:      X
Duplicate candidates:X
Missing metadata:    X
```

The audit should be performed **before and after cleaning** so the effect of each stage can be measured.

---

# 6.3 Corrupt and Invalid Image Detection

Each image is checked to determine whether it can be decoded correctly.

The pipeline identifies:

- Corrupted files.
- Empty files.
- Unsupported formats.
- Invalid image headers.
- Zero-dimension images.
- Extremely small or unusable images.

Invalid files are **logged rather than silently deleted**.

Example:

```text
image_id    status       reason
IMG001      VALID        -
IMG002      INVALID      corrupted JPEG
IMG003      INVALID      zero dimensions
```

---

# 6.4 Duplicate Detection

Duplicate detection is critical for preventing data leakage.

The pipeline should detect:

### Exact duplicates

Using file-level hashing.

```text
SHA-256 / MD5
```

### Visual duplicates

Using perceptual hashing.

```text
pHash
dHash
aHash
```

### Near duplicates

Using image similarity methods where required.

Potential duplicates are logged for review rather than automatically assuming that every visually similar image is the same lesion.

---

# 6.5 Label Harmonization

Different datasets may use different terminology for similar diagnoses.

The pipeline maintains a mapping:

```text
dataset
original_label
canonical_label
mapping_status
```

Example:

```text
Dataset       Original Label       Canonical Label
---------------------------------------------------
ISIC          melanoma             MEL
PAD-UFES      melanoma             MEL
ISIC          nevus                NV
PAD-UFES      nevus                NV
```

Label mappings must be **explicit and documented**.

The pipeline must not merge diagnoses simply because they appear visually similar or have similar names.

Ambiguous labels should be marked for review.

---

# 6.6 Metadata Standardization

Metadata fields are normalized into a common representation.

Examples:

```text
Male / male / M
        ↓
Male
```

```text
45 / "45" / 45.0
        ↓
45
```

Metadata types should also be standardized:

```text
age        → numeric
sex        → categorical
skin_type  → categorical
dataset    → categorical
label      → categorical
```

Unknown values should use a consistent representation rather than multiple variations such as:

```text
N/A
NA
Unknown
?
null
```

---

# 6.7 Patient and Lesion Leakage Analysis

Where patient or lesion identifiers are available, the pipeline checks for multiple images belonging to the same entity.

The goal is to prevent situations such as:

```text
Patient A / Lesion X
        │
        ├── training
        └── validation
```

from occurring later.

The pipeline therefore preserves:

```text
patient_id
lesion_id
image_id
```

throughout processing.

This information will later allow patient/lesion-level dataset splitting.

---

# 6.8 Image Standardization

Images are standardized into a common format.

Typical operations include:

```text
Image
  ↓
Decode
  ↓
RGB conversion
  ↓
Resize
  ↓
Standard format
```

The pipeline should record the original dimensions before resizing.

Example:

```text
original_width
original_height
processed_width
processed_height
```

The target resolution should remain configurable rather than hard-coded into every processing function.

---

# 6.9 Image Quality Analysis

The pipeline calculates image-quality indicators such as:

### Sharpness

Blur detection using methods such as the Laplacian variance.

### Brightness

Mean image intensity.

### Contrast

Intensity dispersion.

### Exposure

Identification of severely underexposed or overexposed images.

These measurements should initially be used for **analysis and flagging**, not automatic deletion.

Example:

```text
image_id | sharpness | brightness | contrast | quality_flag
```

---

# 6.10 Artifact Handling

Depending on the dataset, images may contain:

- Hair.
- Rulers.
- Markers.
- Reflections.
- Borders.
- Acquisition artifacts.

Artifact removal should be treated as an **optional processing module**.

It should not be applied blindly to every dataset because aggressive artifact removal can alter clinically relevant image information.

---

# 6.11 Color and Illumination Processing

Clinical images may differ significantly in:

- Lighting.
- Camera characteristics.
- White balance.
- Color temperature.
- Exposure.

The pipeline may therefore support:

- Pixel normalization.
- Color-space conversion.
- Color constancy.
- Illumination normalization.

Each transformation should be configurable and evaluated experimentally.

---

# 6.12 Feature Engineering

Feature engineering focuses on **dataset analysis and image characterization**, not replacing the raw image.

Potential features include:

```text
width
height
aspect_ratio

brightness
contrast
sharpness

mean_R
mean_G
mean_B

std_R
std_G
std_B

saturation
```

If lesion segmentation is introduced later, additional features may include:

```text
lesion_area
lesion_perimeter
lesion_aspect_ratio
lesion_area_ratio
```

These features can be stored separately:

```text
features.csv
```

The original processed image remains available.

---

# 6.13 Normalization

Normalization occurs at multiple levels.

### Image normalization

Pixel values can be transformed from:

```text
0–255
```

to:

```text
0–1
```

Additional model-specific normalization should be applied later according to the selected model.

### Metadata normalization

Categorical and numerical metadata are standardized independently.

The preprocessing pipeline should not permanently encode assumptions belonging to a future model.

---

# 6.14 Distribution Analysis

After preprocessing, generate distribution reports.

Analyze:

```text
Class distribution
Dataset distribution
Image-source distribution
Skin-type distribution
Age distribution
Sex distribution
Image-quality distribution
```

This is particularly important because combining datasets can introduce **dataset imbalance and domain imbalance**.

Example:

```text
             Dataset A    Dataset B    Dataset C

MEL             █████       ██           █
BCC             ███         ██           █
NV              ███████     ███          ██
DF              █            -           -
```

No artificial balancing is performed at this stage unless explicitly configured.

---

# 7. Output Dataset

The final output should contain:

```text
data/processed/
│
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
├── metadata.csv
│
└── features.csv
```

### `metadata.csv`

```text
image_id
image_path
dataset
patient_id
lesion_id
original_label
canonical_label
age
sex
skin_type
image_type
quality_flag
```

### `features.csv`

```text
image_id
width
height
aspect_ratio
brightness
contrast
sharpness
mean_R
mean_G
mean_B
std_R
std_G
std_B
```

---

# 8. Reproducibility

All configurable preprocessing parameters should be stored in:

```text
configs/pipeline.yaml
```

Example:

```yaml
image:
  target_size: [224, 224]
  color_mode: RGB

quality:
  enable_blur_check: true
  enable_exposure_check: true

duplicates:
  exact_hash: true
  perceptual_hash: true

features:
  color_statistics: true
  sharpness: true
  brightness: true
  contrast: true
```

The pipeline should produce the same processed dataset when run with the same:

- Input data.
- Configuration.
- Code version.
- Random seed where applicable.

---

# 9. Libraries

Core dependencies:

```text
Python
├── pandas
├── numpy
├── Pillow
├── OpenCV
├── scikit-image
├── scikit-learn
├── imagehash
└── matplotlib
```

The pipeline should avoid unnecessary dependencies and keep GPU usage to a minimum.

Most preprocessing operations are CPU-based.

---

# 10. Execution

The complete pipeline should eventually be executable through a single entry point:

```bash
python src/pipeline.py
```

Individual stages should also be independently executable for debugging:

```bash
python src/audit/dataset_audit.py
python src/cleaning/corrupt_images.py
python src/cleaning/duplicates.py
python src/harmonization/labels.py
python src/features/extract_features.py
```

---

# 11. Validation Checklist

Before considering the processed dataset ready for modeling:

- [ ] All datasets successfully ingested.
- [ ] Dataset statistics generated.
- [ ] Corrupt images identified.
- [ ] Duplicate candidates identified.
- [ ] Labels harmonized and documented.
- [ ] Patient/lesion identifiers preserved.
- [ ] Potential leakage identified.
- [ ] Image formats standardized.
- [ ] Image dimensions standardized.
- [ ] Image-quality statistics generated.
- [ ] Feature extraction completed.
- [ ] Normalization verified.
- [ ] Class distributions documented.
- [ ] Dataset distributions documented.
- [ ] Skin-tone distributions documented where available.
- [ ] Processing logs generated.
- [ ] Raw data remains untouched.
- [ ] Processed dataset can be reproduced from configuration.

---

# 12. Design Principles

### Preserve raw data

Never modify the original dataset files.

### Make transformations traceable

Every removed or transformed sample should have a reason recorded in logs.

### Preserve provenance

Every image should retain its source dataset.

### Don't force labels

Only harmonize labels when there is a defensible mapping.

### Don't overprocess

Every image transformation should have a measurable purpose.

### Separate preprocessing from training

Permanent preprocessing and training-time augmentation should remain separate.

### Keep the pipeline model-agnostic

The data pipeline should not depend on a specific CNN, transformer, classifier, or deployment framework.

---

# 13. Final Pipeline

```text
RAW DATASETS
     │
     ▼
┌───────────────────┐
│ Dataset Ingestion │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Dataset Audit     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Data Cleaning     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Duplicate Check   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Label             │
│ Harmonization     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Metadata          │
│ Standardization   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Leakage Analysis  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Image             │
│ Preprocessing     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Feature           │
│ Engineering       │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Normalization     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Distribution &    │
│ Quality Reports   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ CLEAN PROCESSED   │
│ DATASET           │
└───────────────────┘
          │
          ▼
     MODELING STAGE
     (separate project)
```

---

## 14. Future Scope

The processed dataset generated by this pipeline can later be used for:

- Multi-source model development.
- Class-imbalance experiments.
- Fairness-aware training.
- Explainability experiments.
- Model comparison.
- External validation.
- Model compression and deployment.

The modeling layer should consume the processed dataset **without modifying the underlying data-cleaning logic**.