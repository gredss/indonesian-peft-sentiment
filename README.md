# PEFT Evaluation for Indonesian Sentiment and Emotion Classification

This repository contains the data preparation and preprocessing pipeline for evaluating **Parameter-Efficient Fine-Tuning (PEFT)** methods on Indonesian NLP tasks using transformer-based models such as **IndoBERT** and **IndoBERTweet**. The primary goal is to assess how PEFT strategies (LoRA, QLoRA, Prefix Tuning, and Full Fine-Tuning) perform in low-resource and domain-specific contexts.

---

## Dataset Overview

**Source:** [PRDECT-ID Dataset (Product Review Dataset for Emotion and Customer Text – Indonesia)](https://doi.org/10.1016/j.dib.2022.108554)

**Size:** 5,400 samples (after deduplication: 5,393)

**Domains:** E-commerce product reviews across multiple categories

**Main Fields:**

* `Category` – product category (e.g., Computers and Laptops)
* `Product Name` – item name
* `Customer Review` – original Indonesian review text
* `Sentiment` – binary label (`Positive` or `Negative`)
* `Emotion` – fine-grained label (`Happy`, `Sadness`, `Fear`, `Love`, `Anger`)
* `Customer Rating` – numeric rating (1–5)
* Additional numeric features: `Price`, `Overall Rating`, `Number Sold`, `Total Review`

---

## Preprocessing and Feature Engineering

### 1. Text Normalization

A normalization function was applied to clean and unify review text:

```python
def normalize_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " <URL> ", s)
    s = re.sub(r"@\w+", " <USER> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
```

This step standardizes casing, replaces links/usernames, and trims redundant spaces.

---

### 2. Stopword Removal

To examine the role of linguistic simplification in low-resource fine-tuning, an Indonesian stopword list was applied:

```python
stopwords = {"dan","yang","di","ke","dari","untuk","pada","dengan","nya","ini","itu",
             "ya","sih","lah","nih","aja","udah","karena","jadi","ada","sebagai","oleh",
             "atau","lebih","kurang","meng","ter","per","se","kan"}
```

A variant without stopword removal was also preserved for **controlled experiments**.

---

### 3. Dataset Variants

| Filename                                 | Description                                                                                                                  |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `product_reviews_with_norm.csv`          | Normalized text with stopwords retained (base version for all PEFT models)                                                   |
| `product_reviews_canonical_text.csv`     | Canonical subset with essential columns (`Category`, `Product Name`, `text_norm`, `Sentiment`, `Emotion`, `Customer Rating`) |
| `product_reviews_no_stop.csv`            | Normalized text with stopwords removed                                                                                       |
| `product_reviews_balanced.csv`           | Full dataset after sentiment class balancing                                                                                 |
| `product_reviews_balanced_canonical.csv` | Canonical balanced variant (stopwords retained)                                                                              |
| `product_reviews_balanced_no_stop.csv`   | Canonical balanced variant (stopwords removed)                                                                               |

---

## Class Balancing

Before balancing:

```
Happy      1768
Sadness    1201
Fear        918
Love        808
Anger       698
```

After random oversampling:

```
Fear       1768
Anger      1768
Happy      1768
Love       1768
Sadness    1768
```

**Rationale:**
Balancing mitigates class bias, ensuring fairer comparison of PEFT methods. Although oversampling may duplicate certain samples, it is appropriate for low-resource experiments where stability of evaluation metrics (especially F1-macro) is prioritized over large-scale generalization.

---

## 📊 Key Observations from EDA

| Feature                        | Observation                                                                                                                                     |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Price**                      | Highly skewed (mean ≈ 238k, max = 15.3M). Consider `log1p(Price)` if used as numeric feature.                                                   |
| **Overall Rating**             | Very narrow range (4.1–5.0), typical positive bias. Limited predictive utility.                                                                 |
| **Number Sold / Total Review** | Broad spread, likely correlated with product popularity.                                                                                        |
| **Customer Rating**            | Bimodal distribution around 1 and 5 — strong sentiment polarity.                                                                                |
| **text_len**                   | Median 12 words, long-tailed up to 184. Max token length for modeling ≈ 64.                                                                     |
| **Emotion**                    | “Happy” dominates (1770 samples), followed by “Sadness” and “Fear.” Emotion-level experiments may require rebalancing or hierarchical modeling. |

---

## Experimental Plan

1. **Fine-tune multiple PEFT strategies** (LoRA, QLoRA, Prefix Tuning, Full Fine-Tuning) using IndoBERT and IndoBERTweet.
2. **Evaluate efficiency metrics:** GPU memory, fine-tuning time, parameter count.
3. **Assess generalization** across multiple datasets (`with_norm`, `no_stop`, `balanced`).
4. **Report evidence-based guidance** on whether text normalization and stopword removal impact PEFT effectiveness in low-resource Indonesian NLP.

---

To be continued..
