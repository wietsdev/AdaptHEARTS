**HEARTS — Text Stereotype Detection (Dutch adaptation)**

This repository contains work for replicating and extending the EMGSD stereotype-detection experiments into Dutch. The original EMGSD results were replicated, new datasets were merged and translated to Dutch, and experiments were run with a Dutch language model to measure transfer quality and to combine the benefits of quick scaling (machine translation) and contextual accuracy (target-language augmentation).

**Overview**
- **Goal:** Replicate the original EMGSD experiments and evaluate Dutch adaptations using both machine-translated and Dutch-augmented datasets.
- **Key outcomes:** Replication of EMGSD results; new dataset compositions (machine-translated EMGSD subset and augmented Dutch datasets); fine-tuning with a Dutch pre-trained model (RobBERT-v2-Dutch-Base).

**Datasets**
- **Original:** EMGSD (English) — replicated baseline results.
- **Machine-translated EMGSD:** Subset of EMGSD translated EN→NL (used to test quick scaling with MT).
- **GPT-augmented datasets:** Augmented Dutch SeeGULL and augmented Dutch CrowsPairs (used to introduce neutral and unrelated sentences and improve Dutch contextual coverage).
- **Merged compositions used in experiments:**
  - Machine Translation of EMGSD (combined with 20% MGSD + Winoqueer & SeeGULL augmentations)
  - Augmented Dutch SeeGULL + Augmented Dutch CrowsPairs
  - (Dutch) WinoQueer-NL would have been used in order to adapt to Dutch in a way that replicates the English closely as this has been announced in conference, however the actual dataset has not been published yet.

**Model & Method**
- **Original setup:** ALBERT-v2 fine-tuned on EMGSD for stereotype classification (English tokeniser/encoder).
- **This work:** Replace the English-only encoder with `RobBERT-v2-Dutch-Base` to handle Dutch tokenisation and Dutch linguistic phenomena.
  - **Why RobBERT?** RobBERT is a RoBERTa-based language model specifically pre-trained on large Dutch corpora (OSCAR). It better captures Dutch grammar, idioms and sentence structure than a strictly English model or many multilingual models.

**Preprocessing & Filtering**
- Counterfactuals and undesired examples were filtered out before training.
- When using machine translation, a subset selection was applied to avoid class imbalance and overfitting to noisy MT outputs.

**Experiments**
- Fine-tuning was run replacing ALBERT-v2 with `RobBERT-v2-Dutch-Base` and training on the different dataset compositions described above.
- Metrics, splits and hyperparameters mirror the original replication where possible; differences are noted in the experimental logs and scripts.

**Results (summary)**
- Replication: original EMGSD results were reproduced using an equivalent setup.
- Dutch MT: Machine-translated EMGSD gives a fast way to scale to Dutch but may miss subtle Dutch-specific biases and idioms.
- Dutch augmentation: Augmented Dutch SeeGULL and CrowsPairs introduce more contextual variety and improved performance on Dutch-specific examples compared to MT-only data.

**How to run**
- Activate the project virtual environment (example used in development):

```
source aisdCW2/bin/activate
```

- Install dependencies (if needed):

```
pip install -r requirements.txt
```

Run notebooks:
1. DutchAdaptation/DutchSeeGULLAUG2.ipynb
2. DutchAdaptation/mergeDutchSeeGULLCrowSdatasets.ipynb
3. DutchAdaptation/EvalDutchSeeGULLCrows.ipynb

**Repository structure (high-level)**
- **Model Training and Evaluation/** — original training and evaluation scripts (e.g., `BERT_Models_Fine_Tuning.py`).
- **Exploratory Data Analysis/** — EDA scripts used for dataset inspection.
- **DutchAdaptation/** — translated and merged Dutch dataset and notebook files.
- **Replication/** — replication scripts and output directories.

**Files of interest**
- [requirements.txt](requirements.txt) — Python dependencies.
- [Model Training and Evaluation/BERT_Models_Fine_Tuning.py](Model%20Training%20and%20Evaluation/BERT_Models_Fine_Tuning.py) — main fine-tuning script used for experiments.
- [DutchAdaptation](DutchAdaptation) — prepared Dutch datasets and notebooks.

**Notes & Next steps**
- WinoQueer in Dutch dataset presented recently at a conference but not yet published at time of working, this would be great to include in future to replicate EMGSD properly.

**License**
- See `LICENSE` at the repository root for licensing details.

**References & provenance**
1. EMGSD — original dataset (replicated here).
2. SeeGULL / CrowsPairs — used for Dutch augmentations.
3. RobBERT-v2-Dutch-Base — Dutch pre-trained model used to replace ALBERT-v2.


## Replication: baseline evaluation

- **Script:** [Replication/replicate_pretrained.py](Replication/replicate_pretrained.py) evaluates the pre-trained Hugging Face model `holistic-ai/bias_classifier_albertv2` on the assembled test splits (no fine-tuning performed in this script).

- **How to run:**

```bash
source aisdCW2/bin/activate
pip install -r requirements.txt
python Replication/replicate_pretrained.py
```

- **Outputs:** the script saves per-example predictions to `replication_results/EMGSD_Full_Evaluation/full_results.csv` and a machine-readable classification report to `replication_results/EMGSD_Full_Evaluation/classification_report.csv`.

- **Why this approach:** to replicate the paper's evaluation while minimising compute and carbon footprint I intentionally evaluate the released pre-trained model from Hugging Face rather than re-training a model locally (weights available at `holistic-ai/bias_classifier_albertv2`). This preserves reproducibility while reducing energy use.

### Replication results (full merged dataset)

Comparison of the reproduced scores below with the original paper (only Test Set Macro F1 Score presented) in the `Paper Score` column:

| Label / Summary | Precision | Recall | F1-score | Support | Original Paper F1-Score Delta|
|---|---:|---:|---:|---:|---:|
| 0 | 0.8707 | 0.8788 | 0.8747 | 7540 | - |
| 1 | 0.7614 | 0.7478 | 0.7545 | 3901 | - |
| accuracy | 0.8341 | 0.8341 | 0.8341 | 11441 | - |
| macro avg | 0.8161 | 0.8133 | 0.8146 | 11441 | <0.001 |
| weighted avg | 0.8334 | 0.8341 | 0.8337 | 11441 | - |

The paper presents a Test Set Macro F1 Score of 81.5% on the EMGSD Training and EMGSD Test Set. The difference with our replication is <0.1%, so by this metric we have succeeded in replicating the original project.


- **Methodology / Replication:** the script reproduces the baseline evaluation using the released model weights and the assembled test splits; code and outputs are saved so results are reproducible.
- **Reproducibility artifacts:** environment is captured in `requirements.txt`; the script is deterministic where possible (fixed seeds in data splits), and outputs are saved under `replication_results/` for inspection.
- **Dataset & ethics:** input CSV provenance and filter/merge logic are visible in the `Replication/replicate_pretrained.py` loader functions, initial repo is credited and referenced
- **Model performance:** the scores in the table above show that the replication scores the same as the original
- **Energy note:** retraining was deliberately avoided because the baseline weights are publicly available on Hugging Face; this decision was made to consider sustainability in the project.


### Contextual Relevance & SDG Alignment
This project directly addresses **SDG 10 (Reduced Inequalities)** and **SDG 5 (Gender Equality)** within the Dutch linguistic landscape.
* **The Gap:** Most "Safety" filters in Large Language Models (LLMs) are anglocentric. They detect stereotypes common in English language environments but may miss Dutch-specific slurs or biases against groups.
* **Our Contribution:** By fine-tuning **RobBERT**, a native Dutch model, we create a tool capable of flagging these culturally specific biases, ensuring that AI deployment in the Netherlands does not perpetuate local discrimination.

### Scalability and Sustainability
We consciously chose a "Small Language Model" (SLM) approach over prompting massive LLMs.
* **Energy Efficiency:** Fine-tuning RobBERT (~117M parameters) consumes significantly less energy than few-shot prompting a 175B+ parameter model (like GPT-4) for every single classification task.
* **Deployment:** This lightweight model can be deployed on local, CPU-based servers in Dutch institutions (e.g., municipalities or HR departments), democratising access to AI safety tools without reliance on expensive, proprietary APIs.
* **Cross-lingual differences in bias behavior:** Studies show that the same underlying model can exhibit different levels and directions of stereotype bias depending on the language (e.g. more bias in non‑English prompts). [1] So we cannot use one large model capable of multiple languages, test it only in English and expect the performance to hold true for other languages.

### Limitations and Ethical Considerations
* **Augmentation Bias:** A significant portion of our training data was augmented using GPT-4. While cost-effective, this risks creating a "feedback loop" where our model learns the biases inherent in OpenAI's RLHF filters rather than authentic Dutch societal stereotypes.
* **Static vs. Dynamic:** Stereotypes evolve rapidly. A static dataset (like the one created here) risks becoming obsolete as new slurs or coded language emerge. Continuous active learning cycles would be required for a production system.

### Sociocultural and Linguistic Limitations
While our technical adaptation was successful, we acknowledge several critical limitations inherent in cross-lingual stereotype detection:

* **Re-defining “Stereotype” for Dutch Context:** We relied mostly on translating English labels to Dutch. However, stereotypes are deeply cultural; US-centric racial or religious categories do not always map to the Netherlands/Flanders context [1]. A rigorous approach would require working with Dutch cultural experts to define which target groups and attributes are most relevant locally, rather than simply translating English label sets [1]. However, we have attempted to mitigate this by including the Dutch-SeeGULL dataset which contains Dutch context specific stereotypes, this would need to be a much greater effort (with manual review) for a production system. [2]
* **Linguistic and Lexical Asymmetries:** Direct feature transfer is imperfect because Dutch encodes social cues differently (e.g., the gender neutrality of *hun/hen*, the use of diminutives like *-je*, and variable word order) [1]. Bias mitigation techniques designed for English embeddings do not map 1:1 to Dutch grammar [1].
* **Dataset Artifacts:** Our reliance on Machine Translation and LLM augmentation (GPT-4) introduces a risk of "translation artifacts." Without extensive post-editing by native speakers, the data may retain English-style phrasing that is unnatural in Dutch [1]. Furthermore, the definition of "offensive" was not re-calibrated for Dutch norms by native annotators.
* **Domain Specificity:** Our model was fine-tuned on standard text. It may struggle with social media varieties, dialects, or historical Dutch, which often encode stereotypes using slang or creative spelling variations distinct from standard written Dutch.

References: 

[1] Neplenbroek, V., Bisazza, A. and Fernández, R. (2024). MBBQ: A Dataset for Cross-Lingual Comparison of Stereotypes in Generative LLMs. [online] Available at: https://arxiv.org/html/2406.07243v3.

[2] Bhutani, M., Research, G., Robinson, K., Prabhakaran, V., Dave, S. and Dev, S. (2024). SeeGULL Multilingual: a Dataset of Geo-Culturally Situated Stereotypes. [online] 2, pp.842–854. Available at: https://aclanthology.org/2024.acl-short.75.pdf [Accessed 10 Feb. 2026].

[3] [Dutch CrowS-Pairs: Adapting a Challenge Dataset for Measuring Social Biases in Language Models for Dutch](https://aclanthology.org/2025.ranlp-1.138/) (Strazda & Spanakis, RANLP 2025)

## Future Work: Further experimentation on a real-world, "out-of-distribution" Dutch test set is needed to fully validate the model's performance in the wild.

## SDG Alignment
Primary SDGs: Directly supports SDG 5 (Gender Equality) and SDG 10 (Reduced Inequalities) by ensuring AI tools do not perpetuate discrimination against social groups across different linguistic contexts.

Sustainability: Addresses SDG 12 (Responsible Consumption) by analyzing the trade-off between model performance and energy consumption, acknowledging that multilingual adaptation requires additional computational resources.


------------------------------------ Replicated from:
# HEARTS-Text-Stereotype-Detection

[![Paper](https://img.shields.io/badge/ArXiv-2409.11579-red)](https://arxiv.org/abs/2409.11579)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

HEARTS introduces explainable, low-carbon models fine-tuned on the **Expanded Multi-Grain Stereotype Dataset (EMGSD)** to tackle challenges in stereotype detection. This repository includes scripts for training, evaluation, and explainability analysis for sentence-level stereotype classification. For details, refer to the [HEARTS research paper](https://arxiv.org/abs/2409.11579).

---

## References
The State of Multilingual LLM Safety Research: From Measuring the Language Gap to Mitigating It (Yong, 2025)

## Features

- **Exploratory Data Analysis (EDA):** Analyze group distributions, text length, and sentiment/regard trends in EMGSD.
- **Model Training & Evaluation:** Train and test models (e.g., BERT, ALBERT-V2, logistic regression) on EMGSD with ablation studies.
- **Explainability:** Generate SHAP and LIME explanations to interpret predictions.
- **LLM Bias Evaluation:** Classify and evaluate bias in LLM outputs using neutral prompts derived from EMGSD.

---

## Quickstart

1. Clone this repository:
   ```bash
   git clone https://github.com/username/HEARTS-Text-Stereotype-Detection.git
   cd HEARTS-Text-Stereotype-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore the modules (see details below).

---

## Modules

### 1. Exploratory Data Analysis
Scripts to perform basic analysis on EMGSD, available at [Hugging Face](https://huggingface.co/datasets/holistic-ai/EMGSD).

- **`Initial_EDA`**: Analyze target group distribution, stereotype group distribution, text length, and frequency.
- **`Sentiment_Regard_Analysis`**: Classify sentiment ([RoBERTa Sentiment Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)) and regard ([Regard v3](https://huggingface.co/sasha/regardv3)) for dataset entries.

### 2. Model Training and Evaluation
Train and evaluate various models on EMGSD, with ablation studies on its three core datasets (MGSD, Augmented WinoQueer, and Augmented SeeGULL).

- **`BERT_Models_Fine_Tuning`**: Fine-tune and evaluate ALBERT-V2, DistilBERT, and BERT models.
- **`Logistic_Regression`**: Train logistic regression models using:
  - TF-IDF vectorization
  - Pre-trained embeddings ([spaCy embeddings](https://spacy.io/models/en))
- **`DistilRoBERTaBias`**: Evaluate an open-source bias detection model ([DistilRoBERTa Bias](https://huggingface.co/valurank/distilroberta-bias)).
- **`GPT4_Models`**: Evaluate GPT-4o and GPT-4o-mini using API prompting (API credentials required).

### 3. Model Explainability
Interpret model predictions using SHAP and LIME. Weights for the fine-tuned ALBERT-V2 model are available at [Hugging Face](https://huggingface.co/holistic-ai/bias_classifier_albertv2).

- **`SHAP_LIME_Analysis`**: Generate SHAP and LIME explanations for selected model predictions and compare their similarity using metrics such as:
  - Cosine similarity
  - Pearson correlation
  - Jensen-Shannon divergence

### 4. LLM Bias Evaluation
Classify and evaluate bias in LLM responses using neutral prompts derived from EMGSD.

- **`LLM_Prompt_Verification`**: Verify neutrality of prompts using the fine-tuned ALBERT-V2 model.
- **`LLM_Bias_Evaluation`**: Classify LLM outputs to compute aggregate bias scores, representing stereotype prevalence.
- **`SHAP_LIME_Analysis_LLM_Outputs`**: Apply SHAP and LIME to interpret predictions on LLM outputs.

---

## Results

Key findings and performance benchmarks from the paper are outlined [here](https://arxiv.org/abs/2409.11579).


---

## Citation

If you use this work, please cite the following paper:

```
@article{hearts2024,
  title={HEARTS: Enhancing Stereotype Detection with Explainable, Low-Carbon Models},
  author={Author Names},
  journal={arXiv preprint arXiv:2409.11579},
  year={2024}
}
```

---

## License

This repository is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or collaborations, contact [Holistic AI](https://www.holisticai.com).
