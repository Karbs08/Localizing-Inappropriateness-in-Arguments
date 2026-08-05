# From Document to Span: Localizing Inappropriateness in Arguments

This repository contains the experimental code for localizing the textual evidence behind document-level predictions of **inappropriateness in arguments**.

The project starts from a pretrained binary appropriateness classifier that predicts whether an entire argument is appropriate or inappropriate. Its central question is more fine-grained:

> **Which words or text spans caused, supported, or best explain the classifier's prediction that an argument is inappropriate?**

Because the underlying corpus provides document-level labels but no human-annotated rationale spans, the project studies span localization under **weak supervision**. It compares post-hoc attribution methods, lexical and random baselines, and a Multiple Instance Learning approach. The extracted spans are evaluated through classifier perturbations, comparison with LLM-generated silver-reference spans, and a planned human evaluation.

> **Project status:** Work in progress. The core localization methods are implemented. Some cross-method evaluation results, the final human study, and the consolidated result reporting are still being completed.

---

## Contents

- [Research context](#research-context)
- [Dataset and classifier](#dataset-and-classifier)
- [Task formulation](#task-formulation)
- [Localization methods](#localization-methods)
- [Experimental workflow](#experimental-workflow)
- [Evaluation](#evaluation)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Google Gemini API setup](#google-gemini-api-setup)
- [Data preparation](#data-preparation)
- [Running the notebooks](#running-the-notebooks)
- [Reproducibility](#reproducibility)
- [Current status](#current-status)
- [Limitations](#limitations)
- [Citation](#citation)
- [References](#references)

---

## Research context

This work builds directly on:

> Timon Ziegenbein, Shahbaz Syed, Felix Lange, Martin Potthast, and Henning Wachsmuth.  
> **[Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238/).**  
> ACL 2023, pages 4344–4363.

Ziegenbein et al. operationalize appropriateness in online argumentation through a hierarchical taxonomy of inappropriate language. Their taxonomy contains four core dimensions:

- **Toxic Emotions**
- **Missing Commitment**
- **Missing Intelligibility**
- **Other Reasons**

These are further decomposed into more specific dimensions such as excessive intensity, emotional deception, missing seriousness, missing openness, unclear meaning, missing relevance, confusing reasoning, and detrimental orthography.

The original work establishes and predicts appropriateness at the **argument level**. This repository takes the next step from document-level assessment to **span-level localization**. Rather than only asking whether an argument is inappropriate, it investigates which parts of the argument contain evidence relevant to that prediction.

This repository does not retrain or replace the original binary classifier as its primary task. Instead, the classifier is treated as a fixed document-level model whose predictions are analyzed and localized.

---

## Dataset and classifier

### Appropriateness Corpus

The experiments use the official Hugging Face version of the **Appropriateness Corpus**:

- **Dataset:** [timonziegenbein/appropriateness-corpus](https://huggingface.co/datasets/timonziegenbein/appropriateness-corpus)
- **Original paper:** [Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238/)
- **Original project repository:** [timonziegenbein/appropriateness-corpus](https://github.com/timonziegenbein/appropriateness-corpus)

The corpus contains **2,191 English arguments** on **1,154 issues**, compiled from three argument-quality corpora and three argumentative genres:

- 1,590 arguments from debate portals
- 500 arguments from question-answering forums
- 101 reviews

The Hugging Face dataset provides the following predefined splits:

| Split | Arguments |
|---|---:|
| Train | 1,533 |
| Validation | 220 |
| Test | 438 |
| **Total** | **2,191** |

Relevant columns include:

| Column | Description |
|---|---|
| `post_id` | Identifier of the argument |
| `issue` | Issue or topic discussed by the argument |
| `post_text` | Full argument text |
| `Inappropriateness` | Binary document-level inappropriateness label |
| taxonomy columns | Binary labels for the finer-grained inappropriateness dimensions |

The corpus contains document-level annotations. It does **not** contain gold character offsets or token-level rationale annotations for the inappropriate parts of an argument. This absence of gold spans motivates the weakly supervised setup used in this project.

### Binary appropriateness classifier

The experiments use the binary classifier released with the corpus:

- **Model:** [timonziegenbein/appropriateness-classifier-binary](https://huggingface.co/timonziegenbein/appropriateness-classifier-binary)

The Hugging Face model card describes the model as a DeBERTa-v2-based text classifier that distinguishes:

- `LABEL_0`: appropriate
- `LABEL_1`: inappropriate

In this project, the probability assigned to `LABEL_1` is denoted by:

$$
p_{\mathrm{inappropriate}}(x)
=
P(y=\mathrm{inappropriate}\mid x)
$$

All predictions of the original document-level classifier use the argument text, `post_text`, as input. The issue is retained as metadata and may be used by separate experimental components, such as the MIL span encoder, but it is not added to the input of the original binary classifier.

The model is used for two purposes:

1. establishing the original document-level prediction and its probability;
2. evaluating extracted spans by masking or deleting them and running the classifier again.

---

## Task formulation

Let an argument be represented by $x$, and let the fixed document-level classifier be $f$. The classifier returns the probability of the inappropriate class:

$$
f_1(x)
=
P(y=1\mid x)
$$

The span-localization task is to identify one or more spans:

$$
S = \{s_1, s_2, \ldots, s_k\}
$$

that capture evidence relevant to the model's prediction that $x$ is inappropriate.

The task differs from conventional supervised rationale extraction because there are no gold rationale spans available for training. The methods instead use one or more of the following signals:

- internal classifier representations;
- feature-attribution scores;
- lexical salience;
- changes in classifier output after perturbation;
- document-level labels;
- LLM-generated silver-reference spans.

The extracted spans should ideally be:

- **relevant**, by pointing to content connected to inappropriateness;
- **faithful**, by affecting the classifier when perturbed;
- **concise**, by avoiding unnecessary surrounding text;
- **sufficient**, by retaining enough evidence to understand the prediction;
- **interpretable**, by forming readable text spans rather than isolated tokenizer artifacts.

---

## Localization methods

The repository compares six primary localization approaches. LLM-generated spans are handled separately as a silver reference and are not treated as human gold annotations.

### Method overview

| Method | Category | Main localization signal | Requires additional training |
|---|---|---|---:|
| Random spans | Baseline | Randomly sampled words or contiguous spans | No |
| TF-IDF | Lexical baseline | Argument-specific TF-IDF scores | No |
| SHAP | Post-hoc attribution | Positive SHAP values for `LABEL_1` | No |
| Integrated Gradients | Gradient-based attribution | Positive input attributions for `LABEL_1` | No |
| Attention | Internal model signal | Classification-token attention or attention rollout | No |
| Multiple Instance Learning | Weakly supervised learning | Learned latent span-instance scores | Yes |
| LLM spans | **Silver reference** | LLM-generated textual rationales | External annotation only |

### Random span baseline

The random baseline does not use model attributions, gradients, attention weights, labels, or lexical importance. It selects spans randomly and measures how much perturbing them changes the inappropriate-class probability.

Two selection strategies are investigated:

- `single_contiguous`: select one contiguous random word span;
- `multi_word_budgeted`: sample a fixed word budget and merge adjacent selected words into spans.

During configuration search, several random samples are drawn for each argument and configuration. Their means and standard deviations estimate the variability of random selection. For the final test evaluation, exactly one concrete random sample is retained per argument. This avoids selecting the best random sample after observing the result and ensures that each row corresponds to a unique set of spans.

The random baseline is essential for determining whether an attribution method performs better than perturbing arbitrary parts of an argument.

### TF-IDF baseline

The TF-IDF baseline provides a simple lexical notion of salience that is independent of the classifier's internal representations.

A `TfidfVectorizer` is fitted on the development data only. For each argument, the highest-scoring terms are selected, located in the original text, and expanded or merged into readable spans. The main configuration parameters are:

- the number of selected terms, `top_k`;
- the local `window_size` used to expand or merge evidence.

The final vectorizer is fitted on the development set and only transformed on the held-out test set. The test data is not used to learn the TF-IDF vocabulary or select the configuration.

TF-IDF is not expected to provide a faithful explanation of the classifier by itself. It serves as a transparent lexical baseline for assessing whether model-aware methods outperform generic term salience.

### SHAP

The SHAP approach applies local feature attribution to the binary classifier using the SHAP Transformers integration.

Reference:

- Scott M. Lundberg and Su-In Lee. [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). NeurIPS 2017.

SHAP produces a signed contribution score for each text segment. This project focuses on **positive attributions for `LABEL_1`**, because the objective is to identify evidence that supports the inappropriate prediction rather than evidence that opposes it.

The extraction procedure is:

1. compute SHAP values for the inappropriate-class output;
2. retain positively contributing segments;
3. select the strongest segments using a quantile threshold;
4. optionally merge nearby segments using a window parameter;
5. convert the selected segments to character-level spans;
6. perturb all selected spans together;
7. measure the change in the classifier output.

The main tuned parameters are:

- `quantile`;
- `window_size`.

Higher quantiles generally produce shorter, more selective explanations. Lower quantiles retain more evidence but may include irrelevant content.

### Integrated Gradients

Integrated Gradients is implemented with Captum's `LayerIntegratedGradients`.

Reference:

- Mukund Sundararajan, Ankur Taly, and Qiqi Yan. [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html). ICML 2017.

The method computes gradients for the inappropriate target class along a path from a reference input to the observed input. Subword-level attributions are aggregated to words before spans are constructed.

The extraction procedure is:

1. compute token-level Integrated Gradients for `LABEL_1`;
2. aggregate subword contributions to word-level scores;
3. retain positive word attributions;
4. select high-attribution words using a quantile;
5. merge nearby selected words using a configurable gap;
6. evaluate the resulting spans through perturbation.

The main tuned parameters are:

- `quantile`;
- `window_size`.

Because attribution scores are aggregated to complete words before span creation, the resulting explanations are less prone to isolated subword fragments.

### Attention-based localization

The attention approach derives candidate spans from internal attention distributions of the classifier.

Two aggregation strategies are investigated:

- `last_cls`: attention from the classification token to input tokens in the final transformer layer, averaged across heads;
- `rollout`: attention propagated across layers using normalized attention matrices and residual connections.

References:

- Sarthak Jain and Byron C. Wallace. [Attention is not Explanation](https://aclanthology.org/N19-1357/). NAACL 2019.
- Samira Abnar and Willem Zuidema. [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/). ACL 2020.

Attention weights are **not assumed to be faithful explanations by themselves**. They are used only as a signal for generating candidate spans. The selected spans are subsequently evaluated by perturbing the input and observing the classifier response.

The workflow includes:

1. extract token-level attention scores;
2. select high-attention tokens by quantile;
3. merge nearby selected tokens;
4. expand spans to readable word boundaries;
5. remove punctuation-only artifacts;
6. validate the spans through classifier perturbation.

This post-processing is important because raw tokenizer offsets may otherwise produce fragments such as punctuation marks or incomplete subwords.

### Multiple Instance Learning

Multiple Instance Learning is the only primary method in this repository that trains an additional model specifically for span localization.

Each argument is represented as a **bag**, while automatically generated candidate spans are treated as latent **instances**:

$$
B_i
=
\{s_{i1}, s_{i2}, \ldots, s_{im}\}
$$

Only the argument-level label is observed. The model learns instance scores and aggregates them into a bag-level prediction.

Candidate spans are generated as sliding word windows. Their granularity and density are controlled through:

- candidate span length;
- stride;
- maximum number of candidates.

The MIL encoder receives the discussion issue and a candidate span. Including the issue gives the span model access to context that may be required for dimensions such as missing relevance.

Several pooling strategies are explored:

- max pooling;
- top-$k$ mean pooling;
- top-$k$ noisy-or pooling.

After training, the highest-ranked candidate spans are selected and evaluated against the **original Ziegenbein classifier** using the same perturbation protocol as the post-hoc methods.

MIL therefore has two distinct evaluation levels:

1. **bag-level classification quality**, which tests whether the MIL model learns the document-level task;
2. **span-level perturbation behavior**, which tests whether its selected spans affect the original classifier.

Good bag-level performance does not automatically imply that the localized spans are correct. Bag-level metrics are therefore treated as guardrails, while the selected spans require separate evaluation.

The repository contains both multi-scale candidate-span experiments and refined experiments in which each configuration uses a single candidate span length.

### LLM-generated spans as a silver reference

LLM-generated spans are **not gold-standard explanations**, are not treated as human annotations, and are not counted as one of the primary localization methods.

The silver-reference annotations are generated with Google Gemini using:

```python
LLM_NAME = "gemini-3-flash-preview"
```

Gemini is prompted to identify minimal verbatim spans that explain why an argument is inappropriate. The returned span text and character offsets are validated against the original argument. Invalid offsets are either repaired through exact text matching or marked as invalid. The resulting annotations provide a **silver reference** for comparing the spans produced by the six primary localization approaches.

The LLM reference is used to calculate overlap-oriented metrics such as:

- precision;
- recall;
- F1;
- intersection over union;
- overlap hit rate;
- selected-rank statistics.

The LLM spans have several important limitations:

- they may reflect the LLM's semantic judgment rather than the classifier's decision process;
- they may omit valid evidence or include plausible but unnecessary context;
- they can be affected by prompting, model version, and decoding behavior;
- agreement with them does not prove classifier faithfulness;
- disagreement with them does not necessarily imply that a method is wrong.

For these reasons, the LLM annotations are consistently described as a **silver reference**, never as span-level ground truth. The planned human study provides an independent evaluation of the semantic quality of the extracted spans.

---

## Experimental workflow

### 1. Prepare one canonical dataset

The dataset is downloaded once and converted into a shared processed representation. The preparation step:

- downloads or loads the official Hugging Face dataset;
- stores a local raw copy;
- preserves the official train, validation, and test splits;
- applies minimal Unicode and whitespace normalization;
- assigns a stable `global_row_id`;
- obtains the original binary classifier predictions;
- stores `p_inappropriate_original`;
- derives predicted labels and confidence scores;
- assigns confusion types;
- saves the processed data as Parquet;
- records preprocessing metadata for reproducibility.

All experiment notebooks load this same prepared dataset. This prevents accidental differences in normalization, IDs, classifier predictions, or split handling across methods.

### 2. Record classifier confusion types

Each argument is categorized relative to the corpus label and the original classifier prediction:

| Type | Gold label | Classifier prediction |
|---|---:|---:|
| TP | Inappropriate | Inappropriate |
| FN | Inappropriate | Appropriate |
| FP | Appropriate | Inappropriate |
| TN | Appropriate | Appropriate |

Configuration selection focuses primarily on **true positives**. For a true positive, both the human document-level label and the classifier agree that the argument is inappropriate, making the localization of positive inappropriate evidence most directly interpretable.

The other confusion types are retained for complementary analyses of model behavior and robustness.

### 3. Select method configurations on development data

Method-specific hyperparameters are selected using development data rather than the held-out test set. Depending on the experiment notebook, the development pool consists of validation true positives or the combined train-and-validation true positives.

The final standardized evaluation follows the same principle:

- configuration search only on development data;
- no configuration selection on test examples;
- one fixed configuration per method for final test evaluation.

The configuration choice balances several criteria rather than maximizing probability drop alone:

- mean probability drop;
- median probability drop;
- positive drop rate;
- masked token or word ratio;
- explanation length;
- stability across examples;
- qualitative readability.

### 4. Apply the selected configuration to test data

After configuration selection, the chosen setup is fixed and applied to the held-out test split.

Each method exports:

- argument-level results;
- span-level results;
- configuration summaries;
- split-level summaries;
- confusion-type summaries;
- serialized span metadata;
- selected plots and qualitative examples.

### 5. Compare methods under a shared protocol

Although the methods generate spans differently, they are evaluated through a common interface:

1. extract one or more spans;
2. retain their original character offsets;
3. mask or delete the selected text;
4. rerun the same document-level classifier;
5. measure the probability change;
6. normalize by the amount of perturbed text;
7. optionally compare the spans with the LLM silver reference;
8. evaluate semantic quality in the human study.

---

## Evaluation

No single metric can establish that a span is simultaneously faithful, semantically correct, concise, and complete. The project therefore combines complementary evaluation perspectives.

### Perturbation-based faithfulness

For an argument $x$ and selected spans $S$, let $x^{\mathrm{abl}(S)}$ denote the argument after masking or deleting those spans.

The probability drop is:

$$
\Delta p(x,S)
=
p_{\mathrm{inappropriate}}(x)
-
p_{\mathrm{inappropriate}}\left(x^{\mathrm{abl}(S)}\right)
$$

Interpretation:

- $\Delta p > 0$: perturbing the selected spans lowers the inappropriate probability;
- $\Delta p = 0$: the perturbation has no measured effect;
- $\Delta p < 0$: perturbing the selected spans increases the inappropriate probability.

A larger positive drop suggests that the selected spans contain evidence used by the classifier. However, probability drop must be interpreted together with explanation size. Masking half of an argument will often have a stronger effect than masking one precise phrase.

The positive drop rate is:

$$
\mathrm{PDR}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}\left[\Delta p_i > 0\right]
$$

The masked-token ratio is:

$$
r_i
=
\frac{\text{number of masked model tokens in }x_i}
{\text{number of model tokens in }x_i}
$$

Reported perturbation metrics include:

- mean and median probability drop;
- standard deviation of probability drop;
- positive drop rate;
- stronger-drop rates for selected thresholds;
- mean masked token or word ratio;
- mean number and length of selected spans;
- probability drop relative to the perturbation budget.

Masking preserves the input length but introduces the model's mask token. Deletion removes the text and changes the surrounding sequence. Both interventions may create out-of-distribution inputs, so results are interpreted as evaluation proxies rather than direct causal proof.

### Overlap with the LLM silver reference

Let $M$ be the set of positions selected by a localization method and $R$ the set selected by the LLM silver reference.

Precision is:

$$
\mathrm{Precision}
=
\frac{|M\cap R|}{|M|}
$$

Recall is:

$$
\mathrm{Recall}
=
\frac{|M\cap R|}{|R|}
$$

F1 is:

$$
\mathrm{F1}
=
\frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
$$

Intersection over union is:

$$
\mathrm{IoU}
=
\frac{|M\cap R|}{|M\cup R|}
$$

The overlap evaluation also considers whether at least one selected candidate overlaps the reference and, for ranked candidate spans, which rank first achieves an overlap.

These metrics measure agreement with the silver reference, not absolute correctness. They are therefore reported separately from perturbation faithfulness.

### Human evaluation

A human evaluation is planned to assess qualities that automatic metrics cannot reliably capture.

Participants are shown:

- the discussion issue;
- the full argument;
- highlighted spans produced by anonymized methods.

The method order and sampled arguments are randomized. Participants rate each explanation on a 1–10 scale for:

- **Relevance:** Do the highlighted spans indicate actual reasons for the argument's inappropriateness?
- **Sufficiency:** Are the highlighted spans sufficient to judge why the argument is inappropriate?
- **Precision:** Do the highlights consist mainly of relevant evidence, without unnecessary surrounding text?

The human study is intended to complement rather than replace the automatic evaluation. In particular, it can distinguish semantically meaningful spans from perturbations that affect the classifier for technical or distributional reasons.

---

## Repository structure

The repository separates reusable source code, generated data, method notebooks, baselines, evaluation notebooks, and result artifacts.

```text
Localizing-Inappropriateness-in-Arguments/
├── data/
│   ├── raw/
│   │   └── appropriateness_corpus/
│   │       ├── train/
│   │       ├── validation/
│   │       └── test/
│   └── processed/
│       ├── appropriateness_prepared.parquet
│       └── appropriateness_prepared_metadata.json
├── baselines/
│   ├── random_baseline.ipynb
│   └── tf-idf_baseline.ipynb
├── methods/
│   ├── attention_spans.ipynb
│   ├── integrated_gradients_spans.ipynb
│   ├── mil_spans.ipynb
│   ├── mil_spans_singles.ipynb
│   ├── shap_spans.ipynb
│   └── llm_spans.ipynb
├── evaluation/
│   ├── mask_vs_delete.ipynb
│   ├── methods_vs_baselines.ipynb
│   └── methods_vs_llm.ipynb
├── results/
│   ├── ablation_comparison/
│   ├── evaluation/
│   ├── attention_results/
│   ├── ig_results/
│   ├── mil_results/
│   ├── random_baseline_results/
│   ├── shap_results/
│   ├── tfidf_baseline_results/
│   └── llm_reference/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── prepare_data.py
│   └── utils.py
├── requirements.txt
└── README.md
```

The `data/` directory is generated locally by `python -m src.prepare_data`. Large raw and processed dataset files generally do not need to be committed to Git, provided that they can be regenerated through the documented preparation command.

### Directory and file guide

#### Data

| Path | Purpose |
|---|---|
| `data/raw/appropriateness_corpus/` | Local on-disk copy of the official Hugging Face Appropriateness Corpus. |
| `data/raw/appropriateness_corpus/train/` | Original training split stored by Hugging Face Datasets. |
| `data/raw/appropriateness_corpus/validation/` | Original validation split stored by Hugging Face Datasets. |
| `data/raw/appropriateness_corpus/test/` | Original held-out test split stored by Hugging Face Datasets. |
| `data/processed/appropriateness_prepared.parquet` | Canonical table used by all notebooks. It combines the splits and adds normalized text, stable IDs, classifier outputs, and confusion-type information. |
| `data/processed/appropriateness_prepared_metadata.json` | Reproducibility metadata describing the dataset, classifier, preprocessing settings, split sizes, and preparation time. |

#### Baselines

| File | Purpose |
|---|---|
| `baselines/random_baseline.ipynb` | Generates randomly selected spans under controlled span-length and selection configurations. It provides a chance-level reference for perturbation-based evaluation. |
| `baselines/tf-idf_baseline.ipynb` | Selects lexically salient words or spans using TF-IDF. It provides a transparent non-neural baseline that does not use classifier internals. |

#### Localization methods and silver reference

| File | Purpose |
|---|---|
| `methods/attention_spans.ipynb` | Extracts and aggregates attention-based token scores, converts selected tokens into readable spans, and evaluates them through classifier perturbation. |
| `methods/integrated_gradients_spans.ipynb` | Computes Integrated Gradients for the inappropriate class, aggregates subword attributions, builds spans, and evaluates the selected evidence. |
| `methods/mil_spans.ipynb` | Trains and evaluates the main Multiple Instance Learning setup using bags of candidate spans, including experiments with multiple candidate-span granularities. |
| `methods/mil_spans_singles.ipynb` | Runs refined MIL experiments in which each configuration uses a single candidate span length, enabling a more controlled comparison of span granularity. |
| `methods/shap_spans.ipynb` | Computes SHAP attributions for the inappropriate class, selects high-attribution segments, merges them into spans, and evaluates their perturbation effect. |
| `methods/llm_spans.ipynb` | Generates and validates Gemini span annotations. These spans are stored as a **silver reference** for evaluating the other methods and are not treated as human gold labels or as a primary localization method. |

#### Cross-method evaluation

| File | Purpose |
|---|---|
| `evaluation/mask_vs_delete.ipynb` | Compares masking and deletion as perturbation operators across the final method outputs and LLM reference spans. |
| `evaluation/methods_vs_baselines.ipynb` | Places the primary localization methods next to the random and TF-IDF baselines under a shared set of faithfulness, efficiency, and span-size metrics. |
| `evaluation/methods_vs_llm.ipynb` | Compares method spans with the Gemini silver reference using overlap-oriented metrics such as precision, recall, F1, IoU, hit rate, and rank-based measures. |

#### Results

| Path | Purpose |
|---|---|
| `results/attention_results/` | Argument-level and span-level attention results, configuration summaries, and plots. |
| `results/ig_results/` | Integrated-Gradients results, summaries, and plots. |
| `results/mil_results/` | MIL checkpoints or model outputs, selected configurations, span results, summaries, and plots. |
| `results/random_baseline_results/` | Random-baseline samples, final random spans, summaries, and plots. |
| `results/shap_results/` | SHAP attributions, selected spans, configuration summaries, and plots. |
| `results/tfidf_baseline_results/` | TF-IDF span results, summaries, and plots. |
| `results/llm_reference/` | Validated Gemini silver-reference annotations and their perturbation-evaluation outputs. |
| `results/ablation_comparison/` | Consolidated outputs and figures from the mask-versus-delete analysis. |
| `results/evaluation/` | Tables, figures, and merged outputs generated by the notebooks under `evaluation/`. |

Each method-specific result directory may contain several artifact types:

- per-argument predictions and perturbation results;
- per-span character offsets and span texts;
- development-set configuration summaries;
- fixed final test configurations;
- aggregate CSV files;
- qualitative examples;
- plots used for analysis and reporting.

#### Shared source code

| File | Purpose |
|---|---|
| `src/__init__.py` | Marks `src` as a Python package. It can remain minimal; notebook code may import directly from `src.data` and `src.utils`. |
| `src/data.py` | Loads and validates the canonical processed dataset and returns the full data frame or the individual train, validation, and test splits. |
| `src/prepare_data.py` | Downloads or loads the corpus, stores the raw splits, creates stable IDs, normalizes texts, computes the original classifier outputs, and writes the processed dataset and metadata. |
| `src/utils.py` | Contains reusable project-wide helpers for text normalization, split assignment, classifier inference, confusion types, span perturbation, highlighting, and serialization. |
| `requirements.txt` | Defines the Python packages needed for data preparation, notebooks, attribution methods, evaluation, and the Gemini API integration. |
| `README.md` | Documents the research context, setup, methods, data flow, evaluation protocol, and repository usage. |

Method-specific attribution, span-selection, training, and result-building logic remains in the corresponding notebook. Shared logic that must behave identically across methods belongs in `src/`.

## Installation

### Requirements

The repository is designed for **Python 3.11**.

A CUDA-capable GPU is strongly recommended for:

- SHAP;
- Integrated Gradients;
- attention extraction;
- MIL training.

The smaller preprocessing and evaluation steps can also run on CPU.

### 1. Clone the repository

```bash
git clone https://github.com/Karbs08/Localizing-Inappropriateness-in-Arguments.git
cd Localizing-Inappropriateness-in-Arguments
```

### 2. Create a virtual environment

#### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, the terminal should show the environment name, usually `(.venv)`.

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

The requirements include the main experiment stack:

- PyTorch;
- Transformers;
- Hugging Face Datasets;
- pandas and NumPy;
- scikit-learn;
- SHAP;
- Captum;
- Matplotlib;
- JupyterLab.

For GPU execution, make sure that the installed PyTorch build is compatible with the available CUDA version. On managed GPU systems or containers, a suitable PyTorch installation may already be provided.

### 5. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install \
  --user \
  --name localizing-inappropriateness \
  --display-name "Python (Localizing Inappropriateness)"
```

Select **Python (Localizing Inappropriateness)** as the kernel when opening the notebooks.

### 6. Deactivate the environment

When finished:

```bash
deactivate
```

---


## Google Gemini API setup

The notebook `methods/llm_spans.ipynb` uses the Google Gemini API to create the LLM-based silver-reference annotations.

The configured model is:

```python
LLM_NAME = "gemini-3-flash-preview"
```

A valid Google Gemini API key is required only for generating or regenerating the silver reference. The other localization methods, baselines, data preparation, and evaluations do not require this API key.

### 1. Create a project-level `.env` file

Create a file named `.env` in the repository root, next to `README.md` and `requirements.txt`:

```text
Localizing-Inappropriateness-in-Arguments/
├── .env
├── README.md
├── requirements.txt
├── methods/
├── evaluation/
└── src/
```

Add the API key to the file:

```dotenv
GEMINI_API_KEY=your_google_gemini_api_key
```

The notebook can load it with:

```python
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to the project-level .env file."
    )
```

The environment-variable name in `.env` and the name passed to `os.getenv(...)` must match.

### 2. Do not commit the key

The `.env` file contains a secret and must never be pushed to GitHub. Ensure that the project-level `.gitignore` contains:

```gitignore
.env
```

An optional `.env.example` file may be committed to document the required variable without exposing a key:

```dotenv
GEMINI_API_KEY=
```

### 3. Required Python packages

The Gemini notebook requires the Google Gen AI client library and `python-dotenv` in addition to the general experiment dependencies. These packages should be included in `requirements.txt`, so the normal installation command remains sufficient:

```bash
python -m pip install -r requirements.txt
```

### 4. Silver-reference reproducibility

The generated annotations depend on the configured model, prompt, schema, API behavior, and provider-side model version. The exported reference files should therefore retain at least:

- the model name;
- the prompt or prompt version;
- the returned span texts;
- original and validated offsets;
- validation or repair flags;
- confidence values, when requested;
- errors and retry information where relevant.

API keys must never be written to result files, notebooks, logs, or metadata exports.

## Data preparation

Before running the experiment notebooks, prepare the shared dataset once from the repository root:

```bash
python -m src.prepare_data
```

This command downloads the official dataset and model when they are not already available in the local Hugging Face cache. It then creates the canonical local files under:

```text
data/raw/appropriateness_corpus/train/
data/raw/appropriateness_corpus/validation/
data/raw/appropriateness_corpus/test/
data/processed/appropriateness_prepared.parquet
data/processed/appropriateness_prepared_metadata.json
```

The processed file contains the original corpus fields together with shared experiment columns such as:

- `split`;
- `global_row_id`;
- `text_norm`;
- `p_inappropriate_original`;
- `predicted_label`;
- `predicted_score`;
- `confusion_type`;
- `is_true_positive`;
- `is_false_negative`;
- `is_false_positive`;
- `is_true_negative`.

The preparation command should be rerun whenever one of the following changes:

- the dataset version;
- the binary classifier;
- the text-normalization logic;
- the maximum input length;
- the classifier label mapping;
- the shared prediction logic.

Experiment notebooks load the prepared data through:

```python
from src.data import load_prepared_splits

df_all, train_df, val_df, test_df = load_prepared_splits()
```

The data-loading functions do not need to be re-exported through `src/__init__.py`; importing them directly from `src.data` keeps their origin explicit.

---

## Running the notebooks

Start Jupyter from the **repository root**:

```bash
jupyter lab
```

Starting Jupyter from the root ensures that imports such as the following work consistently inside notebooks stored under `baselines/`, `methods/`, and `evaluation/`:

```python
from src.data import load_prepared_splits
from src.utils import normalize_text
```

Recommended order:

1. activate the virtual environment;
2. install the requirements;
3. run `python -m src.prepare_data`;
4. configure `GEMINI_API_KEY` only when the LLM silver reference must be generated;
5. start JupyterLab from the repository root;
6. select the project kernel;
7. run the baseline and method notebooks needed for the experiment;
8. verify their outputs in the corresponding method-specific directories under `results/`;
9. generate or load the validated LLM silver reference;
10. run the notebooks under `evaluation/` after all required final method files exist;
11. store consolidated tables and figures under `results/evaluation/` and `results/ablation_comparison/`.

The baseline and method notebooks are designed to be executable independently after the canonical dataset has been prepared. They still load the original classifier because each approach needs it for attribution, internal representations, span scoring, or perturbation evaluation.

The evaluation notebooks do not generate new localization methods. They consume fixed method outputs and place them under a shared comparison protocol. Consequently, final method configurations should be selected and frozen before the final evaluation notebooks are executed.

### Computational notes

- SHAP can require many classifier calls and may be slow on the full dataset.
- Integrated Gradients performs repeated forward and backward passes for each argument.
- Attention extraction requires model outputs with attention tensors.
- MIL trains additional models over many candidate spans and is the most memory-intensive approach.
- Result directories can become large, especially when storing checkpoints, per-span files, JSONL outputs, and intermediate configuration runs.

For exploratory runs, notebook-level debug limits can be used where available. Final reported results should be generated without debug subsampling.

---

## Reproducibility

The project follows several measures to keep experiments comparable:

- a fixed random seed is used for stochastic procedures where possible;
- all methods use the official train, validation, and test splits;
- the test set is reserved for final evaluation;
- one shared prepared dataset is used across notebooks;
- `global_row_id` provides a stable cross-method identifier;
- character offsets refer to the normalized experiment text;
- selected configurations are saved with their outputs;
- argument-level and span-level results are stored separately;
- random configuration search retains sample-level outputs;
- the final random baseline uses one concrete sample per argument;
- LLM offsets are validated against the source text;
- classifier probabilities before and after perturbation are retained.

Exact reproducibility of LLM-generated annotations may additionally depend on:

- the configured model, `gemini-3-flash-preview`;
- API availability and quotas;
- the prompt and response schema;
- decoding settings;
- batching and retry behavior;
- provider-side model changes.

The LLM annotation files should therefore record the model name and relevant annotation metadata. The API key itself must never be stored in committed notebooks, result files, logs, or exported metadata.

---

## Current status

### Implemented

- [x] Shared text normalization and classifier helpers
- [x] Stable IDs and confusion-type assignment
- [x] Random span baseline
- [x] TF-IDF baseline
- [x] SHAP-based localization
- [x] Integrated-Gradients-based localization
- [x] Attention-based localization
- [x] MIL-based localization
- [x] Perturbation-based evaluation within method notebooks
- [x] LLM-generated silver-reference spans
- [x] Argument-level and span-level result exports

### In progress

- [ ] Consolidated cross-method result tables
- [ ] Complete silver-reference overlap evaluation
- [ ] Human evaluation
- [ ] Final figures and statistical analysis
- [ ] Final documentation of selected configurations and results

No unfinished result should be interpreted as a reported final thesis result until the corresponding evaluation has been completed and consolidated.

---

## Limitations

This repository studies span localization in a setting without human gold rationales. Several limitations therefore apply.

### Dependence on the original classifier

The post-hoc methods explain or probe one specific binary classifier. A span that is important to this model is not necessarily the only valid human reason for judging an argument inappropriate.

### Perturbation artifacts

Masking and deletion modify the input distribution. A probability change may partly reflect grammatical disruption, changed sequence length, or unfamiliar mask patterns rather than removal of meaningful evidence.

### Attribution is not causality

SHAP, Integrated Gradients, and attention provide different notions of importance. Their scores should not automatically be interpreted as causal explanations.

### Span granularity

Quantile thresholds, windows, tokenization, word-boundary correction, candidate lengths, and merging rules can substantially affect explanation length and readability.

### Silver-reference uncertainty

The LLM-generated spans are plausible machine annotations, not gold labels. Automatic overlap with them is only one evaluation perspective.

### Human subjectivity

Appropriateness is context-sensitive, and people may disagree about which parts of an argument are inappropriate or how much context is needed. The human evaluation must therefore report variation and agreement rather than assuming one universally correct span.

---

## Citation

This project relies on the Appropriateness Corpus and classifier introduced by Ziegenbein et al. Please cite their ACL 2023 paper when using the dataset or model:

```bibtex
@inproceedings{ziegenbein-etal-2023-modeling,
    title = "Modeling Appropriate Language in Argumentation",
    author = "Ziegenbein, Timon and
      Syed, Shahbaz and
      Lange, Felix and
      Potthast, Martin and
      Wachsmuth, Henning",
    editor = "Rogers, Anna and
      Boyd-Graber, Jordan and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.238/",
    doi = "10.18653/v1/2023.acl-long.238",
    pages = "4344--4363"
}
```

---

## References

### Core resources

- [Ziegenbein et al. (2023): Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238/)
- [Appropriateness Corpus on Hugging Face](https://huggingface.co/datasets/timonziegenbein/appropriateness-corpus)
- [Binary Appropriateness Classifier on Hugging Face](https://huggingface.co/timonziegenbein/appropriateness-classifier-binary)
- [Original Appropriateness Corpus repository](https://github.com/timonziegenbein/appropriateness-corpus)

### Method references

- [Lundberg and Lee (2017): A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- [Sundararajan et al. (2017): Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html)
- [Jain and Wallace (2019): Attention is not Explanation](https://aclanthology.org/N19-1357/)
- [Abnar and Zuidema (2020): Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/)
