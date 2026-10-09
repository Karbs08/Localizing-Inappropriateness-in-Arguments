# From Document to Span: Localizing Reasons for Inappropriateness in Arguments

This repository contains the experimental code for Karsten Bruns's master's thesis on localizing evidence for **inappropriateness in arguments** under weak supervision.

Starting from a fixed document-level appropriateness classifier, the project asks:

> Which text spans support an inappropriate-class prediction, and how do their influence on the classifier and their human-perceived explanation quality compare?

The Appropriateness Corpus provides document-level labels but no gold rationale spans. The experiments therefore compare Random and TF-IDF baselines, Attention, Integrated Gradients, SHAP, and Multiple Instance Learning (MIL) without supervised span annotations. Evaluation combines masking-based classifier perturbations, ranking faithfulness, overlap with an LLM-generated silver reference, and a completed human study.

**Status:** The experiments, comparative evaluation, and human study reported in the thesis are complete. This README describes the thesis protocol and the repository layout. The thesis is the authoritative source for the experimental findings.

## Contents

- [Dataset and classifier](#dataset-and-classifier)
- [Experimental protocol](#experimental-protocol)
- [Localization methods and final configurations](#localization-methods-and-final-configurations)
- [Evaluation and main findings](#evaluation-and-main-findings)
- [Repository structure](#repository-structure)
- [What runs where and where results are written](#what-runs-where-and-where-results-are-written)
- [Installation and Makefile workflow](#installation-and-makefile-workflow)
- [Data preparation](#data-preparation)
- [Running the experiments](#running-the-experiments)
- [Gemini silver-reference setup](#gemini-silver-reference-setup)
- [Human study](#human-study)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Citation and references](#citation-and-references)

## Dataset and classifier

### Appropriateness Corpus

The experiments use [timonziegenbein/appropriateness-corpus](https://huggingface.co/datasets/timonziegenbein/appropriateness-corpus), introduced by Ziegenbein et al. in [Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238/) (ACL 2023).

The corpus contains **2,191 English arguments** on **1,154 issues**: 1,590 debate-portal arguments, 500 question-answering forum arguments, and 101 reviews. The original splits are preserved.

| Split | Arguments | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|
| Train | 1,533 | 785 | 652 | 48 | 48 |
| Validation | 220 | 88 | 66 | 30 | 36 |
| Test | 438 | 186 | 142 | 71 | 39 |

Confusion types are determined from the corpus's binary document label and the fixed classifier's prediction. A **true positive (TP)** is an argument labeled inappropriate that the classifier also predicts as inappropriate.

Relevant corpus fields are `post_id`, `issue`, `post_text`, the binary `Inappropriateness` label, and the document-level taxonomy labels. The taxonomy distinguishes Toxic Emotions, Missing Commitment, Missing Intelligibility, and Other Reasons, with finer dimensions beneath them. The corpus contains **no gold token-level or character-level rationale annotations**. The experiments localize evidence for the binary label; they do not train a span-level taxonomy classifier.

### Fixed document-level classifier

The original classifier is [timonziegenbein/appropriateness-classifier-binary](https://huggingface.co/timonziegenbein/appropriateness-classifier-binary), a **DeBERTaV3-based** model released with the corpus.

- `LABEL_0`: appropriate.
- `LABEL_1`: inappropriate.
- The fixed classifier receives the normalized argument text only, derived from `post_text`.
- Its maximum input length is **512 model tokens**.
- The issue is retained as metadata and is provided separately to MIL and the LLM annotator. It is not prepended to the fixed classifier's input.

The inappropriate-class probability is denoted by $p_1(x)=P(y=1\mid x)$. The original classifier is used to obtain document predictions and to evaluate selected evidence by rerunning it on perturbed arguments. It is not retrained in these experiments. MIL learns a separate prediction function and is evaluated separately at bag level.

## Experimental protocol

### 1. Prepare a shared dataset

`src/prepare_data.py` creates one canonical dataset for all experiments. It preserves the official splits, applies **NFKC Unicode normalization** and collapses consecutive whitespace, while preserving capitalization and punctuation. It stores the result as `text_norm`, assigns stable `global_row_id` identifiers, and computes original classifier probabilities, predictions, and confusion types.

All method outputs use character intervals in this shared normalized text. Offsets must be interpreted against `text_norm`, rather than an independently normalized copy or the original unnormalized text. Spans use start and exclusive end positions. Invalid intervals are discarded and overlapping intervals are merged before common evaluation.

### 2. Select the perturbation operator before method configuration

`evaluation/mask_vs_delete.ipynb` performs a **preliminary paired ablation on the 88 validation true positives**. It compares masking and deletion using five heterogeneous span sources: Random, Attention, two SHAP masker variants, and a lightweight MIL model. These are preliminary configurations, not the final method outputs, and the LLM reference is not part of this operator-selection experiment.

For each argument and source, the spans remain identical under both operators. **Masking is selected** because it produces more consistent positive drops across the investigated conditions and preserves surrounding sequence structure. It does not maximize the arithmetic mean drop in every condition.

All subsequent configuration searches and final perturbation evaluations use masking. Every non-special classifier token whose character interval overlaps selected evidence is replaced by one tokenizer-specific mask token. Selected multi-token regions are not collapsed into a single placeholder. Multiple spans are perturbed jointly from the original input.

### 3. Select configurations on development data

Random, TF-IDF, Attention, Integrated Gradients, and SHAP select their final span configurations on **873 development true positives**: 785 from train and 88 from validation. The TF-IDF vectorizer is fitted on all **1,753 train and validation texts**, independently of the TP subset used to select span parameters.

MIL is the exception: its models are trained on all **1,533 training arguments**, with document labels only. Each configuration retains the checkpoint with the highest validation F1. After bag-level validity checks, its localization configuration is selected on the **88 validation true positives of the original classifier**.

Configuration selection follows the common thesis rule:

1. Exclude configurations with a **mean masked-token ratio greater than 0.65**.
2. Compute within-method percentile ranks for mean probability drop, positive drop rate, and compactness; lower token coverage receives a higher compactness rank.
3. Select the eligible configuration with the highest weighted score:

$$
S(c)=0.4\,r_{\Delta p}(c)+0.4\,r_{\mathrm{PDR}}(c)+0.2\,r_{\mathrm{compact}}(c).
$$

The selected configuration is additionally checked for Pareto optimality across these three criteria. Median drops, variability, and qualitative examples support analysis but are not additional terms in the selection score. AOPC is a separate ranking diagnostic and **does not enter configuration selection**. The 0.65 limit applies to the development-set mean, not to every individual explanation or to a test-time clipping rule.

### 4. Freeze configurations and evaluate held-out outputs

One configuration per method is fixed before test evaluation. The complete test set contains **438 arguments**; the primary localization analysis uses its **186 true positives**. The LLM comparison uses a fixed random subset of **100 test true positives**. LLM annotations are generated for final evaluation only and are not used to tune the six localization approaches.

Argument-level exports retain selected spans and original/perturbed classifier outputs. Span-level exports retain individual localized regions. `global_row_id` links all method, reference, and study outputs.

## Localization methods and final configurations

| Approach | Localization signal | Final thesis configuration | Additional model training |
|---|---|---|---|
| Random | Uniformly sampled word positions | `multi_word_budgeted`, word budget `25` | No |
| TF-IDF | Lexical term scores | `top_k=8`, word context window `2` | No |
| Attention | Classification-token attention rollout | `rollout`, `quantile=0.40`, merge gap `0` | No |
| Integrated Gradients | Positive attributions for `LABEL_1` | `quantile=0.40`, merge gap `3` | No |
| SHAP | Positive segment attributions for `LABEL_1` | `quantile=0.50`, segment context window `0` | No |
| MIL | Learned candidate-span instance probabilities | `pool=topk_noisy_or`, `topk=3`, `spans=15`, `stride=4`, `freeze=True` | Yes |
| LLM silver reference | Prompted verbatim span annotation | `gemini-3-flash-preview`, temperature `0` | External annotation |

The LLM is an external reference condition, not a seventh primary localization method or human gold standard. Window parameters have different meanings across methods and are not interchangeable.

### Random and TF-IDF baselines

Random compares one contiguous word span with distributed word-budget selection. Distributed selection samples distinct word positions without replacement and merges adjacent selections. Every development configuration uses five reproducible draws per argument, averaged within the argument before aggregation. Final test evaluation retains **one reproducible draw per argument**, without choosing the best draw.

TF-IDF uses lowercased unigrams, English stop-word removal, sublinear term frequency, and L2 normalization, with `min_df=1` and `max_df=1.0`. Every occurrence of a selected term is an anchor. Each anchor receives symmetric word-context expansion, and overlapping or adjacent regions are merged. The development-fitted vectorizer is only transformed on the test set.

### Attention

The notebook compares `last_cls` with `rollout`. `last_cls` averages final-layer attention across heads and uses the classification-token row. Rollout averages heads within each layer, adds identity matrices for residual connections, normalizes rows, and composes attention across layers.

Localization operates on valid **model tokens**. A per-argument quantile selects tokens; a merge gap joins selections separated by at most the specified number of unselected tokens and includes the intervening region. Attention is used as a candidate-localization signal, not assumed to be an explanation by itself.

### Integrated Gradients

Captum's `LayerIntegratedGradients` attributes the inappropriate-class output at the classifier's embedding layer. The baseline preserves special tokens and replaces ordinary tokens with the padding token when available, otherwise the mask or unknown token. The experiment uses **16 integration steps** and internal batch size **4**.

Embedding-dimension attributions are summed to one signed score per model token and normalized by the sum of absolute scores. Only positive scores are eligible for final selection. Quantile selection and gap-based merging operate directly on **model tokens**, without first aggregating subwords to words. Expansion to complete words is applied separately for human-study presentation.

### SHAP

The SHAP implementation wraps the Hugging Face classifier with `shap.models.TransformersPipeline`. An explicit text masker uses the classifier's mask token with `collapse_mask_token=False`. Automatic algorithm selection resolves to `PartitionExplainer` in this setup. Explanations target `LABEL_1`, use batch size **8**, and are cached per argument.

Positive segment attributions are thresholded using a per-argument quantile. The context window expands selected segments by neighboring SHAP segments. Resulting intervals are merged, and empty or punctuation-only regions are discarded. The final configuration adds no segment context.

### Multiple Instance Learning

MIL treats documents as bags of candidate word spans and learns from document labels only. Each candidate is paired with the discussion issue; only the candidate's argument-text interval is returned as evidence and perturbed later. The classifier encoder is frozen and a new linear instance head is trained. Instance probabilities are aggregated using max, top-k mean, or top-k noisy-or pooling.

`mil_spans.ipynb` performs the coarse search over grouped candidate lengths; `mil_spans_singles.ipynb` performs the refined search over individual lengths. The thesis reports **42 coarse** and **70 refined** configurations. The final configuration comes from the refined search.

Candidates contain at least two words, with at most **80 candidates per argument** and maximum issue–span input length **96 model tokens**. Training uses three epochs, bag batch size one, learning rate `1e-3`, weight decay `0.01`, and gradient clipping at `1.0`. Selection requires validation AUROC of at least `0.55` and prediction standard deviation of at least `0.02` before the common localization score is applied.

The **three highest-scoring candidates** form the final explanation, with overlaps merged. Pooling `topk` controls bag prediction; the number of candidates retained for the explanation is a separate setting, although both equal three in the final configuration. Bag classification quality and the influence of the selected spans on the original classifier are evaluated separately.

## Evaluation and main findings

### Perturbation and compactness

For selected spans $S$, masking produces $x^{\mathrm{mask}(S)}$. The probability drop is

$$
\Delta p(x,S)=p_1(x)-p_1\!\left(x^{\mathrm{mask}(S)}\right).
$$

A positive drop means the inappropriate-class probability decreases. A negative drop means it increases. **PDR** is the fraction of arguments with a strictly positive drop. The **masked-token ratio** is the proportion of valid classifier-text tokens covered by the span union, with overlapping coverage counted once.

The comparison reports mean, median, and standard deviation of probability drop, PDR, masked-token coverage, and class-flip rates. Logit drops provide a supplementary diagnostic for highly confident predictions. Higher drops must be interpreted alongside coverage: selecting more text can increase perturbation strength without improving localization precision.

Paired comparisons use Wilcoxon signed-rank tests, exact McNemar tests for binary positive-drop outcomes, bootstrap confidence intervals, and Holm correction within the relevant comparison families.

### Ranking faithfulness

AOPC evaluates relevance rankings independently of final span construction at perturbation fractions **1%, 5%, 10%, 20%, and 50%**, without method-specific context expansion. Each argument has **ten matched random rankings**, averaged before dataset aggregation. AOPC is analyzed for score-based methods against their matched references.

Ranking units differ: TF-IDF uses word occurrences, Attention and IG use model tokens, and SHAP uses explainer segments. AOPC therefore supports within-method comparisons with matched random rankings more directly than strict cross-method ranking comparisons.

### LLM-reference overlap

The fixed 100-argument subset is annotated with Gemini. Predictions and references are mapped to the same classifier-token positions. The comparison reports argument-macro Precision, Recall, F1, IoU, median F1/IoU, overlap-hit rate, and supplementary micro aggregates. A hit means at least one selected token overlaps the reference.

These are agreement measures against a **silver reference**, not accuracy against gold explanations. The comparison also evaluates the perturbation behavior and coverage of the LLM spans on the same 100 arguments.

### Results at a glance

The following values are the final masking results on the **186 test true positives** (thesis Table 6.10), not the development results or the 100-argument LLM subset.

| Method | Mean probability drop | PDR | Mean masked-token ratio |
|---|---:|---:|---:|
| Random | 0.128 | 83.3% | 50.1% |
| TF-IDF | 0.057 | 81.2% | 63.6% |
| Attention | 0.237 | 96.2% | 60.0% |
| Integrated Gradients | 0.123 | 89.8% | 62.7% |
| SHAP | 0.203 | 95.2% | 41.4% |
| MIL | 0.153 | 78.0% | 55.7% |

Attention produces the strongest mean response and highest PDR on this subset. SHAP combines a strong response with substantially lower coverage and provides the most balanced overall trade-off across the thesis's evaluation perspectives. MIL obtains the highest macro overlap with the LLM reference on the 100-argument subset (F1 `0.3831`, IoU `0.2795`). In the human study, LLM spans receive the strongest combined Completeness/Precision assessment; SHAP provides the strongest combined assessment among the six primary approaches. TF-IDF receives the highest mean Completeness rating but low Precision.

These conclusions are specific to the evaluated classifier, corpus, configurations, and samples. No single measure establishes both classifier faithfulness and human plausibility.

## Repository structure

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
├── human_study/
│   ├── prepare_survey_data.py
│   ├── generate_limesurvey.py
│   ├── survey_input/
│   │   ├── study_items.csv
│   │   └── selected_argument_ids.txt
│   ├── survey_output/
│   │   ├── span_human_study_import.txt
│   │   ├── selected_arguments.csv
│   │   ├── variant_design.csv
│   │   ├── question_mapping.csv
│   │   ├── preview_version_1.html
│   │   └── generation_report.txt
│   └── survey_results/
├── results/
│   ├── ablation_comparison/
│   ├── evaluation/
│   │   ├── comparison/
│   │   ├── methods_vs_llm/
│   │   └── mil/
│   ├── attention_results/
│   ├── ig_results/
│   ├── mil_results/
│   ├── random_baseline_results/
│   ├── shap_results/
│   ├── tfidf_baseline_results/
│   └── llm_reference/
├── src/
│   ├── survey/
│   │   ├── limesurvey_tsv.py
│   │   └── survey_design.py
│   ├── __init__.py
│   ├── data.py
│   ├── prepare_data.py
│   └── utils.py
├── Makefile
├── requirements.txt
└── README.md
```

## What runs where and where results are written

All paths below are relative to the repository root. The table documents the result-directory layout; concrete filenames and run subdirectories are controlled by the export cells in the corresponding notebook. Preserve selected configurations and run metadata alongside their outputs.

| Notebook or script | Inputs and work performed | Output directory and contents |
|---|---|---|
| `src/prepare_data.py` | Loads corpus splits, normalizes text, creates stable IDs, and computes fixed-classifier predictions. | `data/raw/appropriateness_corpus/`: raw splits; `data/processed/`: canonical Parquet and preparation metadata. |
| `evaluation/mask_vs_delete.ipynb` | Compares paired masking/deletion on validation TPs using preliminary span sources. Establishes the common operator before final configuration search. | `results/ablation_comparison/`: operator summaries, argument-level comparisons, confidence intervals, tests, and figures. |
| `baselines/random_baseline.ipynb` | Searches random-selection configurations on development TPs; generates one final sample per test argument. | `results/random_baseline_results/`: repeated development samples, configuration summaries, fixed final spans, argument/span exports, and plots. |
| `baselines/tf-idf_baseline.ipynb` | Fits development-text TF-IDF, searches anchor/context settings, and evaluates final test spans and rankings. | `results/tfidf_baseline_results/`: lexical scores/anchors, configuration summaries, final spans, perturbation results, AOPC outputs, and plots. |
| `methods/attention_spans.ipynb` | Computes attention scores, compares `last_cls`/`rollout`, selects token spans, and evaluates final outputs. | `results/attention_results/`: scores, configuration summaries, argument/span exports, AOPC outputs, and plots. |
| `methods/integrated_gradients_spans.ipynb` | Computes embedding-layer IG, searches token quantile/gap settings, and evaluates final outputs. | `results/ig_results/`: signed attributions, configuration summaries, argument/span exports, AOPC outputs, and plots. |
| `methods/shap_spans.ipynb` | Caches SHAP explanations, searches positive-segment selection settings, and evaluates final outputs. | `results/shap_results/`: cached explanations, configuration summaries, argument/span exports, AOPC outputs, and plots. |
| `methods/mil_spans.ipynb` | Trains coarse MIL configurations over grouped candidate lengths and evaluates bag/span behavior. | `results/mil_results/`: coarse-search checkpoints, configuration summaries, candidate scores, and localization outputs. |
| `methods/mil_spans_singles.ipynb` | Trains refined single-length MIL configurations and selects the final checkpoint and spans. | `results/mil_results/`: refined-search checkpoints and final bag/candidate/span outputs; `results/evaluation/mil/`: consolidated MIL bag-classification diagnostics and figures. |
| `methods/llm_spans.ipynb` | Annotates the fixed 100 test TPs, validates verbatim offsets, and evaluates reference spans with the fixed classifier. | `results/llm_reference/`: annotation checkpoints, raw/validated responses, repair metadata, reference spans, and perturbation outputs. |
| `evaluation/methods_vs_baselines.ipynb` | Aligns fixed method outputs by ID and compares perturbation strength, consistency, compactness, and supplementary diagnostics. | `results/evaluation/comparison/`: merged comparisons, statistical tables, plots, and qualitative examples. |
| `evaluation/methods_vs_llm.ipynb` | Aligns method outputs with the 100-argument silver reference; compares overlap, perturbation response, and coverage. | `results/evaluation/methods_vs_llm/`: overlap summaries, matched-subset comparisons, statistical tables, and plots. |
| `human_study/prepare_survey_data.py` | Aligns seven explanation sources, selects seven eligible arguments with distinct issues, and prepares presentation spans. | `human_study/survey_input/`: `study_items.csv` and `selected_argument_ids.txt`. |
| `human_study/generate_limesurvey.py` | Builds the blinded four-explanation study and seven balanced variants from the prepared study items. | `human_study/survey_output/`: import file, selected arguments, variant design, question mapping, HTML preview, and generation report. |
| Human-study response analysis | Maps exported question codes to sources, aggregates participant-level ratings/rankings, and analyzes study results. | `human_study/survey_results/`: result plots and tables from the completed survey. The supplied layout does not list a separate analysis notebook or script. |

**Generation and evaluation have different roles:** method notebooks create scores and final spans; evaluation notebooks consume fixed outputs and create shared comparisons. `make study` generates the questionnaire artifacts; it does not collect responses or perform response analysis.

### Shared source files

| File | Responsibility |
|---|---|
| `src/data.py` | Loads the canonical prepared dataset and exposes the full table and individual splits. |
| `src/prepare_data.py` | Owns raw-data preparation, normalization, IDs, original predictions, and preparation exports. |
| `src/utils.py` | Shared normalization, classifier, confusion-type, span-perturbation, highlighting, and serialization helpers. |
| `src/survey/limesurvey_tsv.py` | Reusable LimeSurvey TSV construction. |
| `src/survey/survey_design.py` | Balanced explanation assignments and questionnaire variants. |
| `src/__init__.py` | Package marker; imports can come directly from `src.data` and `src.utils`. |
| `Makefile` | Environment setup, data preparation, Jupyter startup, cleanup, and survey-generation shortcuts. |
| `requirements.txt` | Python dependencies for the experimental workflow. |

Method-specific attribution, training, configuration search, and export logic remains in the corresponding notebooks.

## Installation and Makefile workflow

Use **Python 3.11**. CUDA acceleration is recommended for attribution and MIL; the implementation also supports MPS and CPU. The thesis experiments used Python `3.11.10`, PyTorch `2.5.0` with CUDA `12.1`, primarily an NVIDIA A100 MIG `3g.20gb` allocation with 20 GB GPU memory on a Slurm-managed cluster.

```bash
git clone https://github.com/Karbs08/Localizing-Inappropriateness-in-Arguments.git
cd Localizing-Inappropriateness-in-Arguments
make init
make jupyter
```

`make init` creates the environment, installs requirements, and prepares the shared dataset. Survey generation is a separate step because it requires final explanation outputs.

| Command | Purpose |
|---|---|
| `make init` | Create `.venv`, install requirements, and prepare the shared dataset. |
| `make venv` | Create the virtual environment. |
| `make install` | Install or update requirements. |
| `make prepare-data` | Prepare data if the processed files are missing. |
| `make force-prepare-data` | Rerun preparation after relevant data, model, or preprocessing changes. |
| `make kernel` | Register the project Jupyter kernel. |
| `make jupyter` | Start JupyterLab using the project environment. |
| `make study` | Run study-data preparation, then questionnaire generation. |
| `make clean-data` | Remove generated raw and processed data. |
| `make clean-venv` | Remove the virtual environment. |

### Manual setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name localizing-inappropriateness --display-name "Python (Localizing Inappropriateness)"
python -m src.prepare_data
jupyter lab
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. Select the project kernel in Jupyter. GPU execution requires a PyTorch build compatible with the available CUDA environment.

## Data preparation

Run preparation from the repository root:

```bash
make prepare-data
# Alternatively, with the project environment activated:
python -m src.prepare_data
```

The canonical output is `data/processed/appropriateness_prepared.parquet`, accompanied by `appropriateness_prepared_metadata.json`. In addition to corpus fields, shared columns include `split`, `global_row_id`, `text_norm`, `p_inappropriate_original`, `predicted_label`, `predicted_score`, `confusion_type`, and the confusion-type indicator fields.

Notebooks load the same prepared splits through:

```python
from src.data import load_prepared_splits

df_all, train_df, val_df, test_df = load_prepared_splits()
```

Rerun preparation when the dataset, classifier/tokenizer revision, normalization, maximum input length, label mapping, or prediction logic changes. Dependent cached attributions and result files must also be regenerated or kept under distinct run identities.

## Running the experiments

1. Initialize the environment and prepare the shared dataset.
2. Run `evaluation/mask_vs_delete.ipynb` to reproduce the preliminary operator comparison.
3. Run the required baseline and localization notebooks. For MIL, run the coarse search before the refined single-length search.
4. Select configurations using development data, retain their settings, and produce fixed final test outputs in the method-specific result directories.
5. Run `evaluation/methods_vs_baselines.ipynb` after the required final outputs exist.
6. Generate or load the validated 100-argument LLM reference, then run `evaluation/methods_vs_llm.ipynb`.
7. Generate the questionnaire from fixed explanations and inspect the study artifacts. Analyze collected response exports separately.

Run commands from the repository root and select the project environment. Notebook kernels may use their notebook directory as the working directory even when Jupyter is launched from the root; ensure each notebook's root-path setup resolves `src/`, `data/`, and `results/` relative to the repository root.

Attribution can be costly: SHAP requires many classifier calls, IG repeatedly evaluates gradients, Attention retains attention tensors, and MIL trains separate models for the configuration search. Use debug limits for exploration where available; final thesis results require the documented complete evaluation subsets.

## Gemini silver-reference setup

Only generating or regenerating the silver reference requires a Gemini API key. Evaluating existing annotations does not require fresh API calls.

Create a root-level `.env` file containing:

```dotenv
GEMINI_API_KEY=your_google_gemini_api_key
```

The notebook uses the Google Gen AI client and `python-dotenv`. Ensure `.env` is excluded from Git and never store the key in notebook outputs, result metadata, or logs.

The thesis reference configuration is:

```python
LLM_NAME = "gemini-3-flash-preview"
```

The annotator receives the argument ID, discussion issue, and normalized text, with inappropriateness treated as given. It returns minimal **verbatim** spans under a structured JSON schema with start/exclusive-end offsets and self-reported confidence. Generation uses temperature `0`, batches of **five arguments**, intermediate checkpoints, and retries for temporary failures.

Offsets are checked against the source text. Exact string matching repairs recoverable positions; unalignable spans are rejected. Store raw responses, original and validated offsets, repair flags, model/prompt settings, and unresolved cases. Confidence values are auxiliary metadata, not calibrated probabilities or evaluation weights. Preview-model availability and provider changes can affect regeneration; a substitute model constitutes a different reference run.

## Human study

### Completed study design

The study uses **seven arguments from seven different discussion issues**, sampled from the common test-TP pool with non-empty explanations from all seven sources. Because the LLM annotations cover 100 test TPs, this subset defines the initial candidate pool.

The questionnaire follows a **balanced incomplete-block design**:

- Four blinded explanations, **A–D**, are shown for each argument.
- Each participant evaluates seven arguments and **28 explanations**.
- Each source occurs four times per completed questionnaire.
- Each pair of sources occurs together twice.
- Each source appears once in each of the four display positions.
- Seven questionnaire variants rotate assignments across arguments and vary argument order.

The seven sources are Random, TF-IDF, Attention, IG, SHAP, MIL, and the LLM reference. Participants receive an introduction, comprehension question, and practice example. The supplied inappropriate document label is treated as given; participants evaluate localization rather than relabeling documents.

Explanations receive two independent **1–7 ratings**:

| Criterion | Question |
|---|---|
| Completeness | Do the highlights cover all important parts that could reasonably explain the argument's inappropriateness? |
| Precision | Do the highlights focus on those parts without unnecessary or unrelated text? |

Participants then rank the four explanations from best to worst. Relevance and Sufficiency are not separate collected criteria in the final study. A supplementary `F_human` combines the rescaled Completeness and Precision scores using a harmonic mean; it is an exploratory summary, not a conventional classification F1 score.

**Ten participants** completed the study, yielding **280 explanation evaluations** and **40 evaluations per source**. Analysis uses participant-level method summaries, Friedman tests, Kendall's W, Holm-adjusted Wilcoxon comparisons, and ordinal Krippendorff's alpha. Inferential findings remain exploratory given the small sample and incomplete-block design.

Before survey presentation, intersected words are expanded to complete word boundaries and resulting overlaps are merged consistently across sources. This affects the **human-facing highlights only**. Automatic comparisons retain the original model-derived intervals and scores.

### Regenerating the questionnaire

With all final method and reference outputs available:

```bash
source .venv/bin/activate
make study
```

The two stages can also be executed separately:

```bash
python human_study/prepare_survey_data.py
python human_study/generate_limesurvey.py
```

Preparation writes `human_study/survey_input/study_items.csv` and `selected_argument_ids.txt`. The generator writes:

| Artifact in `human_study/survey_output/` | Purpose |
|---|---|
| `span_human_study_import.txt` | Importable LimeSurvey questionnaire structure. |
| `selected_arguments.csv` | Selected study arguments. |
| `variant_design.csv` | Argument/source/display assignments across the seven variants. |
| `question_mapping.csv` | Mapping from blinded question codes to explanation sources. |
| `preview_version_1.html` | Static preview of variant 1. |
| `generation_report.txt` | Generation summary and validation information. |

Inspect the import and preview before reusing the questionnaire. Preserve the selected IDs, variant design, and question mapping with each response export. Survey-result plots and tables belong in **`human_study/survey_results/`**, separately from generated questionnaire artifacts.

## Reproducibility and limitations

- The global seed is **42** for Python, NumPy, and PyTorch where applicable. Random draws additionally use deterministic argument/configuration/repetition seeds.
- Official splits, shared normalized text, stable IDs, fixed configurations, and the common classifier interface support cross-method alignment.
- Save the exact dataset, classifier, and tokenizer revisions, package versions, preprocessing settings, and run parameters. A Hugging Face repository name alone does not identify an immutable snapshot. No exact model commit is asserted here because it is not specified in the supplied thesis/README.
- Retain original and perturbed predictions, span offsets, configuration summaries, MIL checkpoints, cached attribution settings, and LLM validation metadata.
- The original classifier truncates to 512 tokens. Selected evidence outside the classifier-visible input cannot explain its current prediction and must be distinguished from effective perturbation coverage.
- Masking is a controlled intervention but still creates artificial inputs. Probability drops and AOPC are proxies for model sensitivity, not causal proof or a unique rationale.
- Attention, IG, SHAP, TF-IDF, and MIL use different scoring units and context. Span size and merging rules influence both effects and readability.
- The LLM receives issue context and generates semantic evidence rather than reproducing the fixed classifier's internal decision process. Overlap with it is not gold accuracy.
- The human study covers ten participants and seven arguments. Subjectivity, incomplete-block presentation, and presentation-only word expansion limit generalization.
- This repository localizes binary inappropriateness evidence. Associating spans with taxonomy categories and evaluating explanation sufficiency through separate retention experiments are directions for future work rather than completed outputs.

## Citation and references

When using the corpus or released classifier, cite Ziegenbein et al.:

```bibtex
@inproceedings{ziegenbein-etal-2023-modeling,
  title = "Modeling Appropriate Language in Argumentation",
  author = "Ziegenbein, Timon and Syed, Shahbaz and Lange, Felix and Potthast, Martin and Wachsmuth, Henning",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2023",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.acl-long.238/",
  doi = "10.18653/v1/2023.acl-long.238",
  pages = "4344--4363"
}
```

Core resources and method references:

- [Ziegenbein et al. (2023): Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238/)
- [Appropriateness Corpus on Hugging Face](https://huggingface.co/datasets/timonziegenbein/appropriateness-corpus)
- [Released binary classifier on Hugging Face](https://huggingface.co/timonziegenbein/appropriateness-classifier-binary)
- [Original corpus repository](https://github.com/timonziegenbein/appropriateness-corpus)
- [Lundberg and Lee (2017): A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- [Sundararajan et al. (2017): Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html)
- [Jain and Wallace (2019): Attention is not Explanation](https://aclanthology.org/N19-1357/)
- [Abnar and Zuidema (2020): Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/)

See the thesis bibliography for the complete set of references and the thesis chapters on experiments, discussion, and conclusion for detailed results and interpretation.
