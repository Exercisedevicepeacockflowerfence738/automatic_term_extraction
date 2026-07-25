<div align="center">

# Automatic Term Extraction from Scratch

**Keyword extractors tell you what a document is about.<br>Term extractors tell you what a domain is made of.**

Building the C-value algorithm and a complete ATE pipeline in pure Python —<br>
with error analysis, benchmark context, and a decision guide.

[![Medium](https://img.shields.io/badge/Medium-Read%20the%20article-000000?logo=medium&logoColor=white)](https://medium.com/@huseyinceniik/automatic-term-extraction-the-research-field-behind-your-keyword-extractor-f212338829b8)
[![Kaggle](https://img.shields.io/badge/Kaggle-Run%20the%20notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/huseyincenik/automatic-term-extraction)
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?logo=github&logoColor=white)](https://github.com/huseyincenik/automatic_term_extraction)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Core pipeline](https://img.shields.io/badge/core%20pipeline-zero%20dependencies-2ECC40)](automatic-term-extraction-from-scratch.ipynb)
[![Licence](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

<img src="figures/Automatic_Term_Extraction_Infographic.png" alt="Automatic Term Extraction — overview infographic" width="920">

</div>

---

## Table of Contents

- [What this is](#what-this-is)
- [Terms are not keywords](#terms-are-not-keywords)
- [The pipeline](#the-pipeline)
- [Quick start](#quick-start)
- [The problem C-value solves](#the-problem-c-value-solves)
- [Results](#results)
- [What goes wrong, and why it matters](#what-goes-wrong-and-why-it-matters)
- [Choosing a method](#choosing-a-method)
- [How the field got here](#how-the-field-got-here)
- [Repository layout](#repository-layout)
- [Figures](#figures)
- [References](#references)

---

## What this is

**Automatic Term Extraction (ATE)** is an NLP task that identifies domain-specific
terminology in specialised corpora. It has a formal definition, a manually annotated
multilingual benchmark, three survey papers published between 2023 and 2026, and an active
research frontier — and almost no practical coverage outside the academic literature.

This repository is the code behind an article that tries to close that gap. It contains:

| | |
|---|---|
| 📓 **A runnable notebook** | The full pipeline built step by step, executed with outputs |
| 🐍 **A dependency-free implementation** | ~150 lines of standard-library Python, inside the notebook |
| 📊 **The article's figures** | All 20, regenerable from source |
| 📝 **The article itself** | [on Medium](https://medium.com/@huseyinceniik/automatic-term-extraction-the-research-field-behind-your-keyword-extractor-f212338829b8) |

> **Start here:** [**Run the notebook on Kaggle**](https://www.kaggle.com/code/huseyincenik/automatic-term-extraction) —
> nothing to install, nothing to download.

---

## Terms are not keywords

The distinction is not pedantic; it determines the entire system design.

| Dimension | Keyword extraction | Automatic term extraction |
|---|---|---|
| **Question answered** | What is this document about? | What vocabulary defines this domain? |
| **Unit of analysis** | A single document | A domain corpus |
| **Candidate generation** | All n-grams, or stopword-delimited chunks | POS-constrained patterns |
| **Ranking signal** | TF-IDF, TextRank, YAKE, KeyBERT | C-value, weirdness, neural token classification |
| **Reference corpus** | Implicit — the same collection | Explicit general-language contrast corpus |
| **Nested terms** | Not modelled | **Central concern** |
| **Typical output** | 5–10 keywords per document | A ranked termbase for the domain |
| **Evaluation** | Largely qualitative | Gold standard; precision, recall, F1 |

A TF-IDF pipeline over a unigram vocabulary cannot, in principle, emit
*doubly fed induction generator* as a single unit. Neither can it model the relationship
between *wind farm* and *offshore wind farm*.

---

## The pipeline

Nearly all classical ATE systems share a two-stage architecture: a **linguistic filter**
proposes syntactically plausible candidates, and a **statistical ranker** scores them.
A 2026 systematic review found this hybrid design in 50% of 113 papers surveyed.

```mermaid
flowchart LR
    A["<b>Domain corpus</b><br/><i>raw specialised text</i>"]
    B["<b>① Linguistic filter</b><br/>POS patterns<br/><i>ADJ or NOUN, then NOUN</i>"]
    C["<b>② Unithood ranking</b><br/>C-value<br/><i>nested-term correction</i>"]
    D["<b>③ Termhood ranking</b><br/>Weirdness<br/><i>vs. general corpus</i>"]
    E["<b>④ Termbase</b><br/><i>ranked and thresholded</i>"]

    A --> B --> C --> D --> E

    classDef src fill:#eef2f5,stroke:#7F8C8D,stroke-width:2px,color:#2C3E50
    classDef ling fill:#e8f0f8,stroke:#5B8DB8,stroke-width:2px,color:#2C3E50
    classDef stat fill:#fdf4e0,stroke:#E1A624,stroke-width:2px,color:#2C3E50
    classDef out fill:#e9f6f1,stroke:#5BAD8D,stroke-width:2px,color:#2C3E50

    class A src
    class B ling
    class C,D stat
    class E out
```

The ranking stage measures two **distinct** properties:

- **Unithood** — is this sequence of words a coherent lexical unit, or an accidental collocation?
- **Termhood** — assuming it is a unit, is it specific to this domain, or general language?

Systems that measure only one fail characteristically: unithood alone accepts frequent but
general phrases; termhood alone accepts fragments of terms.

---

## Quick start

The fastest path is the notebook — it needs nothing installed and nothing downloaded:

**[▶ Open it on Kaggle](https://www.kaggle.com/code/huseyincenik/automatic-term-extraction)**

Or run it locally:

```bash
git clone https://github.com/huseyincenik/automatic_term_extraction.git
cd automatic_term_extraction
jupyter notebook automatic-term-extraction-from-scratch.ipynb
```

Sections 3–8 of the notebook depend on nothing outside the Python standard library.
Running them end to end produces:

```
=== Multi-word terms only (have_single_word=False) ===
 #  TERM                               FREQ   C-VALUE    WEIRD  TERMHOOD
------------------------------------------------------------------------
 1  wind turbine                          6      4.92     10.7     12.10
 2  wind farm                             5      3.86      8.5      8.67
 3  wind speed                            5      4.00      5.8      7.66
 4  offshore wind                         3      1.83      7.6      3.94
 5  offshore wind farm                    2      1.58      7.3      3.36
```

Using the pipeline on your own text — `extract_terms` is defined in the notebook
(and also in `figures/generate_figures.py`, which inlines the whole pipeline):

```python
terms = extract_terms(
    technical_corpus=my_domain_text,
    general_corpus=my_background_text,
    top_n=25,
    min_freq=2,
    have_single_word=False,   # multi-word terms only
)

for t in terms:
    print(f"{t['termhood']:6.2f}  {t['term']}")
```

To regenerate every figure in the article (needs `matplotlib` and `numpy`):

```bash
pip install -r requirements.txt
python figures/generate_figures.py    # all 20 figures, ~10 seconds
```

`generate_figures.py` is **self-contained**: it carries its own copy of the pipeline, so
the figures and the notebook are computed by the same code.

---

## The problem C-value solves

Raw frequency systematically over-ranks substrings. If *offshore wind farm* occurs twice,
then *wind farm* necessarily occurs at least twice as well — and a frequency-ranked list
credits the shorter string with occurrences it never independently earned.
Frantzi et al. (2000) named these **nested terms**.

```mermaid
flowchart TD
    O1["<b>offshore wind farm</b><br/>occurs 2×"]
    O2["<b>wind farm</b> standing alone<br/>occurs 3×"]
    F["Raw frequency of 'wind farm'<br/><b>f = 5</b>"]
    D{"Is it nested inside<br/>a longer candidate?"}
    S["Subtract the borrowed frequency:<br/>the mean f of every longer<br/>candidate containing it"]
    R["<b>C-value = 3.86</b><br/><i>not 5</i>"]
    K["<b>axis wind</b><br/>almost always nested<br/>f = 2 → C-value <b>0.75</b><br/><i>demoted to the bottom</i>"]

    O1 --> F
    O2 --> F
    F --> D
    D -- yes --> S --> R
    D -- no --> N["Keep full frequency,<br/>scaled by log₂ of length"]
    S -.-> K

    classDef obs fill:#eef2f5,stroke:#7F8C8D,color:#2C3E50
    classDef calc fill:#fdf4e0,stroke:#E1A624,stroke-width:2px,color:#2C3E50
    classDef good fill:#e9f6f1,stroke:#5BAD8D,stroke-width:2px,color:#2C3E50
    classDef bad fill:#fdecea,stroke:#C0392B,stroke-width:2px,color:#2C3E50

    class O1,O2,F obs
    class D,S,N calc
    class R good
    class K bad
```

The formula:

$$
\text{C-value}(a) =
\begin{cases}
\log_2 |a| \cdot f(a) & \text{if } a \text{ is not nested} \\[8pt]
\log_2 |a| \cdot \left( f(a) - \dfrac{1}{P(T_a)} \sum_{b \in T_a} f(b) \right) & \text{otherwise}
\end{cases}
$$

where $|a|$ is the length of candidate $a$ in words, $f(a)$ its corpus frequency, $T_a$ the
set of longer candidates containing $a$, and $P(T_a)$ the size of that set.

Termhood comes from **weirdness**, contrasting the technical corpus against general language:

$$
\text{weirdness}(w) = \frac{f_{\text{tech}}(w) \,/\, N_{\text{tech}}}{f_{\text{gen}}(w) \,/\, N_{\text{gen}}}
$$

and the two are combined so that weirdness *re-ranks* rather than dominates:

$$
\text{termhood}(a) = \text{C-value}(a) \cdot \log\!\big(1 + \text{weirdness}(a)\big)
$$

---

## Results

Running the pipeline on a wind-energy corpus — deliberately, because **wind energy is one
of the four ACTER benchmark domains**:

| Rank | Term | f(a) | C-value | Weirdness | Termhood | Assessment |
|---:|---|---:|---:|---:|---:|---|
| 1 | wind turbine | 6 | 4.92 | 10.7 | 12.10 | ✅ valid |
| 2 | wind farm | 5 | 3.86 | 8.5 | 8.67 | ✅ valid |
| 3 | wind speed | 5 | 4.00 | 5.8 | 7.66 | ✅ valid |
| 4 | offshore wind | 3 | 1.83 | 7.6 | 3.94 | ⚠️ fragment of *offshore wind farm* |
| 5 | offshore wind farm | 2 | 1.58 | 7.3 | 3.36 | ✅ valid |
| 6 | axis wind turbine | 2 | 1.58 | 6.8 | 3.26 | ⚠️ should be *horizontal axis wind turbine* |
| 7 | electrical energy | 2 | 2.00 | 3.2 | 2.85 | ✅ valid |
| 8 | power output | 3 | 2.00 | 2.4 | 2.46 | ✅ valid |
| 9 | rotor blades | 2 | 1.00 | 4.7 | 1.75 | ✅ valid |
| 10 | axis wind | 2 | 0.75 | 5.4 | 1.39 | ❌ spurious |

**Precision@10 = 70%**, with no training data, no annotation, and nothing larger than a
`Counter`.

> ⚠️ This figure is **not** comparable to published ACTER scores. It is precision@10 on a
> 250-word toy corpus judged by one person; published systems report precision, recall and
> F1 over a fully annotated held-out domain. Treat it as a sanity check, never as a
> benchmark result.

---

## What goes wrong, and why it matters

The failures are more instructive than the successes.

```mermaid
flowchart TD
    T["A genuine domain term<br/>in the source text"]
    G{"Did the linguistic filter<br/>generate it as a candidate?"}
    L["<b>Lost before scoring</b><br/><i>yaw control</i> — a line break fell<br/>between the two words, and the<br/>sentence splitter treats it as a boundary"]
    F{"Does it clear<br/>min_freq = 2?"}
    M["<b>Lost to the threshold</b><br/><i>pitch control</i>, <i>capacity factor</i>,<br/><i>power curve</i>, <i>rotor diameter</i><br/>— each occurs exactly once"]
    B{"Are its boundaries<br/>correct?"}
    P["<b>Boundary error</b><br/><i>axis wind turbine</i> instead of<br/><i>horizontal axis wind turbine</i>"]
    OK["<b>Extracted correctly</b>"]

    T --> G
    G -- no --> L
    G -- yes --> F
    F -- no --> M
    F -- yes --> B
    B -- no --> P
    B -- yes --> OK

    classDef q fill:#eef2f5,stroke:#7F8C8D,color:#2C3E50
    classDef fail fill:#fdecea,stroke:#C0392B,stroke-width:2px,color:#2C3E50
    classDef warn fill:#fdf4e0,stroke:#E1A624,stroke-width:2px,color:#2C3E50
    classDef good fill:#e9f6f1,stroke:#5BAD8D,stroke-width:2px,color:#2C3E50

    class T,G,F,B q
    class L,M fail
    class P warn
    class OK good
```

Three lessons, each of which anticipates a finding in the literature:

1. **Frequency thresholding destroys recall.** Candidates fall from 98 at `min_freq=1` to
   33 at `min_freq=2`. Frequency-based methods are *structurally* incapable of recovering
   rare terms — which is exactly the dimension on which TermEval 2020 found systems to
   differ most sharply.

2. **A term the filter never proposes cannot be recovered by any ranker.** *Yaw control*
   is absent from the output not because it scored badly, but because a line break split
   the bigram before scoring began. The filter stage deserves as much attention as the
   algorithm — which is why production systems use a real sentence segmenter, not a regex.

3. **Boundary errors are an open research problem, not a bug here.** The 2023 survey lists
   nested terms as an open challenge for *Transformer* models: current systems "often
   predict nested shorter terms rather than complete multi-word terms." The phenomenon
   Frantzi et al. named in 1998 appeared unprompted in a 60-line implementation.

---

## Choosing a method

```mermaid
flowchart TD
    Q1{"Do you have annotated<br/>in-domain training data?"}
    Q2{"Any labelled examples<br/>at all?"}
    Q3{"Must every inclusion decision<br/>be explainable to a domain expert?"}

    U["<b>Unsupervised statistical</b><br/>C-value + weirdness · PyATE<br/>F1 ≈ 15–39"]
    L["<b>LLM in-context learning</b><br/>select demos by <i>syntax</i>, not meaning<br/>F1 ≈ 60 cross-domain"]
    Z["<b>Cross-lingual zero-shot</b><br/>XLM-R transfer<br/>F1 58.3–69.8"]
    H["<b>Feature-engineered supervised</b><br/>HAMLET · 152 features<br/>F1 54.2–66.1"]
    T["<b>Transformer token classification</b><br/>XLM-R · sequence labelling<br/>F1 58.3–69.6"]

    Q1 -- "no" --> Q2
    Q1 -- "only in another language" --> Z
    Q1 -- "yes" --> Q3
    Q2 -- "none" --> U
    Q2 -- "a handful" --> L
    Q3 -- "yes" --> H
    Q3 -- "no" --> T

    classDef q fill:#eef2f5,stroke:#7F8C8D,color:#2C3E50
    classDef unsup fill:#fdf4e0,stroke:#E1A624,stroke-width:2px,color:#2C3E50
    classDef neural fill:#fdecea,stroke:#C0392B,stroke-width:2px,color:#2C3E50
    classDef llm fill:#f6eefb,stroke:#8E44AD,stroke-width:2px,color:#2C3E50

    class Q1,Q2,Q3 q
    class U,H unsup
    class T,Z neural
    class L llm
```

> **The decision rule in one line:** LLMs win when labelled data is scarce; fine-tuned
> encoders win when it is not. On the in-domain BCGM benchmark the gap is 53.8 vs. **88.5** F1.

Two findings worth knowing before you reach for an LLM:

- **Retrieve few-shot demonstrations by syntax, not semantics.** Chun et al. (2025) show
  semantic similarity has a *negative* correlation with F1 across domains, because words
  that are terms in one field are generic in another.
- **Syntactic retrieval works with zero term overlap.** Their demonstrations contained
  *none* of the gold terms and still performed competitively. A term's meaning does not
  transfer across domains; its shape does.

---

## How the field got here

```mermaid
timeline
    title From hand-crafted statistics to large language models
    1998 : C-value / NC-value : Frantzi, Ananiadou and Mima : hybrid linguistic + statistical
    2000s : Statistical era : TF-IDF, weirdness, term extractor : evaluation is incomparable
    2020 : ACTER and TermEval : 3 languages, 4 domains : the field becomes measurable
    2021 : HAMLET : 152 engineered features : best non-neural, F1 66.1
    2022 : Transformer token classification : XLM-R sequence labelling : F1 69.6
    2024 : LLM few-shot : GPT-3.5 beats fine-tuned BERT when data is scarce
    2025 : Syntactic retrieval : demonstrations chosen by parse tree : F1 60.2 cross-domain
    2026 : Nested terms still unsolved : 28 years after they were named
```

Two statistics that put this in perspective, from a systematic mapping of 113 papers
published 2015–2022:

- **C-value is used in 50%** of studies — an algorithm from 1998, still the most-used
  ranking function in its field. TF-IDF is second at 31%.
- **Hybrid designs account for 50%** of studies, purely statistical 38.2%, purely
  linguistic 11.8%.

---

## Repository layout

```
automatic_term_extraction/
├── automatic-term-extraction-from-scratch.ipynb   ← the notebook (executed, 70 cells)
├── requirements.txt                               ← only needed to regenerate figures
├── LICENSE
└── figures/
    ├── generate_figures.py     ← self-contained; regenerates all 20 figures
    ├── fig1–fig7*.png          ← charts
    ├── table1–table5*.png      ← tables
    ├── formula1–formula8*.png  ← rendered formulas
    └── Automatic_Term_Extraction_Infographic.png
```

Figures 2, 3, 4 and Table 3 are computed live from the pipeline; Figures 5, 6 and 7
reproduce published results and name their source in the caption.

---

## Figures

| | |
|---|---|
| <img src="figures/fig1_ate_pipeline.png" width="380"> | <img src="figures/fig2_frequency_vs_cvalue.png" width="380"> |
| **Fig. 1** — the canonical hybrid pipeline | **Fig. 2** — raw frequency vs. C-value |
| <img src="figures/fig3_termhood_plane.png" width="380"> | <img src="figures/fig4_threshold_and_length.png" width="380"> |
| **Fig. 3** — the unithood/termhood plane | **Fig. 4** — two failure modes |
| <img src="figures/fig6_acter_f1_by_system.png" width="380"> | <img src="figures/fig7_llm_vs_plm.png" width="380"> |
| **Fig. 6** — ATE performance on ACTER | **Fig. 7** — where LLMs win and where they don't |

---

## References

- **Frantzi, K., Ananiadou, S., & Mima, H.** (2000). Automatic recognition of multi-word terms:
  the C-value/NC-value method. *International Journal on Digital Libraries*, 3(2), 115–130.
  [doi:10.1007/s007999900023](https://doi.org/10.1007/s007999900023)
- **Rigouts Terryn, A., Hoste, V., & Lefever, E.** (2019). In no uncertain terms: a dataset for
  monolingual and multilingual automatic term extraction from comparable corpora.
  *Language Resources and Evaluation*, 54(2), 385–418.
  [doi:10.1007/s10579-019-09453-9](https://doi.org/10.1007/s10579-019-09453-9)
- **Rigouts Terryn, A., Hoste, V., Drouin, P., & Lefever, E.** (2020). TermEval 2020: Shared Task
  on Automatic Term Extraction Using the ACTER Dataset. *COMPUTERM 2020 @ LREC*, 85–94.
  [aclanthology.org/2020.computerm-1.12](https://aclanthology.org/2020.computerm-1.12/)
- **Tran, H. T. H., Martinc, M., Caporusso, J., Doucet, A., & Pollak, S.** (2023). The Recent
  Advances in Automatic Term Extraction: A survey.
  [arXiv:2301.06767](https://arxiv.org/abs/2301.06767)
- **Banerjee, S., Chakravarthi, B. R., & McCrae, J. P.** (2024). Large Language Models for
  Few-Shot Automatic Term Extraction. *NLDB 2024*, LNCS 14762.
  [doi:10.1007/978-3-031-70239-6_10](https://doi.org/10.1007/978-3-031-70239-6_10)
- **Chun, Y., Kim, M., Kim, D., Park, C., & Lim, H.** (2025). Enhancing Automatic Term
  Extraction with Large Language Models via Syntactic Retrieval. *Findings of ACL 2025*.
  [aclanthology.org/2025.findings-acl.516](https://aclanthology.org/2025.findings-acl.516/)
- **Xu, K., Feng, Y., Li, Q., Dong, Z., & Wei, J.** (2025). Survey on terminology extraction
  from texts. *Journal of Big Data*.
  [doi:10.1186/s40537-025-01077-x](https://doi.org/10.1186/s40537-025-01077-x)
- **Blandón Andrade, J. C., et al.** (2026). Approaches, Tools, Algorithms, and Methods for
  Automatic Term Extraction: A Systematic Literature Mapping.
  [doi:10.1177/18758967251392652](https://doi.org/10.1177/18758967251392652)
- **Tran, H. T. H., et al.** (2026). Recent Advances in Automatic Term Extraction:
  A Comprehensive Survey. *ACM Computing Surveys*.
  [doi:10.1145/3787584](https://doi.org/10.1145/3787584)

**Dataset** — ACTER: [github.com/AylaRT/ACTER](https://github.com/AylaRT/ACTER) (CC BY-NC-SA 4.0)
**Library** — PyATE: [github.com/kevinlu1248/pyate](https://github.com/kevinlu1248/pyate)

---

## Licence

Code is released under the [MIT Licence](LICENSE). The cited papers and the ACTER dataset
remain under their own terms.

---

<div align="center">

**[📖 Read the article](https://medium.com/@huseyinceniik/automatic-term-extraction-the-research-field-behind-your-keyword-extractor-f212338829b8)** ·
**[📓 Run the notebook](https://www.kaggle.com/code/huseyincenik/automatic-term-extraction)** ·
**[💼 LinkedIn](https://www.linkedin.com/in/huseyincenik/)**

If this was useful, a ⭐ helps other people find it.

</div>
