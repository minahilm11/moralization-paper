This is a repository for storing the code for the working paper "Strategic Moralization in Parliamentary Debates" (Malik et al., 2026)

The SWERIK Riksdag records data used for the analysis is available: https://github.com/swerik-project/riksdagen-records/releases#release-v1.4.2
The SWERIK Riksdag persons data used for the analysis is available: https://github.com/swerik-project/riksdagen-persons/releases/tag/v1.2.1

To parse the XML data and link the speeches with the applicable metadata to construct the full dataset used for the analysis, run data/create_dataset.py and make sure the data is placed under the data/ directory.

Running descriptives & Regression analysis: scripts/moralisation_analysis.ipynb

## Run

Python 3.12. The analysis data and saved R&R labels are included.

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1` (Windows).

```sh
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name moralization --display-name Moralization
python reproduce.py
```

Or open the notebooks in Jupyter and run all cells with the **Moralization** kernel. Figures and tables are saved under `results/`.

| Notebook | Paper outputs |
| --- | --- |
| `scripts/moralisation_analysis.ipynb` | Main Figures 2-5, Table 2; Appendix A5-A8 |
| `scripts/analyze_clusters.ipynb` | Main Figure 1; Appendix Figures A3.1-A3.2 |
| `scripts/descriptives.ipynb` | Appendix Tables A1.1-A1.3 |
| `appendix/validation.ipynb` | Appendix Tables A4.1/A9.1 and Figures A4.1-A4.2 |
| `appendix/examples.ipynb` | Main Table 1; Appendix Tables A2.1/A3.1 |
| `rr/r_r_clustering_visuals.ipynb` | Additional R&R clustering and topic/style visuals |

**Policy-topic labels are not included.** Main Figure 5, Appendix A5, A6.2-A6.4 and A7, and the topic-controlled columns of Table 2 require `data/policy_topics.parquet` (`original_row_idx`, `topic_label`) or the original `data/data_with_topics.pkl`. These sections are skipped until the labels are supplied. Only load pickle files from a trusted source.

## Classification and clustering

`scripts/topic_classification.ipynb` contains the eleven-topic speech classifier. Install `requirements-topics.txt` and place the full speech-text file at `data/new_prompt_noshot_4omini_analysis_input.parquet`, or set `MORALIZATION_INPUT` to its path. New labels may change the topic-controlled estimates.

The R&R prompt, batch runner, clustering and taxonomy-labeling scripts are under `rr/`. Install `requirements-rr.txt` for new model runs; these are not executed by `reproduce.py` and can incur API costs.

```sh
python rr/replication/scripts/cluster_reasoning.py --sample rr/artifacts/cluster_assignments_with_original_idx.parquet --output results/new_clusters --cache-dir /path/to/embedding_cache
python rr/replication/scripts/label_reasoning.py --input rr/artifacts/classification_dataset_topic_style_50k.parquet --taxonomy rr/taxonomy.json --config rr/labeling.json --output results/new_labels --allow-api
```

For the XML data-building scripts, install `requirements-data.txt`. `data/speeches.parquet` contains the ratings and metadata; speech text for the 50,000-trace sample is included in `rr/artifacts/classification_dataset_topic_style_50k.parquet`.
