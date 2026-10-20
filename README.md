This is a repository for storing the code for the working paper "Strategic Moralization in Parliamentary Debates" (Malik et al., 2026)

The SWERIK Riksdag records data used for the analysis is available: https://github.com/swerik-project/riksdagen-records/releases#release-v1.4.2
The SWERIK Riksdag persons data used for the analysis is available: https://github.com/swerik-project/riksdagen-persons/releases/tag/v1.2.1

To parse the XML data and link the speeches with the applicable metadata to construct the full dataset used for the analysis, run data/create_dataset.py and make sure the data is placed under the data/ directory.

Running descriptives & Regression analysis: scripts/moralisation_analysis.ipynb

## Run

Python 3.12. The analysis data and saved reasoning labels are included.

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
| `scripts/clustering_visuals.ipynb` | Additional clustering and topic/style visuals |

`data/policy_topics.parquet` contains the eleven policy-topic labels for all 427,014 speeches, keyed by `original_row_idx`. These labels are used for Main Figure 5, the topic-controlled columns of Table 2, and Appendix A5, A6.2-A6.4 and A7.

## Classification and clustering

`scripts/classification/` contains the moralization prompt and batch runner. The runner reads `data/speeches_1990_2022.parquet`; merged ratings are saved to `outputs/noshot_4omini_batches/new_prompt_noshot_4omini_ratings_all_available.parquet`.

`scripts/topic_classification.ipynb` reads this merged file using `custom_id` to identify speeches and saves `data/policy_topics.parquet`. Set `MORALIZATION_INPUT` to use another file with the same columns.

`scripts/clustering/` contains the reasoning clustering, taxonomy and labeling code. New API requests require `OPENAI_API_KEY`; these scripts run separately from `reproduce.py`.

```sh
python scripts/clustering/cluster_reasoning.py --sample data/reasoning/cluster_assignments_with_original_idx.parquet --output results/new_clusters --cache-dir /path/to/embedding_cache
python scripts/clustering/label_reasoning.py --input data/reasoning/classification_dataset_topic_style_50k.parquet --taxonomy scripts/clustering/taxonomy.json --config scripts/clustering/labeling.json --output results/new_labels --allow-api
```

`data/speeches.parquet` contains the ratings and metadata; speech text for the 50,000-trace sample is included in `data/reasoning/classification_dataset_topic_style_50k.parquet`.
