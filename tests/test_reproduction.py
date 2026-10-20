from pathlib import Path
import unittest

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


class ReproductionTests(unittest.TestCase):
    def test_notebook_sources(self):
        for folder in ["scripts", "appendix"]:
            for path in (ROOT / folder).rglob("*.ipynb"):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for i, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        compile(cell.source, f"{path.name}:{i}", "exec")
                        self.assertFalse(cell.outputs)

    def test_sample_and_text_join(self):
        corpus = pd.read_parquet(ROOT / "data" / "speeches.parquet")
        sample = pd.read_parquet(ROOT / "data/reasoning/classification_dataset_topic_style_50k.parquet")
        self.assertEqual(len(corpus), 427014)
        self.assertEqual(len(sample), 50000)
        self.assertTrue(sample.original_row_idx.is_unique)
        self.assertTrue(sample.text.notna().all())
        expected = corpus.set_index("original_row_idx").moral_rating.reindex(sample.original_row_idx)
        self.assertEqual(sample.rating.tolist(), expected.tolist())
        self.assertEqual(len({label for labels in sample.style_labels for label in labels}), 23)

    def test_corpus_results(self):
        yearly = pd.read_csv(TABLES / "table_a1_3_year.csv")
        self.assertEqual(yearly.Speeches.sum(), 395943)
        overall = pd.read_csv(TABLES / "table_a1_2.csv").set_index("Measure").Value
        self.assertEqual(overall["Unique speakers"], "1,707")
        self.assertEqual(overall["Mean speech length, words"], "343.0")

    def test_policy_topics(self):
        corpus = pd.read_parquet(ROOT / "data" / "speeches.parquet")
        topics = pd.read_parquet(ROOT / "data" / "policy_topics.parquet")
        self.assertEqual(topics.columns.tolist(), ["original_row_idx", "topic_label"])
        self.assertTrue(topics.original_row_idx.is_unique)
        self.assertEqual(sorted(topics.original_row_idx), sorted(corpus.original_row_idx))
        self.assertTrue(topics.topic_label.notna().all())
        self.assertEqual(topics.topic_label.nunique(), 11)

    def test_topic_tables(self):
        distribution = pd.read_csv(TABLES / "table_a5_1.csv")
        self.assertEqual(distribution.Count.sum(), 395943)
        self.assertEqual(len(distribution), 11)
        for name in ["table_2.tex", "table_a5_1.tex", "table_a6_2.tex",
                     "table_a6_3.tex", "table_a6_4.tex", "table_a7_1.tex"]:
            self.assertGreater((TABLES / name).stat().st_size, 100)

    def test_model_results(self):
        monthly = pd.read_csv(TABLES / "mp_month_coefficients.csv", index_col=0)
        self.assertEqual(round(monthly.loc["gov_status", "estimate"], 3), -.418)
        self.assertEqual(round(monthly.loc["log_months_to_election", "estimate"], 3), -.035)
        party = pd.read_csv(TABLES / "party_control_coefficients.csv", index_col=0)
        panel = pd.read_csv(TABLES / "mp_fixed_effects.csv", index_col=0)
        self.assertEqual(round(party.loc["gov_status", "estimate"], 3), -.447)
        self.assertEqual(round(panel.loc["gov_status", "estimate"], 3), -.435)

    def test_rating_agreement(self):
        models = pd.read_csv(TABLES / "table_a4_1.csv")
        self.assertTrue(models.n.eq(20000).all())
        self.assertEqual(round(models.iloc[0].quadratic_weighted_kappa, 3), .613)
        human = pd.read_csv(TABLES / "table_a9_1.csv").set_index("comparison")
        self.assertEqual(human.loc["Expert 2 vs LLM", "n"], 99)
        self.assertAlmostEqual(human.loc["Expert 2 vs LLM", "quadratic_weighted_kappa"], .7346000883782591)

    def test_paper_figures(self):
        for name in ["parties.jpg", "rolling.jpg", "pooled.jpg", "average_moral_rating_by_frame.png",
                     "frame_prevalence_vs_morality.png", "keyness_panels_part1.png", "keyness_panels_part2.png",
                     "equal_length_bins_two_years_before_election.jpg",
                     "model_rating_distributions_by_model_color.png", "yearly_average_model_ratings.png",
                     "predicted_speech_level_two_models.jpg", "gov-opp.jpg", "opp-gov.jpg"]:
            self.assertGreater((ROOT / "results" / "figures" / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
