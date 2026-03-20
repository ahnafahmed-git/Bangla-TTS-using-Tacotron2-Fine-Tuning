from IPython.display import display as ipy_display, HTML

class ExperimentLogger:
    """
    Formats and displays the two logging tables
    required by the project spec:
      - TTSExperiments    : training runs + metrics
      - GeneratedSamples  : generated audio log
    """

    @staticmethod
    def display_experiments(experiment_log: list, eval_metrics: dict = None):
        """
        Renders TTSExperiments table.
        eval_metrics: optional dict {experiment_id: {mcd, pesq, stoi, mosnet}}
        """
        rows = []
        for e in experiment_log:
            row = {
                "ID"            : e["id"],
                "Model"         : e["model_name"],
                "LR"            : e["hyperparameters"]["lr"],
                "Batch Size"    : e["hyperparameters"]["batch_size"],
                "Train Loss"    : e["train_loss"],
                "Val Loss"      : e["val_loss"],
                "MCD"           : None,
                "PESQ"          : None,
                "STOI"          : None,
                "MOSNet"        : None,
                "Epoch Time (s)": e["epoch_time_sec"],
                "Timestamp"     : e["timestamp"][:19],
            }
            if eval_metrics and e["id"] in eval_metrics:
                m = eval_metrics[e["id"]]
                row["MCD"]    = m.get("mcd")
                row["PESQ"]   = m.get("pesq")
                row["STOI"]   = m.get("stoi")
                row["MOSNet"] = m.get("mosnet")
            rows.append(row)

        df = pd.DataFrame(rows)
        print("\n" + "="*60)
        print("  TTSExperiments Table")
        print("="*60)
        ipy_display(df)
        return df

    @staticmethod
    def display_generated_samples(generated_samples: list):
        """
        Renders GeneratedSamples table.
        """
        rows = []
        for s in generated_samples:
            metrics = s.get("metrics", {})
            rows.append({
                "ID"           : s["id"],
                "Experiment ID": s["experiment_id"],
                "Text Input"   : s["text_input"][:50] + "..." if len(s["text_input"]) > 50 else s["text_input"],
                "Duration (s)" : s["duration_sec"],
                "MCD"          : metrics.get("mcd"),
                "PESQ"         : metrics.get("pesq"),
                "STOI"         : metrics.get("stoi"),
                "MOSNet"       : metrics.get("mosnet"),
                "Audio URL"    : s["audio_url"],
                "Timestamp"    : s["timestamp"][:19],
            })

        df = pd.DataFrame(rows)
        print("\n" + "="*60)
        print("  GeneratedSamples Table")
        print("="*60)
        ipy_display(df)
        return df

    @staticmethod
    def save_tables_to_drive(exp_df, samples_df, base_dir: str):
        """Save both tables as CSVs to Drive for documentation."""
        exp_path     = f"{base_dir}/TTSExperiments.csv"
        samples_path = f"{base_dir}/GeneratedSamples.csv"
        exp_df.to_csv(exp_path,     index=False)
        samples_df.to_csv(samples_path, index=False)
        print(f"Saved TTSExperiments.csv    → {exp_path}")
        print(f"Saved GeneratedSamples.csv  → {samples_path}")
