# ViLLA-MMBench

## ⚙️ Configuration Table (`config.yml`)

| Section         | Parameter          | Description                                             | Options / Values           | Default           |
| --------------- | ------------------ | ------------------------------------------------------- | -------------------------- | ----------------- |
| **General**     | `name`             | Name of the framework                                   | -                          | `"ViLLA-MMBench"` |
| **Experiment**  | `fast_prototype`   | Run a quick prototype with 1 epoch                      | `true`, `false`            | `true`            |
|                 | `use_gpu_for_hpo`  | Use GPU for hyperparameter optimization                 | `true`, `false`            | `false`           |
|                 | `parallel_hpo`     | Enable parallel HPO                                     | `true`, `false`            | `true`            |
|                 | `seed`             | Random seed for reproducibility                         | Integer                    | `42`              |
|                 | `verbose`          | Enable verbose logging                                  | `true`, `false`            | `true`            |
|                 | `n_epochs`         | Number of training epochs (ignored if `fast_prototype`) | Integer                    | `20`              |
| **Modality**    | `model_choice`     | Model to use                                            | `cf`, `vbpr`, `amr`, `vmf` | `"cf"`            |
|                 | `llm_prefix`       | LLM for text processing                                 | `openai`, `st`, `llama`    | `"llama"`         |
|                 | `text_augmented`   | Use augmented textual input                             | `true`, `false`            | `true`            |
|                 | `audio_variant`    | Audio embedding variant                                 | `blf`, `i_ivec`            | `"blf"`           |
|                 | `visual_variant`   | Visual embedding variant                                | `avf`, `cnn`               | `"cnn"`           |
|                 | `text_max_parts`   | Max number of text parts to include                     | Integer                    | `15`              |
| **Data**        | `ml_version`       | MovieLens dataset version                               | `100k`, `1m`               | `"1m"`            |
|                 | `split.mode`       | Data split strategy                                     | `random`, `sequential`     | `"random"`        |
|                 | `split.test_ratio` | Ratio of test samples                                   | Float (0–1)                | `0.2`             |
|                 | `k_core`           | Min interactions per user/item (k-core filtering)       | Integer                    | `10`              |
| **Recommender** | `topN_k`           | Top-N cutoff for evaluation                             | Integer                    | `10`              |
|                 | `cold_threshold`   | Threshold for "cold" items/users (≤ value)              | Integer                    | `5`               |
