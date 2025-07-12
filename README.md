# ViLLA-MMBench

Welcome to the **ViLLA-MMBench** repository! This project provides the source code and fully reproducible results for our upcoming paper submitted to **RecSys'25**.

## 📦 What's Included?

- ✅ Source code to reproduce experiments
- 📄 Recommendation result files for all model variants
- 🔁 Benchmarks using visual, audio, and textual modalities
- 📊 Evaluation metrics including accuracy and beyond-accuracy (BA) metrics

## 🚀 Using the Framework

- Clone the current repository using `git@github.com:RecSys-lab/ViLLA-MMBench.git`
- Create and activate a virtual environment using `python -m venv venv` and then `.\venv\Scripts\activate` (Windows)
- Install the packages using `pip install -e .` (running `setup.py` file)
- Check the configurations required for running the experiments in [villa_mmbench/config.yml](/villa_mmbench/config.yml)
- Run the framework by running [villa_mmbench/main.py](/villa_mmbench/main.py)!

## 📂 Folders and Files

- **Colabs**
  - `villa_mmbench.ipynb`: the primary toolkit file containing all functions and configurations
  - `rank_aggregation.ipynb`: functions for rank aggregation
  - `data_visualization.ipynb`: procedures to visualize processed data
- **RecList**: contains the list of generated recommendation lists

## 📚 Citation

```bibtex
@article{villammbench,
  title={ViLLA-MMBench: A Unified Benchmark Suite for LLM-Augmented Multimodal Movie Recommendation},
  author={TBD},
  journal={TBD},
  year={2025}
}
```

## 📬 Contact

If you have any questions or collaboration opportunities, please open an issue or contact the authors.
