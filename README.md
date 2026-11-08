# House Price Prediction

Project for estimating listed residential property prices from the supplied dataset. The app reports estimates in lakhs and includes a dataset explorer. It is an educational model and must not be used as a certified valuation.

## Open and run

1. Open this `House_Price_Prediction` folder in VS Code.
2. Create and activate a virtual environment in the integrated terminal:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Train the model (this writes `models/house_price_pipeline.joblib` and `reports/model_metrics.json`):

   ```powershell
   python src/train_model.py
   ```

4. Start Streamlit:

   ```powershell
   streamlit run app.py
   ```

Use the project virtual environment when launching the app. Streamlit uses PyArrow to convert chart and table data; mixing Conda and pip copies of this native package in a shared Anaconda base environment can cause Windows DLL import errors. The pinned PyArrow wheel is compatible with Python 3.13 and bundles the Arrow native libraries.

If Streamlit is already running from Anaconda base, stop it, create the project environment above, install the updated `requirements.txt`, and launch Streamlit from that activated environment. In VS Code, select `.venv\Scripts\python.exe` as the interpreter.

## Project map

- `app.py` — Streamlit estimator, data explorer, and model notes.
- `src/train_model.py` — deterministic split, preprocessing, model comparison, metrics, and final artifact export.
- `data/house_price.csv` — source data copy.
- `models/house_price_pipeline.joblib` — fitted best pipeline from the fixed holdout experiment.
- `reports/model_metrics.json` — evaluation metrics and data summary.
- `reports/Model_Development_Document.docx` — MDD built around the supplied report template.
- `reports/*_source.docx` — original supplied data description and template retained for project traceability.

## Reproducibility and limitations

The split uses `random_state=42` and a 20% test set. The target is transformed with `log1p` for model fitting and returned to lakhs for evaluation. The dataset has no timestamp or geographic grouping key beyond address and coordinates, so this random holdout does not prove generalization to future sales, unseen markets, or new neighborhoods. The source data has extreme area and coordinate values; inspect `reports/Model_Development_Document.docx` before interpreting estimates. Retrain after changing data or dependencies. Scikit-learn joblib files should only be loaded from a trusted source and within compatible dependency versions.
