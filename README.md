# Used Car Price Estimator

Task 04 of my AI/ML internship at Devixo Solutions — an end-to-end ML app,
not just a notebook. You give it a car's details (brand, age, mileage, fuel
type, etc.) and it estimates what that car would sell for, using a model
trained on real CarDekho listings.

I picked the used-car project out of the suggested list because the data
is genuinely messy in ways I could actually reason about — mangled brand
names, duplicate listings, a "Test Drive Car" ownership category that
doesn't behave like a normal sale — rather than a dataset that's already
been scrubbed for you.

## What's in here

```
data/                 raw dataset (CarDekho used-car listings, ~4,340 rows)
notebooks/             EDA — where I actually worked out the cleaning rules
src/data_processing.py cleaning + feature engineering, shared by everything else
src/train.py            trains 4 models, tunes the best one, saves the pipeline
models/                 trained pipeline + metadata (brand lists, ranges, metrics)
reports/                model comparison table + a few EDA plots
app.py                  the Streamlit app
```

## The data

[CarDekho used-car listings](https://github.com/chandanverma07/DataSets) —
name, year, selling price, kilometers driven, fuel type, seller type,
transmission, and ownership history. Scraped in 2020, prices in INR.

Cleaning steps (all in `src/data_processing.py`, worked out in
`notebooks/exploration.ipynb`):

- Dropped 763 duplicate rows.
- Extracted brand from the car name, fixed two mangled entries ("Land" →
  "Land Rover", "OpelCorsa" → "Opel"), and bucketed brands with fewer than
  5 listings into "Other" so the model isn't trying to learn from single
  examples.
- Converted `year` into `car_age` (relative to 2020, the year the data was
  collected).
- Dropped "Test Drive Car" listings — these are dealer demo units and their
  pricing doesn't follow normal resale depreciation.
- Trimmed outliers on price and mileage using a widened IQR rule (3× IQR
  rather than the usual 1.5×), so legitimate expensive cars and near-new
  low-mileage listings don't get cut along with genuine data errors.

3,458 listings survive cleaning, down from 4,340.

## Models

Four regressors, compared on a held-out 20% test split:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ₹1,30,153 | ₹1,77,740 | 0.632 |
| Decision Tree | ₹1,47,123 | ₹2,25,859 | 0.406 |
| Random Forest | ₹1,18,655 | ₹1,73,626 | 0.649 |
| **Gradient Boosting** | ₹1,12,470 | ₹1,59,263 | 0.705 |

Gradient Boosting won, then got tuned with `RandomizedSearchCV` (25
iterations, 5-fold CV) over learning rate, tree depth, subsample ratio, and
estimator count:

| | RMSE | R² |
|---|---|---|
| Before tuning | ₹1,59,263 | 0.705 |
| After tuning | ₹1,57,871 | 0.710 |

Best params: `n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.85`.

Full numbers are in `reports/model_comparison.csv`. Worth being honest
about: R² of 0.71 is decent, not spectacular — condition, service history,
and location aren't in this dataset, and those matter a lot for what a car
actually sells for. The app shows a margin of error alongside every
prediction rather than pretending the number is exact.

The final pipeline (preprocessing + tuned model, so the app doesn't need to
duplicate any cleaning logic) is saved to `models/car_price_model.pkl` with
`joblib`.

## Running it

```bash
pip install -r requirements.txt

# retrain from scratch (optional — a trained model is already committed)
python src/train.py

# launch the app
streamlit run app.py
```

## Deployment

Set up for [Streamlit Community Cloud](https://share.streamlit.io):
point it at this repo, branch `claude/human-touch-project-5yvlva` (or `main`
once merged), entry point `app.py`. No secrets or extra config needed —
`requirements.txt` covers everything, and the model is committed to the
repo so there's no separate training step at deploy time.

## Notes

- `notebooks/exploration.ipynb` is genuinely exploratory — it's where I
  decided on the cleaning rules before writing `src/data_processing.py`,
  not a write-up produced after the fact.
- The reference year for `car_age` (2020) is fixed to when the dataset was
  collected, not the current date — so a "2 year old car" always means
  "made around 2018" as far as the model's concerned, regardless of when
  someone actually uses the app.

— Anaya Farhan
