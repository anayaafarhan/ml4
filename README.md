# Used Car Price Estimator

Task 04 of my AI/ML internship at Devixo Solutions — an end-to-end ML app,
not just a notebook. You give it a car's details (brand, age, mileage, fuel
type, city, how you'd sell it) and it estimates what that car would go for,
using a model trained on real OLX Pakistan listings.

I picked OLX Pakistan over the usual tutorial datasets because the data is
genuinely messy in ways I could actually reason about — missing cities,
new and used listings mixed together, a "Range Rover" vs. "Land Rover"
brand split, price entries with an extra zero or two — rather than
something already scrubbed clean for you.

## What's in here

```
data/                  raw dataset (OLX Pakistan used-car listings, ~25,000 rows)
notebooks/              EDA — where I actually worked out the cleaning rules
src/data_processing.py  cleaning + feature engineering, shared by everything else
src/train.py            trains 4 models, tunes the best one, saves the pipeline
models/                 trained pipeline + metadata (brand/city lists, ranges, metrics)
reports/                model comparison table + a few EDA plots
app.py                  the Streamlit app
```

## The data

[OLX Pakistan used-car listings](https://github.com/pintuiitbhi/car-price-pred)
(originally scraped for a Kaggle dataset, "used-cars-data-pakistan") — brand,
model, condition, fuel, kilometers driven, price, registered city, and
transaction type (cash vs. installment/leasing). Prices in PKR.

Cleaning steps (all in `src/data_processing.py`, worked out in
`notebooks/exploration.ipynb`):

- Dropped rows missing brand, condition, fuel, mileage, price, transaction
  type, or year — around 2,400 rows, mostly incomplete ad scrapes.
- Dropped ~3,800 duplicate rows.
- Kept only `Condition == "Used"` — the dataset mixes new and used
  listings, and a new car's price follows completely different rules than
  resale depreciation, so it's out of scope for a *used* car estimator.
- Merged "Range Rover" into "Land Rover" (it's a model line, not a separate
  manufacturer) and folded the site's own "Other Brands" catch-all into
  the "Other" bucket this project already builds for rare brands.
- Converted `Year` into `car_age` (relative to 2019, the most recent
  listing year in the data).
- Filled missing `Registered City` with "Not Specified" rather than
  dropping those rows — a car's price doesn't depend on whether the seller
  filled in that field.
- Trimmed outliers on price, mileage, and car age using a widened IQR rule
  (3× IQR rather than the usual 1.5×) — this dataset has real data-entry
  errors (a "car" priced above 8 crore PKR, a listing with model year
  1915, mileages past a million km), but also genuine expensive cars and
  near-new low-mileage listings that a tighter cutoff would have wrongly
  cut too.
- Bucketed brands with fewer than 5 listings and cities with fewer than 10
  into "Other" / "Other City".

15,650 listings survive cleaning, down from just under 25,000.

## Models

Four regressors, compared on a held-out 20% test split:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | Rs 2,00,068 | Rs 2,89,925 | 0.691 |
| Decision Tree | Rs 1,72,799 | Rs 2,96,663 | 0.676 |
| Random Forest | Rs 1,51,231 | Rs 2,50,804 | 0.768 |
| **Gradient Boosting** | Rs 1,49,463 | Rs 2,45,305 | 0.779 |

Gradient Boosting won, then got tuned with `RandomizedSearchCV` (25
iterations, 5-fold CV) over learning rate, tree depth, subsample ratio, and
estimator count:

| | RMSE | R² |
|---|---|---|
| Before tuning | Rs 2,45,305 | 0.779 |
| After tuning | Rs 2,38,830 | 0.790 |

Best params: `n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.7`.

Full numbers are in `reports/model_comparison.csv`. R² of 0.79 is decent —
brand, age, mileage, fuel, city, and sale type explain most of the price,
but things like actual condition, accident history, and interior/exterior
wear aren't in this dataset and clearly matter too. The app shows a margin
of error alongside every prediction instead of presenting the number as
exact.

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
point it at this repo (`main` branch), entry point `app.py`. No secrets or
extra config needed — `requirements.txt` covers everything, and the model
is committed to the repo so there's no separate training step at deploy
time.

## Notes

- `notebooks/exploration.ipynb` is genuinely exploratory — it's where I
  decided on the cleaning rules before writing `src/data_processing.py`,
  not a write-up produced after the fact.
- The reference year for `car_age` (2019) is fixed to the dataset's most
  recent listing year, not the current date — so a "2 year old car" always
  means "made around 2017" as far as the model's concerned, regardless of
  when someone actually uses the app.
- Karachi makes up roughly three-quarters of all listings in this dataset,
  so the model has a lot more signal for Karachi-registered cars than for
  smaller cities — worth keeping in mind when reading a prediction for a
  city outside the top few.

— Anaya Farhan
