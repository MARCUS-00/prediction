## Run Order

# Step 1 — Data Collection

python data_collection/build_technical.py
python data_collection/build_fundamental.py
python data_collection/build_news.py
python data_collection/build_events.py

# Step 2 — Merge (handles news + lag + rolling + rename internally)

python features/merge_features.py

# Step 3 — Train

python models/xgboost/train.py
python models/lstm/train.py
python models/ensemble/train_meta.py

# Step 4 — Predict

python prediction/single_stock.py INFY
python prediction/watchlist.py

# Step 5 — App

streamlit run app/app.py
