# Collections Compass: risk model, contact strategy and daily call list

Oriel Credit group project. This folder holds the data scientist's part of the project:

- **C1 Model features**, **C2 Default prediction model**, **C3 Model evaluation and fairness**, **C4 Contact strategy**
- an **automated daily call list** for the 24 collections agents
- a comparison with today's alphabetical call order, as an input to the team's recovery uplift

## What is in the folder

| File or folder | What it is |
|---|---|
| `Collections_Compass_Model.ipynb`
| `compass_core.py`
| `train_and_save_models.py`
| `contact_strategy.py`
| `daily_call_list.py`
| `backtest.py` 
| `models/` | Saved models (`model_B_day1.joblib`, `model_A_due_date.joblib`) and `model_card.json` |
| `outputs/` | All result tables, `summary_numbers.json` and `summary_numbers_part2.json` |
| `outputs/figures/` | Charts used in the report |
| `outputs/call_lists/` | Dated call lists, digital contact lists and run summaries |