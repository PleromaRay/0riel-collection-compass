# Collections Compass: risk model, contact strategy and daily call list

Oriel Credit group project. This folder holds the data scientist's part of the project:

- **C1 Model features**, **C2 Default prediction model**, **C3 Model evaluation and fairness**, **C4 Contact strategy**
- an **automated daily call list** for the 24 collections agents
- a comparison with today's alphabetical call order, as an input to the team's recovery uplift

All numbers come from the 14 files in `data/`. The data end on 4 November 2026.

## What is in the folder

| File or folder | What it is |
|---|---|
| `Collections_Compass_Model.ipynb` | The notebook, Steps 0 to 18, already run (runs top to bottom with no errors) |
| `Collections_Compass_Model_Report.docx` | The written report (C1 to C4, call list, uplift input) |
| `Collections_Compass_Presenter_Guide.docx` | Plain-English guide: every step explained, key ideas, slide storyline, likely questions |
| `compass_core.py` | Cleaning and feature code shared by the scripts (same logic as the notebook) |
| `train_and_save_models.py` | Rebuilds the models and saves them in `models/` |
| `contact_strategy.py` | The contact rules (C4) as code |
| `daily_call_list.py` | Builds the daily call list |
| `backtest.py` | Replays past days to compare the new list with alphabetical order |
| `models/` | Saved models (`model_B_day1.joblib`, `model_A_due_date.joblib`) and `model_card.json` |
| `outputs/` | All result tables, `summary_numbers.json` and `summary_numbers_part2.json` |
| `outputs/figures/` | Charts used in the report |
| `outputs/call_lists/` | Dated call lists, digital contact lists and run summaries |
| `.github/workflows/daily_call_list.yml` | Optional GitHub Action to build the list every morning |
| `cleaning/` | The script that cleans the raw customers file (not needed for the model) |
