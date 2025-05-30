#!/bin/bash

set -e

# Activate virtual environment
source ../../.venv/bin/activate

PROCESSED_DIR="../../data/processed"
TUNING_OUTPUT="../../outputs/baseline_tuning"
FINAL_OUTPUT="../../outputs/baseline_final"

if [[ ! -f "train.py" ]] || [[ ! -f "tune_hyperparams.py" ]]; then
    echo "Error: Run from src/base_model/ directory"
    exit 1
fi

mkdir -p "$TUNING_OUTPUT" "$FINAL_OUTPUT"

echo "Step 1/4: Quick tuning"
python tune_hyperparams.py --processed_dir "$PROCESSED_DIR" --use_subset --subset_ratio 0.3 --model_types linear ridge lasso --output_dir "$TUNING_OUTPUT/quick"

echo "Step 2/4: Full tuning"
python tune_hyperparams.py --processed_dir "$PROCESSED_DIR" --model_types ridge lasso --output_dir "$TUNING_OUTPUT/full"

TUNING_RESULTS="$TUNING_OUTPUT/full/tuning_results.json"
BEST_RIDGE_ALPHA=$(python -c "import json; data=json.load(open('$TUNING_RESULTS')); print(data['model_results']['ridge']['best_alpha'])")
BEST_LASSO_ALPHA=$(python -c "import json; data=json.load(open('$TUNING_RESULTS')); print(data['model_results']['lasso']['best_alpha'])")

echo "Step 3/4: Training models"
python train.py --processed_dir "$PROCESSED_DIR" --model_type linear --output_dir "$FINAL_OUTPUT/linear"
python train.py --processed_dir "$PROCESSED_DIR" --model_type ridge --alpha "$BEST_RIDGE_ALPHA" --output_dir "$FINAL_OUTPUT/ridge"
python train.py --processed_dir "$PROCESSED_DIR" --model_type lasso --alpha "$BEST_LASSO_ALPHA" --output_dir "$FINAL_OUTPUT/lasso"

echo "Step 4/4: Comparison report"

python -c "
import json
import os

results = {}
for model in ['linear', 'ridge', 'lasso']:
    with open('$FINAL_OUTPUT/' + model + '/results.json') as f:
        results[model] = json.load(f)['test_metrics']

for model, metrics in results.items():
    print(f'{model}: RMSE={metrics[\"rmse\"]:.4f}, MAE={metrics[\"mae\"]:.4f}, R²={metrics[\"r2\"]:.4f}')

best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
print(f'Best: {best_model}')

with open('$FINAL_OUTPUT/comparison.json', 'w') as f:
    json.dump({'models': results, 'best': best_model}, f, indent=2)
"

echo "Done. Results in $FINAL_OUTPUT/"