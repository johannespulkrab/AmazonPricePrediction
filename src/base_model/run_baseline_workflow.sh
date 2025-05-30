#!/bin/bash

set -e

source ../../.venv/bin/activate

PROCESSED_DIR="../../data/processed"
FINAL_OUTPUT="../../outputs/baseline_final"

if [[ ! -f "train.py" ]]; then
    echo "Error: Run from src/base_model/ directory"
    exit 1
fi

mkdir -p "$FINAL_OUTPUT"

echo "Training baseline models with nested splits..."

echo "Step 1/3: Training Linear Regression"
python train.py --processed_dir "$PROCESSED_DIR" --model_type linear --output_dir "$FINAL_OUTPUT/linear"

echo "Step 2/3: Training Ridge Regression (alpha=1.0)"
python train.py --processed_dir "$PROCESSED_DIR" --model_type ridge --alpha 1.0 --output_dir "$FINAL_OUTPUT/ridge"

echo "Step 3/3: Training Lasso Regression (alpha=1.0)"
python train.py --processed_dir "$PROCESSED_DIR" --model_type lasso --alpha 1.0 --output_dir "$FINAL_OUTPUT/lasso"

echo "Creating comparison report..."

python -c "
import json
import os

results = {}
for model in ['linear', 'ridge', 'lasso']:
    try:
        with open('$FINAL_OUTPUT/' + model + '/results.json') as f:
            results[model] = json.load(f)['test_metrics']
    except FileNotFoundError:
        print(f'Warning: Results for {model} not found')
        continue

print('Baseline Model Comparison:')
print('=' * 40)
for model, metrics in results.items():
    print(f'{model.upper():>8}: RMSE={metrics[\"rmse\"]:.4f}, MAE={metrics[\"mae\"]:.4f}, R²={metrics[\"r2\"]:.4f}')

if results:
    best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
    print(f'Best Model: {best_model.upper()} (lowest RMSE)')

    with open('$FINAL_OUTPUT/comparison.json', 'w') as f:
        json.dump({'models': results, 'best': best_model}, f, indent=2)

    print(f'Comparison saved to: $FINAL_OUTPUT/comparison.json')
else:
    print('No results found for comparison')
"

echo "Done. Results in $FINAL_OUTPUT/"