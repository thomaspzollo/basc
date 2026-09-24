#!/usr/bin/env bash
# Step 6 (CPU only): run every experiment on the cached artifacts and write the
# result tables consumed by the notebooks. Each script parallelizes over
# (model, dataset) with processes; BLAS threads are pinned to 16 per process.
source "$(dirname "$0")/config.sh"
R="${BASC_RESULTS_DIR:-results}"
mkdir -p "$R"
E=scripts/experiments

# Table 1, Figures 2 and 3(a), appendix per-model/per-dataset figures.
$PYTHON $E/run_main.py --out_csv "$R/main.csv"
# Table 1: black-box variant (Gemma embeddings).
$PYTHON $E/run_main.py --variant gemma --k_variants 2 5 100 --ridge_k_variants 5 10 20 100 \
  --ssc_k_variants 5 10 20 100 --out_csv "$R/main_blackbox.csv"
# Table 1 / Figure 3(a): BASC-TP.
$PYTHON $E/run_basc_tp.py --setting in_distribution --out_csv "$R/basc_tp.csv"
# Figure 3(a): 1-stage ablation.
$PYTHON $E/run_one_stage.py --out_csv "$R/one_stage.csv"
# Figure 3(b): Stage-1 estimator.
$PYTHON $E/run_stage1_scaling.py --out_csv "$R/stage1_scaling.csv"
# Figure 3(c): Stage-2 recalibrator.
$PYTHON $E/run_recalibrators.py --out_csv "$R/stage2_recalibrators.csv"
# Table 2 and appendix shift figures.
$PYTHON $E/run_shift.py --out_dir "$R"
$PYTHON $E/run_basc_tp.py --setting shift --out_csv "$R/basc_tp_shift.csv"
# Figure 4 and appendix selection figures.
$PYTHON $E/run_selection.py --out_csv "$R/selection.csv"
# Appendix: probe-temperature ablation (the T=0.5 / 0.9 pools reuse the
# canonical deployed responses).
for cell in "0.5 20" "0.7 100" "0.9 20"; do
  set -- $cell
  extra=""
  [[ "$1" != "0.7" ]] && extra="--test_pool_temp 0.7 --test_pool_k 100"
  $PYTHON $E/run_main.py --models "${TEMP_ABLATION_MODELS[@]}" --pool_temp "$1" --pool_k "$2" $extra \
    --k_variants 2 5 20 --ridge_k_variants 5 20 --ssc_k_variants 5 10 20 100 \
    --out_csv "$R/temp_ablation_T$1_k$2.csv"
done
