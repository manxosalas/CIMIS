#!/usr/bin/env bash
# Análisis de sensibilidad y repetibilidad (modo elástico, hiperparámetros fijos)
set -e
P=../results/best_params_elastic.json
for s in 2 8 inf; do python3 run_experiment.py --mode elastic --tag el_snr$s --snr $s --no-tune --params $P > ../results/log_el_snr$s.txt 2>&1; done
python3 run_experiment.py --mode elastic --tag el_phase30 --phase_err 30 --no-tune --params $P > ../results/log_el_phase30.txt 2>&1
python3 run_experiment.py --mode elastic --tag el_f0err --f0_err 0.85 --no-tune --params $P > ../results/log_el_f0err.txt 2>&1
for f in 2 8; do python3 run_experiment.py --mode elastic --tag el_flf$f --f_lf $f --no-tune --params $P > ../results/log_el_flf$f.txt 2>&1; done
for sd in 2027 2028 2029 2030; do python3 run_experiment.py --mode elastic --tag el_seed$sd --seed $sd --no-tune --params $P > ../results/log_el_seed$sd.txt 2>&1; done
echo DONE
