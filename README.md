CIMIS: experimento reproducible (FORCE 2020)
Código del artículo "Arquitectura computacional híbrida para inversión sísmica: validación con registros de pozo".
Datos
Registros públicos FORCE 2020 (Bormann et al., 2020, https://doi.org/10.5281/zenodo.4351156, CC BY 4.0).
Clonar https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition y ajustar `DATA` en `src/build_dataset.py`
a la carpeta `lithology_competition/data/` (descomprimir `train.zip`).
Entorno
Python 3.13; ver `requirements.txt`.
Ejecución
```bash
cd src
python run_experiment.py --mode acoustic --tag ac_base --prepare      # Experimento 1
python run_experiment.py --mode elastic  --tag el_base --prepare      # Experimento 2 (con búsqueda de hiperparámetros)
bash run_sensitivity.sh                                               # sensibilidad y 4 semillas adicionales
python make_figures.py                                                # figuras 1-6 (300 dpi)
```
Cada corrida escribe en `results/`: `summary_<tag>.json` (configuración, pozos, métricas, pruebas),
`preds_<tag>.parquet` (predicciones P10/P50/P90 por muestra) y `data_<tag>.parquet` (atributos invertidos).
`seed_summary_elastic.csv` contiene media y desviación estándar de las cinco realizaciones de ruido (semillas 2026-2030).
Módulos
`src/cimis_core.py`: control de calidad, petrofísica, conversión a tiempo, modelado (acústico y Aki-Richards/Fatti),
inversión bayesiana linealizada, atributos.
`src/build_dataset.py`: construcción de la tabla de 1 ms por pozo.
`src/run_experiment.py`: métodos REF, LIN-AI, LIN, QGB-L, CIMIS-ML, CIMIS-ML+CQR; validación agrupada por pozo; pozos ciegos; pruebas.
