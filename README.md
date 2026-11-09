# OWL challenge

Extracción de embeddings con Hibou-L congelado y entrenamiento MIL de supervivencia.

## Uso

Python 3.11, PyTorch con CUDA y un snapshot local de Hibou-L con sus pesos y código:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
python train/extract_all_wsi.py \
  --raw data/data/raw \
  --snapshot /ruta/al/snapshot/hibou-L \
  --output data/data/processed/embeddings/hibou_l/10x_native_224_stride224_tissue005 \
  --level 1 --gpus 0,1,2 --dry-run
```

Quitar `--dry-run` para ejecutar la extracción. La entrada contiene `images/**/*.tif` y `tissue_masks/<nombre>_tissue.tif`. `--level 1` es el valor por defecto y corresponde al ≈10× nativo de estas WSI de OWL.

Por defecto: parches de 224 × 224, stride 224, mínimo 5 % de tejido, lote 1.024 y ocho lectores por GPU. Se ajustan con `--stride`, `--min-tissue`, `--batch-size` y `--workers`. `--dry-run` revisa el inventario sin escribir ni lanzar la extracción.

Cada WSI genera un HDF5 con vectores `[N,2048]` FP16, coordenadas y fracción de tejido. Los logs quedan en `output/logs/`. Repetir el comando omite resultados completos compatibles y retoma parciales. Ctrl+C permite parar conservando los lotes confirmados; evitar cierres forzados. Para dejarlo ejecutando al cerrar la terminal, usar una sesión persistente externa.

Para una sola WSI: `python train/extract_wsi_embeddings.py --help`; `--plan-only` cuenta parches sin extraer embeddings. Datos, pesos y entornos locales quedan fuera de Git.

Pruebas: `python -m unittest discover -s train/tests -v`.

## Split y entrenamiento

Desde la raíz del repositorio, con `.venv` activado y los requirements instalados:

```bash
python train/make_split.py \
  --labels data/data/raw/labels/training_labels.csv \
  --output data/training/splits/60_20_20_seed42.txt --seed 42
```

```bash
python train/train_mil.py \
  --labels data/data/raw/labels/training_labels.csv \
  --split data/training/splits/60_20_20_seed42.txt \
  --embeddings data/data/processed/embeddings/hibou_l/10x_native_224_stride224_tissue005 \
  --output "data/training/experiments/hibou_l_10x_mil_$(date +%Y%m%d_%H%M%S)" \
  --device cuda:0 --epochs 30 --seed 42 --cache-in-ram
```

El primer comando genera un TXT tabulado con cabecera `case_id split`, una fila por paciente y valores `train`, `val` o `test`. Divide por paciente, estratifica por `event` y fija 305/102/101 pacientes para las 508 WSI actuales. Ordena los IDs antes de dividir; no sobrescribe un TXT existente. La semilla del split y la del entrenamiento son independientes.

El entrenamiento lee los HDF5 existentes: no extrae embeddings ni necesita las WSI originales. `--cache-in-ram` mantiene train y validación en RAM (aproximadamente 39 GB de vectores FP16 para este dataset, más memoria de trabajo); omitirlo para leer del disco en cada época. La precarga usa dieciséis lectores por defecto, ajustables con `--cache-workers`. Por defecto usa todos los parches de cada WSI, una WSI por paso y acumula gradientes de ocho pacientes.

La base es MIL con atención y cuatro probabilidades condicionales de recaída, con pérdida LogisticHazard de pycox sin ponderación adicional. Los tres cortes internos son cuartiles de los tiempos de recaída de **train** y el horizonte final es su máximo seguimiento. Los censurados aportan los intervalos completos observados; una recaída posterior al horizonte se trata como censura en ese horizonte. El riesgo para Harrell es el negativo del área trapezoidal bajo la curva de supervivencia hasta ese horizonte. No convierte todavía la salida en las clases ni los archivos de entrega de OWL.

`--input-dim`, `--hidden-dim`, `--bins`, `--pooling attention|mean`, `--dropout`, `--lr`, `--weight-decay`, `--accumulation` y `--seed` permiten variar el modelo y el entrenamiento. Para otro encoder, indicar su carpeta de HDF5 y su dimensión con `--input-dim`.

Los splits y los experimentos se agrupan en `data/training/`, dentro del almacenamiento externo ignorado por Git. Los experimentos quedan en `data/training/experiments/`. Cada ejecución requiere una carpeta nueva y conserva configuración, copia de split/etiquetas/código, `metrics.csv`, predicciones de validación y **todos** los checkpoints por época (modelo y optimizador). No selecciona `best` ni carga embeddings o evalúa pacientes de test. Los TXT de split quedan en `data/training/splits/`. Todos los artefactos persistentes generados (splits, embeddings, checkpoints, métricas, predicciones y cachés de ejecución) se guardan bajo `data/`, la carpeta montada, y quedan fuera de Git.
