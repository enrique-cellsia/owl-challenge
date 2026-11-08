# OWL challenge

Extracción de embeddings con Hibou-L congelado. El entrenamiento MIL está pendiente.

## Uso

Python 3.11, PyTorch con CUDA y un snapshot local de Hibou-L con sus pesos y código:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train/extract_all_wsi.py \
  --raw /ruta/al/dataset/data/raw \
  --snapshot /ruta/al/snapshot/hibou-L \
  --output /ruta/a/embeddings/10x \
  --mpp 0.9701389525095192 --gpus 0,1,2 --dry-run
```

Quitar `--dry-run` para ejecutar la extracción. La entrada contiene `images/**/*.tif` y `tissue_masks/<nombre>_tissue.tif`. La MPP del ejemplo corresponde al ≈10× nativo de OWL; para 20× usar `--mpp 0.5` y otra carpeta de salida.

Por defecto: parches de 224 × 224, stride 224, mínimo 5 % de tejido, lote 1.024 y ocho lectores por GPU. Se ajustan con `--stride`, `--min-tissue`, `--batch-size` y `--workers`. `--dry-run` revisa el inventario sin escribir ni lanzar la extracción.

Cada WSI genera un HDF5 con vectores `[N,2048]` FP16, coordenadas y fracción de tejido. Los logs quedan en `output/logs/`. Repetir el comando omite resultados completos compatibles y retoma parciales. Ctrl+C permite parar conservando los lotes confirmados; evitar cierres forzados. Para dejarlo ejecutando al cerrar la terminal, usar una sesión persistente externa.

Para una sola WSI: `python train/extract_wsi_embeddings.py --help`; `--plan-only` cuenta parches sin extraer embeddings. Datos, pesos y entornos locales quedan fuera de Git.

Pruebas: `python -m unittest discover -s train/tests -v`.
