# Nuevas técnicas de detección de intrusiones Zero-Day usando Deep Learning

Código del Trabajo Fin de Máster en Ingeniería de Telecomunicación (Universidad de Alcalá).

Autor: Agustín Casuso. Tutor: José Antonio Portilla Figueras. Cotutor: Alberto Palomo Alonso.

El trabajo compara nueve arquitecturas de autoencoder para detectar ataques no vistos durante el entrenamiento. Todas se entrenan solo con tráfico normal y marcan como ataque las muestras con un error de reconstrucción alto. Las pruebas se hacen sobre tres conjuntos de referencia (NSL-KDD, DS2OS y Mirai). Sobre la mejor arquitectura, el autoencoder denso, se estudia una variación propia, el denoising autoencoder, y ambos modelos se validan con tráfico NetFlow real, cuyas decisiones se explican con un EBM sustituto.

## Estructura

| Carpeta | Contenido |
|---|---|
| `notebooks/Generacion_Flujos_Mirai.ipynb` | Generación de `data/flows.csv` a partir de las capturas originales de Mirai |
| `notebooks/01_Autoencoder_Baseline` | Autoencoder denso |
| `notebooks/02_Autoencoder_Sparsity` | Autoencoder disperso |
| `notebooks/03_Autoencoder_Temporal` | LSTM, GRU, Conv1D, Conv2D y TCN |
| `notebooks/04_Transformers` | Transformer y LogSparse Transformer |
| `notebooks/05_Barrido_Ventana` | Barrido del tamaño de ventana con GRU, incluido el diagnóstico de la fuga de `src_ip` en Mirai |
| `notebooks/06_Autoencoder_DAE` | Formas de corrupción del denoising autoencoder y repeticiones con cinco semillas |
| `notebooks/Autoencoder_TRAFICO_REAL.ipynb` | Validación con tráfico NetFlow real y explicabilidad con EBM |
| `figuras` | Ilustraciones de la memoria y scripts que las generan |

Cada cuaderno es autocontenido: carga el conjunto, aplica el preprocesado, entrena el modelo, fija el umbral y calcula las métricas. Las salidas guardadas son las de la ejecución que aparece en la memoria.

## Entorno

```bash
conda env create -f environment.yml
conda activate tfm-zero-day
```

Los experimentos se ejecutaron en Windows 11, en CPU, con Python 3.10.13 y TensorFlow 2.10.0. Todos los cuadernos fijan la semilla 42, salvo las repeticiones con las semillas 42, 1, 2, 3 y 4, y activan las operaciones deterministas de TensorFlow. Con otras versiones o en GPU los resultados pueden variar ligeramente.

## Datos

Los conjuntos de datos no se incluyen en el repositorio.

- **NSL-KDD** y **DS2OS** se descargan con `kagglehub`, que los guarda en su caché local, donde los buscan los cuadernos:

  ```python
  import kagglehub
  kagglehub.dataset_download("hassan06/nslkdd")
  kagglehub.dataset_download("libamariyam/ds2os-dataset")
  ```

- **Mirai** parte de las capturas originales ([Mirai raw dataset](https://www.kaggle.com/datasets/nguyenquanitmo/mirai-raw-dataset)), que se descargan igual con `kagglehub.dataset_download("nguyenquanitmo/mirai-raw-dataset")`. `Generacion_Flujos_Mirai.ipynb` las convierte en flujos con `dpkt` y guarda el resultado en `data/flows.csv`, que es el fichero que leen los cuadernos de Mirai. Para pasar las capturas de pcapng a pcap necesita `editcap`, que se instala con Wireshark.

## Tráfico NetFlow real

`Autoencoder_TRAFICO_REAL.ipynb` se incluye para documentar el procedimiento, pero no puede ejecutarse fuera del servidor de la Universidad de Alcalá donde se alojan los datos, que no son públicos. Se ejecutó en un entorno propio del servidor, con Python 3.12.3 e `interpret` 0.7.8, y necesita que la variable de entorno `NETFLOW_ROOT` apunte a la raíz del proyecto en ese servidor.

## Figuras

```bash
cd figuras
python generar_figuras.py
python figuras_cap5.py ../notebooks/01_Autoencoder_Baseline
```

La Ilustración 1, el esquema del autoencoder, se incluye directamente como imagen (`fig_esquema_autoencoder.png`). `generar_figuras.py` dibuja los esquemas de las arquitecturas. `figuras_cap5.py` genera las figuras de resultados y necesita los ficheros `err_*.npz` que guarda la última celda de los cuadernos de `01_Autoencoder_Baseline`. La figura del EBM (`python figuras_cap5.py ebm`) usa datos exportados del servidor que no se publican.
