# Sistema para la detección de patrones de desinformación
Este repositorio contiene el Trabajo de Fin de Grado (TFG) orientado a la **detección y clasificación de técnicas, tácticas y procedimientos (TTPs)** de desinformación, siguiendo el marco **DISARM**, aplicadas a textos de noticias.  
El sistema está basado en una arquitectura de **Retrieval-Augmented Generation (RAG)** para mejorar la precisión en el análisis contextual.

---

## 📂 Estructura del Repositorio

- `rag.ipynb`: Notebook que implementa el sistema **RAG** para analizar una noticia individual.
- `zeroShot.ipynb`: Notebook que usa un enfoque **zero-shot** para el análisis de una noticia individual.
- `rag_script.py`: Script que permite el análisis **en lote** de noticias a través del sistema RAG.

  ### 🔧 Uso del script:
  ```bash
  python rag_script.py news.csv
  ```
  Donde `news.csv` es el archivo con las noticias que deseas analizar (debe estar en formato CSV).

---

## 📊 Dataset Utilizado

Se ha utilizado el **ISOT Fake News Dataset**, que contiene noticias clasificadas como falsas y verdaderas en archivos separados:

- [`True.csv`](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset)
- [`Fake.csv`](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset)

Puedes descargar ambos archivos desde su repositorio en Kaggle.

---

## 🧠 TTPs de DISARM

Se incluye un archivo `.json` con la lista de TTPs del framework **DISARM** que se han escogido para realizar la clasificación (solo se incluyen las que pueden ser detectadas en textos de noticias).

---

## ⚙️ Instalación de Dependencias

Para garantizar el correcto funcionamiento del proyecto, es importante instalar las siguientes dependencias **en el orden especificado**:

```bash
pip install tiktoken langchain langchain_community langchain-chroma
pip install numpy==1.23.5
```

Asegúrate de que la versión de `numpy` es correcta:
```python
import numpy as np
print(np.__version__)  # Debe mostrar 1.23.5
```

```bash
pip install sentencepiece torch
pip install huggingface_hub transformers langchain-huggingface
pip install eventregistry keybert nltk
python -m nltk.downloader stopwords
```

También puedes revisar el archivo `requirements.txt`, que contiene las versiones exactas utilizadas durante el desarrollo.

---

## ✅ Requisitos Generales

- Cuenta en Hugging Face para cargar el modelo de Mistral-7B-Instruct y un modelo de embeddings.
- Cuenta en Event Registry para acceder a su API de noticias.

---

## 📌 Notas Finales

Este TFG ha sido desarrollado como un enfoque exploratorio hacia la aplicación de arquitecturas modernas como RAG en el ámbito de la detección automatizada de desinformación.
Dejando atrás la tarea clásica de clasificación binaria (verdadero/falso), el proyecto busca **demostrar que es posible identificar y categorizar patrones específicos de desinformación**, utilizando el marco **DISARM** como estándar para esta tarea.

---

**Autora**: *Inés Berlanga García*  
**Universidad**: *Escuela Técnica Superior de Ingenieros de Telecomunicación, Universidad Politécnica de Madrid*  
**Año**: 2025

