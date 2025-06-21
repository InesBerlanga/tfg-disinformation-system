from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems, ReturnInfo, ArticleInfoFlags

from keybert import KeyBERT
from nltk.corpus import stopwords
import string
from collections import Counter
import nltk
from difflib import SequenceMatcher
import re

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

import argparse
import json
import pandas as pd
import time
import datetime
import psutil

#login(token=HuggingFaceToken) 

# CARGAR Mistral 7B Instruct y modelo de embeddings
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Función para gererar una respuesta 
def llm_generate(prompt):
    # Tokenizar input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # return_tensors="pt" porque AutoModelForCausalLM está basado en PyTorch y espera como entrada tensores de PyTorch
    
    # Generar respuesta
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result[len(prompt):].strip()

    
# EXTRAER KEYWORDS
nltk.download('stopwords')

def similar(a, b, threshold=0.7):
    # Devuelve True si a y b son suficientemente similares (ratio > threshold)
    return SequenceMatcher(None, a, b).ratio() > threshold

# Creación del extractor de keywords
class KeywordExtractor:
    def __init__(self):
        self.kw_model = KeyBERT()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.min_word_length = 3

    def clean_keyword(self, kw):
        # Limpia y normaliza una keyword
        kw = kw.lower().strip()
        kw = ''.join([c for c in kw if c not in self.punctuation])
        return kw if (kw and kw not in self.stop_words 
                     and len(kw) >= self.min_word_length) else None

    # Extraer keywords usando KeyBERT + filtrado tradicional
    def extract_keywords(self, text, top_n=5, diversity=0.5):
        # Extracción inicial con KeyBERT
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),  # Permite palabras sueltas y bigramas
            stop_words='english',
            top_n=top_n*3  # Extraemos más para luego filtrar
        )
        
        # Procesamiento y limpieza
        cleaned = []
        for kw, score in keywords:
            kw_clean = self.clean_keyword(kw)
            if kw_clean:
                # Separar bigramas para mejor filtrado
                cleaned.extend(kw_clean.split())
        
        # Frecuencia de términos y selección
        word_counts = Counter(cleaned)
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], -len(x[0])))
        
        # Selección diversificada
        selected = []
        for word, count in sorted_words:
            # Evitar palabras demasiado similares
            if not any(similar(word, s) for s in selected):
                selected.append(word)
                if len(selected) >= top_n:
                    break
        
        return selected[:top_n]


# OBTENER NOTICIAS PARECIDAS
er_key = "" # API key de Event Registry
er = EventRegistry(apiKey=er_key)

# Extraer título y cuerpo asumiendo formato "TITLE: ... BODY: ..."
def extraer_titulo_y_cuerpo(texto):
    titulo_match = re.search(r"TITLE:\s*(.*?)\s*BODY:", texto, re.DOTALL | re.IGNORECASE)
    cuerpo_match = re.search(r"BODY:\s*(.*)", texto, re.DOTALL | re.IGNORECASE)

    titulo = titulo_match.group(1).strip() if titulo_match else ""
    cuerpo = cuerpo_match.group(1).strip() if cuerpo_match else ""
    return titulo, cuerpo

# Para no extraer de la API la misma noticia que la de entrada    
def mismo_articulo(art, noticia):
    titulo_original, cuerpo_original = extraer_titulo_y_cuerpo(noticia)

    titulo_art = art["title"].strip()
    cuerpo_art = art.get("body", "").strip()

    titulo_sim = similar(titulo_art, titulo_original, threshold=0.7)
    cuerpo_sim = similar(cuerpo_art[:500], cuerpo_original[:500], threshold=0.7)

    es_mismo = titulo_sim or cuerpo_sim
    return es_mismo
        
# Devuelve noticias relacionadas con la noticia de entrada
def buscar_noticias_parecidas(texto, max_noticias=100, lang="eng"):
    extractor = KeywordExtractor()
    keywords = extractor.extract_keywords(texto)
    print(f"Keywords seleccionadas: {keywords}")
    
    if not keywords:
        return []

    # Construir query
    q = QueryArticlesIter(
        keywords=QueryItems.OR(keywords),
        lang=lang
    )
    
    ret_info = ReturnInfo(articleInfo=ArticleInfoFlags(bodyLen=-1))

    # Llamada a la API
    try:
        articles = list(q.execQuery(er, returnInfo=ret_info, maxItems=max_noticias))
        return [art for art in articles if not mismo_articulo(art, texto)]
    except Exception as e:
        print(f"Error en la búsqueda: {e}")
        return []


# SPLIT
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

#  Convertir las noticias en formato Document
def convertir_a_documento(art):
    body = art["body"][:5000] # Hasta 5000 caracteres para no generar demasiados tokens y sobrecargar al llm. 
    metadata = {
        "title"        : art["title"],
        "published"    : art["dateTimePub"],  
        "source_name"  : art["source"]["title"],
        "source_domain": art["source"]["uri"],
        "url"          : art["url"],
        "authors"      : ", ".join(a["name"] for a in art["authors"]),
        "event_uri"    : art.get("eventUri"),
        "sentiment"    : art.get("sentiment"),
        "body_len"     : len(art["body"]),
    }
    doc = Document(
        page_content = body,
        metadata     = metadata
    )
    return doc

# Fragmentar los documentos en splits
def hacer_splits(articulos):
    documentos = [convertir_a_documento(articulo) for articulo in articulos]
    splits = text_splitter.split_documents(documentos)
    return splits

    
# INDEX AND RETRIEVAL
# Crear base de datos vectorial y extraer de ella los 4 splits más relevantes
def obtener_splits_relevantes_bbdd(splits, noticia):
    # Inicializar el cliente de Chroma
    client = chromadb.Client(Settings())
    collection_name = "temporal_context"

    # Eliminar si ya existía (con cada noticia se crea una nueva base de datos)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # No pasa nada si no existía

    # Crear una colección con similitud del coseno
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Especificar la métrica de distancia
    )

    # Integrar con LangChain
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings # Se vectorizan automáticamente los splits con el modelo definido de embeddings
    )

    # Agregar documentos a la colección
    vectorstore.add_documents(splits)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Obtenemos los más relevantes
    docs = retriever.invoke(noticia) #invoke transforma la noticia en un embedding y llama a getRelevantDocuments.
    print("Cantidad de splits relevantes:", len(docs))
    context_docs = "\n\n".join(serialize_doc(d) for d in docs)
    return context_docs, docs

def serialize_doc(doc):
    meta = doc.metadata
    return (
        f"<<TITLE>>\n"
        f"{meta['title']}\n"
        #f"source={meta['source_name']} ({meta['source_domain']})\n"
        f"<<BODY>>\n"
        f"{doc.page_content}\n"
        f"<<END>>"
    )

    
# CARGAR DISARM
def cargar_ttps(path="DISARMfiltrado.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    bloques = []
    counters_dict = {}
    for obj in data["objects"]:
        if obj["type"] == "tactic":
            tactic_name = obj["name"]
            for tech in obj.get("techniques", []):
                id_ = tech["id"]
                name = tech["name"]
                desc = tech["description"].strip()
                counters = tech.get("counters", [])
                counters_dict[id_] = counters
                bloques.append(f"Technique: {id_} - {name} (Tactic: {tactic_name})\nTechnique Description: {desc}")
    return "\n\n".join(bloques), counters_dict

# Extrae el ID de la TTP de un string que puede venir con ruido.
def extraer_ttp_id(ttp_string):
    match = re.match(r"(T\d{4}(?:\.\d{3})?)", ttp_string)
    if match:
        return match.group(1)
    else:
        return None

    

#CARGAR NOTICIAS:
def cargar_noticias_csv(path, columna_texto="text", columna_titulo="title"):
    df = pd.read_csv(path)
    df = df[[columna_titulo, columna_texto]].dropna()
    noticias = df.astype(str).to_dict(orient="records")  # lista de dicts con 'title' y 'text'
    return noticias



# GENERATION
def construir_prompt(news_text,news_context,ttps_text):
    return f"""
        You are an expert in communication, disinformation, and discourse analysis. You will now receive one news article to analyze, a list of known manipulation and disinformation techniques (DISARM TTPs), and a few other news articles with related context.
        
        Your task is to determine whether the FIRST ARTICLE that you receive contains any patterns of manipulation or disinformation using the CONTEXT ARTICLES as background evidence.

        Use the TTPs section below as the reference list of patterns to detect.
        
        Return the result **strictly in valid JSON format** with this structure:

        {{
          "ttps_detected": [string], // TTP names detected in the FIRST ARTICLE, e.g. ["T0066-Degrade Adversary", "T0015-Create Hashtags and Search Artefacts"]
          "key_phrases": [string], // Phrases or evidence that support your findings
          "justification": string // A concise explanation of your reasoning
        }}
        
        If no clear technique is detected, respond:
        
        {{
          "ttps_detected": [],
          "key_phrases": [],
          "justification": string // A concise explanation of your reasoning
        }}

        
        FIRST ARTICLE to analyze:
        {news_text}

        CONTEXT ARTICLES:
        {news_context}

        TTPs (DISARM framework):
        {ttps_text}

        RESPONSE:
        ===
    """

# Se procesa cada noticia, el resultado se guarda en output_path y los logs generados en el procedimiento se guardan en proceso_log_path
def analizar_noticias_en_lote(lista_noticias, output_path="resultados.jsonl", proceso_log_path="procesos6.jsonl"):
    resultados = []
    process = psutil.Process()

    for i, noticia in enumerate(lista_noticias):
        print(f"Procesando noticia {i+1}/{len(lista_noticias)}...")

        try:
            # Para los logs
            start_time = time.time()
            cpu_times_start = process.cpu_times()
            mem_info_start = process.memory_info().rss

            # Se pone la noticia en formato "TITLE:... BODY:..."
            titulo = noticia["title"]
            texto = noticia["text"]
            noticia_completa = f"TITLE: {titulo}\n\n BODY: {texto}"

            # OBTENER CONTEXTO
            extractor = KeywordExtractor() 
            keywords = extractor.extract_keywords(noticia_completa)
            articulos = buscar_noticias_parecidas(noticia_completa)
            num_context_articles = len(articulos) if articulos else 0

            splits = hacer_splits(articulos) if articulos else []
            num_splits = len(splits)

            if splits:
                context_docs, relevant_docs = obtener_splits_relevantes_bbdd(splits, noticia_completa)
                num_relevant_splits = len(relevant_docs)
                relevant_splits_text = [d.page_content for d in relevant_docs]

            else:
                context_docs = ""
                relevant_docs = []
                num_relevant_splits = 0
                relevant_splits_text = []

            # GENERACIÓN 
            prompt = construir_prompt(noticia_completa, context_docs, ttps_text)
            respuesta_raw = llm_generate(prompt)


            # Intentar parsear como JSON
            try:
                respuesta = json.loads(respuesta_raw)
            except Exception as e:
                print(f"[!] No se pudo parsear JSON: {e}")
                respuesta = {
                    "ttps_detected": [],
                    "justification": "Could not parse response as valid JSON"
                }

            # Generar contramedidas asociadas a las TTPs detectadas
            contramedidas = {}
            for ttp in respuesta.get("ttps_detected", []):
                ttp_id = extraer_ttp_id(ttp)
                if ttp_id:
                    contramedidas[ttp] = counters_dict.get(ttp_id, [])
                else:
                    contramedidas[ttp] = []

            # Construir resultado
            resultado = {
                "input": titulo,
                "output": respuesta,
                "countermeasures": contramedidas
            }

            resultados.append(resultado)

            # Guardar incrementalmente resultado
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(resultado) + "\n")

            # Logs del proceso
            exec_time = time.time() - start_time
            cpu_times_end = process.cpu_times()
            mem_info_end = process.memory_info().rss

            cpu_user_time_delta = cpu_times_end.user - cpu_times_start.user
            cpu_system_time_delta = cpu_times_end.system - cpu_times_start.system

            mem_mb_start = mem_info_start / (1024 * 1024)
            mem_mb_end = mem_info_end / (1024 * 1024)

            proceso_info = {
                "title": titulo,
                "keywords": keywords,
                "num_context_articles": num_context_articles,
                "num_splits": num_splits,
                "num_relevant_splits": num_relevant_splits,
                "relevant_splits_text": relevant_splits_text,
                "execution_time_sec": round(exec_time, 2), # Tiempo total en segundos que ha tardado en procesar toda la noticia 
                "cpu_user_time_sec": round(cpu_user_time_delta, 2), #Tiempo de CPU consumido en modo usuario (ejecución del código Python)
                "cpu_system_time_sec": round(cpu_system_time_delta, 2), # Tiempo de CPU consumido en modo sistema (llamadas al sistema operativo, por ejemplo I/O, operaciones de red, filesystem)
                "mem_start_MB": round(mem_mb_start, 2), # Memoria RAM ocupada (MB) justo antes de empezar a procesar la noticia
                "mem_end_MB": round(mem_mb_end, 2), # Memoria RAM ocupada (MB) justo después de terminar de procesar la noticia
            }

            with open(proceso_log_path, "a", encoding="utf-8") as f_log:
                f_log.write(json.dumps(proceso_info) + "\n")


        except Exception as e:
            print(f"[!] Error procesando noticia {i+1}: {e}")
        
        # pequeña pausa entre iteraciones
        time.sleep(1)  # evitar sobrecarga de recursos

    print(f"Proceso terminado. {len(resultados)} noticias analizadas.")
    return resultados

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cargar y analizar noticias desde un archivo CSV.")
    parser.add_argument('csv_file', type=str, help="Archivo CSV con las noticias a cargar")
    args = parser.parse_args()

    noticias = cargar_noticias_csv(args.csv_file)
    ttps_text, counters_dict = cargar_ttps("DISARMfiltrado.json")
    analizar_noticias_en_lote(noticias)