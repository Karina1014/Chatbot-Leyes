from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from docx import Document
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
import google.generativeai as genai
from datetime import datetime
import os

# === Cargar configuración de entorno y validar ===
def cargar_config():
    load_dotenv()
    config = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY")
    }

    if not all(config.values()):
        raise EnvironmentError("Faltan variables de entorno requeridas.")
    
    return config

config = cargar_config()

# === Inicializar servicios ===
genai.configure(api_key=config["GOOGLE_API_KEY"])
modelo_gemini = genai.GenerativeModel("gemini-1.5-flash")
model_embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

qdrant_client = QdrantClient(
    url=config["QDRANT_URL"],
    api_key=config["QDRANT_API_KEY"]
)

# === Constantes de la app ===
UPLOAD_FOLDER = "docs_upload"
COLLECTION_NAME = "codigo_ninez_y_adolescencia_qdrant"
CHUNK_SIZE = 500
MODEL_DIM = 384
EXCEL_PATH = "registro_chat.xlsx"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === FastAPI App ===
app = FastAPI(title="Chatbot Leyes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Utilidades ===
def inicializar_qdrant():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=MODEL_DIM, distance=Distance.COSINE)
    )

# extrae texto de pdf y lo divide en fragmentos
def pdf_a_chunks(file_path: str, chunk_size: int = CHUNK_SIZE):
    texto = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            texto += page.get_text().strip() + "\n"

    texto = texto.strip()
    chunks = [texto[i:i + chunk_size] for i in range(0, len(texto), chunk_size)]
    vectores = model_embeddings.encode(chunks)
    return chunks, vectores

def construir_prompt(contexto: str, pregunta: str) -> str:
    return f"""
Eres un juez imparcial, justo y altamente capacitado en derecho.

Debes analizar un caso basándote únicamente en el contenido legal proporcionado en el contexto, el cual corresponde a leyes oficiales. No puedes inventar leyes ni aplicar criterios personales. Tu única fuente de verdad es el contexto legal.

Tu objetivo es emitir un juicio razonado, imparcial y bien estructurado que responda a la situación planteada.

Analiza la situación presentada, identifica qué normas aplican y explica por qué, usando los artículos del contexto. Emite un veredicto final claro al final de tu respuesta.

No uses formato Markdown, listas ni símbolos especiales. Usa lenguaje claro y legal, con texto plano y bien organizado.

Contexto legal:
\"\"\"{contexto}\"\"\"

Hechos del caso:
Pregunta: {pregunta}

Respuesta del juez (razonamiento jurídico + veredicto final):
"""

def guardar_en_excel(pregunta: str, respuesta: str, path: str = EXCEL_PATH):
    if os.path.exists(path):
        wb = load_workbook(path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws[get_column_letter(1) + "1"] = "Pregunta"
        ws[get_column_letter(2) + "1"] = "Respuesta"
        ws[get_column_letter(3) + "1"] = "tiempo"

    fila = ws.max_row + 1
    segundos_actuales = datetime.now().strftime("%S")
    ws[f"A{fila}"] = pregunta
    ws[f"B{fila}"] = respuesta
    ws[f"C{fila}"] = segundos_actuales
    wb.save(path)

# === Endpoints ===
# Subir docuemento PDF y cargar a Qdrant
@app.post("/documento/subir", summary="Subir documento PDF y cargar a Qdrant")
async def subir_documento(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .pdf")

    ruta = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(ruta, "wb") as f:
        f.write(await file.read())

    chunks, vectores = pdf_a_chunks(ruta)
    inicializar_qdrant()

    puntos = [
        PointStruct(id=i, vector=vectores[i].tolist(), payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=puntos)

    return {
        "estado": "ok",
        "fragmentos_cargados": len(puntos),
        "archivo": file.filename
    }

class ConsultaChat(BaseModel):
    pregunta: str

@app.post("/chat", summary="Consulta al chatbot usando contexto de documentos")
async def consultar_chat(req: ConsultaChat):
    try:
        vector_pregunta = model_embeddings.encode([req.pregunta])[0]
        resultados = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_pregunta,
            limit=4
        )

        contexto = "\n\n".join([r.payload["text"] for r in resultados])
        prompt = construir_prompt(contexto, req.pregunta)

        respuesta = modelo_gemini.generate_content(prompt)
        texto_respuesta = respuesta.text.strip()

        # Guardar pregunta y respuesta en Excel
        guardar_en_excel(req.pregunta, texto_respuesta)

        return {"respuesta": texto_respuesta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {e}")
