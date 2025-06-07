import google.generativeai as generativeai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import time
import pickle

# Carrega a chave secreta do .env
load_dotenv()
chave_secreta = os.getenv('API_KEY')
print("Chave carregada:", chave_secreta)

generativeai.configure(api_key=chave_secreta)

# Lê os dados da planilha do Google Sheets
csv_url = 'https://docs.google.com/spreadsheets/d/1Iev42buaGr5jV29O8kg52_tOoV60X31TeWpR53oRAtE/export?format=csv&id=1Iev42buaGr5jV29O8kg52_tOoV60X31TeWpR53oRAtE'
df = pd.read_csv(csv_url)
print("Dados carregados:")
print(df.head())

# Define o modelo de embedding
model = 'models/gemini-embedding-exp-03-07'

# Função segura para gerar embeddings
def gerar_embedding_seguro(titulo, conteudo):
    try:
        result = generativeai.embed_content(
            model=model,
            content=conteudo,
            task_type="retrieval_document",
            title=titulo
        )
        return result["embedding"]
    except Exception as e:
        print(f"[ERRO] Falha ao gerar embedding para '{titulo}': {e}")
        return None

# Gera os embeddings com controle de tempo
embeddings = []
for i, row in df.iterrows():
    print(f"Gerando embedding {i+1}/{len(df)}: {row['Titulo']}")
    emb = gerar_embedding_seguro(row["Titulo"], row["Conteúdo"])
    embeddings.append(emb)
    time.sleep(17)  # pausa para evitar sobrecarga da API

df["Embeddings"] = embeddings

# Salva os dados com embeddings em arquivo .pkl
pickle.dump(df, open('datasetEmbedding2025.pkl', 'wb'))
print("\nEmbeddings salvos em 'datasetEmbedding2025.pkl'")

# Testa a leitura do arquivo
modeloEmbeddings = pickle.load(open('datasetEmbedding2025.pkl', 'rb'))
print("\nModelo carregado do pickle:")
print(modeloEmbeddings.head())
