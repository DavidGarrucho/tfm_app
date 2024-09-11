import streamlit as st
import warnings
import configparser
import os
import logging
import random
import string
import time
from langchain_core.chat_history import BaseChatMessageHistory
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



GOOGLE_API_KEY = ""
MODEL_EMBEDDINGS_GOOGLE = "models/embedding-001"
MODEL_LLM_GOOGLE = "gemini-1.5-pro"
OUTPUT_PATH = "output"
FAISS_GOOGLE_PATH = "output/faiss_index_ollama"
NUMEXPR_MAX_THREADS = "16"

reranker_model = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    torch_dtype="float32",  # Cambiar a float32 para evitar BFloat16
    trust_remote_code=True,
)

reranker_model.to('cuda') # or 'cpu' if no GPU is available
reranker_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reranker_tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-reranker-v2-base-multilingual')

# CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

# PROMPT_TEMPLATE_GOOGLE = """You are an intelligent assistant. Answer the question based solely on the information provided in the context, specific for Spain.
# Do not add any information beyond what is given. 
# If the context does not contain the answer, respond with: "The answer is not available in the provided context."
# Be concise and accurate. All the answers must be in Spanish language from Spain, in an informal manner.
# Answers must be clear, precise, and unambiguous, and a teenager should be able to understand the answer.
# Explicitly avoid phrases such as "según el documento", "según el capítulo", "en el texto", "como se menciona en el artículo", or any implication of external texts. Do not construct questions that require knowledge of the structure of the document or the location of information in it.
# Include the content-specific information that supports the answer to allow the answer to be independent of any external text.
# If the content lacks sufficient information to form a complete answer, do not force one.
# Create the answers in your own words; Direct copying of content is not permitted.
# NEVER mention the words "documento", "texto", "presentación", "archivo", "tabla", "artículo", "ley", "capítulo", "preámbulo", "título preliminar", "disposición" or "disposiciones generales" in your questions or answers.
# ALWAYS make sure that all answers are accurate, self-contained, and relevant, without relying on any original document or text or implying its existence, strictly avoiding any invention or speculation.
# IMPORTANT: if in the question there is no mention of a Comunidad Autonoma or the name of a city or province, try that the answer applies to Spain as a country.

# Context:
# {context}
# """

# CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Dado un historial de chat y la última pregunta del usuario que podría referenciar contexto en dicho historial, reformula la pregunta para que sea independiente y comprensible sin necesidad de acceder al historial. No respondas la pregunta; solo reformúlala si es necesario, o devuélvela tal como está si ya es clara.
# """

# PROMPT_TEMPLATE_GOOGLE = """Eres un asistente inteligente. Responde a la pregunta basándote únicamente en la información proporcionada en el contexto, específico para España.

# - No añadas ninguna información que no esté presente en el contexto.
# - Si el contexto no contiene la respuesta, responde con: "La respuesta no está disponible en el contexto proporcionado."
# - Sé conciso, preciso y exacto. Todas las respuestas deben ser en español de España y estar dirigidas a un público juvenil, con un tono cercano pero respetuoso.
# - Las respuestas deben ser claras, fáciles de entender, y no deben depender de documentos externos o de la estructura de estos, **excepto si se solicita una referencia a un artículo o ley específica.**
# - Evita frases como "según el documento", "como se menciona en el artículo", **a menos que el usuario pida explícitamente información sobre un artículo o ley**.
# - La información específica que soporta la respuesta debe estar explícitamente mencionada, de modo que la respuesta sea autocontenida.
# - Si el contexto no proporciona suficiente información, no inventes ni especules.
# - Genera las respuestas con tus propias palabras, sin copiar directamente el contenido.
# - **Puedes mencionar artículos, leyes u otros términos legales si el usuario lo solicita en su pregunta o si la información relevante proviene de un artículo legal.**
# - Si no se menciona una Comunidad Autónoma o ciudad en la pregunta, genera una respuesta aplicable a toda España. Si hay una mención geográfica, adapta la respuesta a la región mencionada cuando corresponda.

# Contexto:
# {context}
# """

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Dado un historial de chat y la última pregunta del usuario que podría referenciar contexto en dicho historial, reformula la pregunta para que sea independiente y comprensible sin necesidad de acceder al historial. No respondas la pregunta; solo reformúlala si es necesario, o devuélvela tal como está si ya es clara.
"""

PROMPT_TEMPLATE_GOOGLE = """Eres un asistente inteligente. Responde a la pregunta basándote únicamente en la información proporcionada en el contexto, específico para España.

- Siempre prioriza mencionar artículos o leyes que estén directamente relacionados con la pregunta del usuario.
- Si el contexto no contiene la respuesta, responde con: "La respuesta no está disponible en el contexto proporcionado."
- Si la pregunta se refiere a un tema legal, intenta identificar y citar el artículo de la ley que aplique.
- Genera las respuestas con tus propias palabras, sin copiar directamente el contenido, y asegúrate de que sean claras, fáciles de entender y aplicables a un público juvenil.

Contexto:
{context}
"""


def load_config():
    global GOOGLE_API_KEY
    global MODEL_EMBEDDINGS_GOOGLE
    global OUTPUT_PATH
    global FAISS_GOOGLE_PATH
    global LOG_PATH
    global NUMEXPR_MAX_THREADS
    
    config = configparser.ConfigParser()
    config.read('streamlit_google_history_final.ini')
    GOOGLE_API_KEY = config['KEYS']['google_api_key']
    MODEL_EMBEDDINGS_GOOGLE = config['MODELS']['model_embeddings_google']
    MODEL_LLM_GOOGLE = config['MODELS']['model_llm_google']
    OUTPUT_PATH = config['DEFAULT']['output_path']
    FAISS_GOOGLE_PATH = config['DEFAULT']['faiss_google_path']
    NUMEXPR_MAX_THREADS = config['DEFAULT']['numexpr_max_threads']
    LOG_PATH = config['DEFAULT']['log_path']


def get_conversational_chain(retriever, session_id):
    
    llm = ChatGoogleGenerativeAI(model = MODEL_LLM_GOOGLE, temperature = 0.3)
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_TEMPLATE_GOOGLE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    msgs = StreamlitChatMessageHistory(key=session_id)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

# Añadir función para calcular la similitud
def calculate_similarity(embedding_1, embedding_2):
    # Calcular similitud de coseno entre dos vectores
    return cosine_similarity([embedding_1], [embedding_2])[0][0]

def rerank_documents(query, documents):
    # Cargar el modelo y tokenizer del reranker
    # Crear los pares (query, document) para el reranker
    sentence_pairs = [[query, doc] for doc in documents]
    
    # Tokenizar los pares
    inputs = reranker_tokenizer(
        [query] * len(documents),
        documents,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Ejecutar el modelo para obtener los scores
    with torch.no_grad():
        outputs = reranker_model(**inputs)
        scores = outputs.logits.squeeze().cpu().numpy()

    # Ordenar los documentos según las puntuaciones
    ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    return ranked_docs

def user_input(user_question, k_value, session_id):
    #embeddings = GoogleGenerativeAIEmbeddings(model = MODEL_EMBEDDINGS_GOOGLE, task_type="retrieval_document")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-xlm-r-multilingual-v1")
    new_db = FAISS.load_local(FAISS_GOOGLE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": k_value})

    # Obtener documentos relevantes
    results = retriever.get_relevant_documents(user_question)
    
    # Extraer el contenido de los documentos para rerankear
    docs_content = [doc.page_content for doc in results]

    # Aplicar el reranker
    ranked_documents = rerank_documents(user_question, docs_content)

    # Seleccionar los mejores documentos rerankeados
    top_documents = [doc for doc, score in ranked_documents[:3]]  # Puedes ajustar el número de documentos a usar

    # Continuar con el proceso normal de generación de respuestas
    chain = get_conversational_chain(retriever, session_id)
    
    # Pasar los documentos relevantes (top_documents) al generador de respuestas
    relevant_context = "\n".join(top_documents)
    response = chain.invoke(
        {"input": user_question, "context": relevant_context},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    
    return response


    
    
def response_generator(prompt, k_value, session_id, logger):
    start_model_exec = time.time()
    response = user_input(prompt, k_value, session_id)
    end_model_exec = time.time()
    resp_text = "{0} (Tiempo de respuesta: {1:.2f} seg. Session ID: {2}).".format(response["answer"], end_model_exec - start_model_exec,  session_id)
    logger.debug(session_id + ' - Contexto ' + str(response) )
    for word in resp_text.split():
        yield word + " "
        time.sleep(0.05)


def main():

    st.set_page_config(page_title="Chat legal v1.0", page_icon="🇪🇸")
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    load_config()
    os.environ["NUMEXPR_MAX_THREADS"] = NUMEXPR_MAX_THREADS
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

    logger = logging.getLogger(__name__)
    log_ruta = LOG_PATH + '/streamlit_google_history_final.log'
    logging.basicConfig(filename=log_ruta, level=logging.DEBUG, 
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Initialization
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
  
    k_value = st.sidebar.number_input( "Inserta el valor de k", value=16, placeholder="Ingresa un entero...", min_value=1, format="%d")
    nombre = st.sidebar.text_input( "Inserta tu nombre", value="", placeholder="Ingresa un texto...", max_chars=50)
    
    st.header("Chat con el modelo", divider="gray")
  
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Inicializar el contexto de la conversación
    if "context" not in st.session_state:
        st.session_state.context = ""
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
  
    # Accept user input
    if prompt := st.chat_input("Haz una pregunta sobre el Código de Tráfico y Seguridad Vial en España"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        logger.info(st.session_state['session_id'] + ' - Pregunta nueva de ' + nombre +': ' + prompt )

        with st.spinner("Procesando, por favor espere..."):
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                #response = st.write_stream(response_generator(prompt, model_chat, st.session_state.context, chunk_size, k_value))
                response = st.write_stream(response_generator(prompt, k_value, st.session_state['session_id'], logger))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            logger.info(st.session_state['session_id'] + ' - Respuesta: ' + response )
            
            # Agregar la pregunta y la respuesta al contexto
            st.session_state.context += f"Pregunta: {prompt}\n"
            st.session_state.context += f"Respuesta: {response}\n"

  
if __name__ == "__main__":
   main()
