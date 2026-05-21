from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

PERSIST_DIR = "rag_space/chroma_db"
SCORE_THRESHOLD = 0.4


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


embeddings = LocalEmbeddings()

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

llm = OllamaLLM(
    model="llama3",
    temperature=0.3,
    num_predict=150,
    top_p=0.8
)


def get_relevant_docs(question: str) -> list:
    results = vectordb.similarity_search_with_score(question, k=4)
    relevant_docs = []
    for doc, score in results:
        print(f"[Score: {score:.4f}] {doc.metadata.get('source', 'inconnu')}")
        if score <= SCORE_THRESHOLD:
            relevant_docs.append(doc)
    return relevant_docs


def rag_agent(question: str) -> str:
    docs = get_relevant_docs(question)

    if not docs:
        print("[]")
        prompt = f"""Tu disposes de 150 tokens maximum. Tu DOIS terminer ta réponse par une phrase complète, jamais coupée en plein milieu. Sois direct et concis, sans liste ni introduction.

Question : {question}
Réponse complète et courte :"""
        return llm.invoke(prompt)

    print(f"[Mode : RAG — {len(docs)} document(s) pertinent(s) utilisé(s)]")

    context = "\n\n".join([
        f"Source : {doc.metadata.get('source', 'inconnue')}\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""Tu es un assistant documentaire. Tu disposes de 150 tokens maximum. Tu DOIS terminer ta réponse par une phrase complète, jamais coupée. Réponds uniquement à partir du contexte, sans liste ni introduction.
Si la réponse n'est pas dans le contexte, dis-le en une phrase.

Contexte :
{context}

Question : {question}
Réponse complète et courte :"""

    return llm.invoke(prompt)