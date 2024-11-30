import os
import requests
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
import torch
import streamlit as st
import PyPDF2

class PDFVectorAssistant:
    def __init__(self, pinecone_api_key, huggingface_api_key):
        # Inicializar Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "pdf-embeddings"

        # Criar índice se não existir
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Obter índice
        self.index = self.pc.Index(self.index_name)

        # Modelo de embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # Modelo de QA
        self.qa_tokenizer = AutoTokenizer.from_pretrained("recogna-nlp/bode-7b-alpaca-pt-br")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("recogna-nlp/bode-7b-alpaca-pt-br")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model,
            tokenizer=self.qa_tokenizer
        )
        
        # Chave da API do Hugging Face
        self.huggingface_api_key = huggingface_api_key

    def generate_embedding(self, text):
        # Gerar embedding
        with torch.no_grad():
            tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**tokens)
            embeddings = self._mean_pooling(outputs, tokens['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze().numpy().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        # Pooling médio
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def index_pdf(self, pdf_file):
        # Indexar PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        # Dividir texto em chunks
        chunks = self._split_text(full_text)

        # Indexar chunks
        upserts = []
        for i, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk)
            upserts.append({
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk}
            })

        self.index.upsert(upserts)
        return len(chunks)

    def _split_text(self, text, chunk_size=500):
        # Dividir texto em chunks
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def query_documents(self, query, top_k=3):
        # Busca vetorial
        query_embedding = self.generate_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match['metadata']['text'] for match in results['matches']]

    def ask_huggingface_model(self, question, context):
        try:
            # Verifica se o contexto não está vazio
            if not context.strip():
                return "Não foi possível encontrar contexto relevante para responder à pergunta."
            
            # Trunca o contexto se for muito longo para evitar erros de memória
            max_context_length = 4000
            context = context[:max_context_length]
            
            # Executa a pipeline de question answering
            result = self.qa_pipeline({
                'question': question,
                'context': context
            })

            return result['answer']

        
        except Exception as e:
            return f"Erro ao processar a pergunta: {str(e)}"

# Streamlit App
def main():
    st.title("Assistente")

    pinecone_api_key = "pcsk_5Sv89r_2otFGncovVGUmVed9UumLiW6MEfag2uEJpYV1FARyueouH72JdAmDaFiDb4NBnd"
    huggingface_api_key = "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"

import os
import requests
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
import torch
import streamlit as st
import PyPDF2

class PDFVectorAssistant:
    def __init__(self, pinecone_api_key, huggingface_api_key):
        # Inicializar Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "pdf-embeddings"

        # Criar índice se não existir
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Obter índice
        self.index = self.pc.Index(self.index_name)

        # Modelo de embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Modelo de Question Answering
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=self.qa_model, 
            tokenizer=self.qa_tokenizer
        )
        
        # Chave da API do Hugging Face
        self.huggingface_api_key = huggingface_api_key

    def generate_embedding(self, text):
        # Gerar embedding
        with torch.no_grad():
            tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**tokens)
            embeddings = self._mean_pooling(outputs, tokens['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze().numpy().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        # Pooling médio
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def index_pdf(self, pdf_file):
        # Indexar PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        # Dividir texto em chunks
        chunks = self._split_text(full_text)

        # Indexar chunks
        upserts = []
        for i, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk)
            upserts.append({
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk}
            })

        self.index.upsert(upserts)
        return len(chunks)

    def _split_text(self, text, chunk_size=500):
        # Dividir texto em chunks
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def query_documents(self, query, top_k=3):
        # Busca vetorial
        query_embedding = self.generate_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match['metadata']['text'] for match in results['matches']]

    def ask_huggingface_model(self, question, context, max_answer_length=100):
        try:
            # Verifica se o contexto não está vazio
            if not context.strip():
                return "Não foi possível encontrar contexto relevante para responder à pergunta."
            
            # Trunca o contexto se for muito longo para evitar erros de memória
            max_context_length = 4000
            context = context[:max_context_length]
            
            # Executa a pipeline de question answering
            result = self.qa_pipeline({
                'question': question,
                'context': context
            })

            return result['answer']

        
        except Exception as e:
            return f"Erro ao processar a pergunta: {str(e)}"

# Streamlit App
def main():
    st.title("Assistente")

    pinecone_api_key = "pcsk_5Sv89r_2otFGncovVGUmVed9UumLiW6MEfag2uEJpYV1FARyueouH72JdAmDaFiDb4NBnd"
    huggingface_api_key = "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"

    if pinecone_api_key and huggingface_api_key:
        try:
            assistant = PDFVectorAssistant(pinecone_api_key, huggingface_api_key)
            
            uploaded_file = st.sidebar.file_uploader("Carregar PDF", type=['pdf'])
            
            if uploaded_file:
                if st.sidebar.button("Indexar Documento"):
                    try:
                        num_chunks = assistant.index_pdf(uploaded_file)
                        st.sidebar.success(f"Indexado {num_chunks} chunks")
                    except Exception as e:
                        st.sidebar.error(f"Erro de indexação: {e}")
            
            query = st.text_input("Faça uma pergunta sobre os documentos")

            if query:
                try:
                    # Obter trechos relevantes
                    relevant_texts = assistant.query_documents(query)
                    context = " ".join(relevant_texts)

                    # Perguntar ao modelo Hugging Face
                    answer = assistant.ask_huggingface_model(query, context)
                    st.write(f"**Resposta:** {answer}")
                    
                    # Mostrar contexto relevante para transparência
                    with st.expander("Contexto Relevante"):
                        for i, text in enumerate(relevant_texts, 1):
                            st.text(f"Trecho {i}: {text[:500]}...")

                except Exception as e:
                    st.error(f"Erro na consulta: {e}")
        
        except Exception as e:
            st.error(f"Erro de inicialização: {e}")

if __name__ == "__main__":
    main()
