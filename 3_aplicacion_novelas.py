import os
import groq
import chromadb
from chromadb.config import Settings
import streamlit as st
from sentence_transformers import SentenceTransformer

# Para inicializar el modelo de embeddings
@st.cache_resource # Cacheamos la inicialización para mejorar el rendimiento
def load_model():
    return SentenceTransformer(
        "jinaai/jina-embeddings-v2-small-en",
        trust_remote_code=True
    )

# Para inicializar la conexión a Chroma
@st.cache_resource # Cacheamos la inicialización para mejorar el rendimiento
def init_chroma():
    # Configurar cliente Chroma con persistencia
    client = chromadb.PersistentClient(
        path="./chroma_novelas_db",
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )
    
    # Obtener la colección de novelas
    collection = client.get_collection("novelas_corpus")
    
    return client, collection

# Clase para el Agente de Novelas
class NovelasAgent:
    def __init__(self):
        self.name = "Chatbot RAG especialista en novelas españolas del siglo XIX"
        self.model = load_model()
        self.client, self.collection = init_chroma()
        
        # Mapeo de nombres de novelas para facilitar la detección
        self.novelas_map = {
            "la de bringas": "La_de_Bringas_314648",
            "bringas": "La_de_Bringas_314648",
            "el sombrero de tres picos": "El_sombrero_de_tres_picos_pg29506",
            "sombrero tres picos": "El_sombrero_de_tres_picos_pg29506",
            "tristana": "Tristana_pg66979",
            "la gaviota": "La_gaviota_pg23600",
            "gaviota": "La_gaviota_pg23600",
            "pepita jimenez": "Pepita_Jimenez_pg17223",
            "pepita": "Pepita_Jimenez_pg17223",
            "jimenez": "Pepita_Jimenez_pg17223",
            "su unico hijo": "Su_unico_hijo_pg17341",
            "único hijo": "Su_unico_hijo_pg17341",
            "unico hijo": "Su_unico_hijo_pg17341",
            "la desheredada": "La_desheredada_pg25956",
            "desheredada": "La_desheredada_pg25956",
            "peñas arriba": "Peñas_arriba_pg24127",
            "penas arriba": "Peñas_arriba_pg24127",
            "platero y yo": "Platero_y_yo_pg9980",
            "platero": "Platero_y_yo_pg9980",
            "los pazos de ulloa": "Los_pazos_de_Ulloa_18005-8_UTF8",
            "pazos ulloa": "Los_pazos_de_Ulloa_18005-8_UTF8",
            "pazos": "Los_pazos_de_Ulloa_18005-8_UTF8",
            "ulloa": "Los_pazos_de_Ulloa_18005-8_UTF8"
        }
    
    def detectar_novelas_objetivo(self, query):
        """Determina qué novelas son relevantes para la consulta"""
        query_lower = query.lower()
        novelas_mencionadas = []
        
        # Buscar novelas mencionadas explícitamente
        for keyword, novela_id in self.novelas_map.items():
            if keyword in query_lower:
                if novela_id not in novelas_mencionadas:
                    novelas_mencionadas.append(novela_id)
        
        # Si no se menciona ninguna novela específicamente, buscar en todas
        if not novelas_mencionadas:
            return "todas"
        
        return novelas_mencionadas
    
    def search(self, query, top_k=5, novelas_filtro=None):
        """Función de búsqueda en el corpus de novelas"""
        # Genero el vector de embeddings para la consulta
        query_embedding = self.model.encode(query)
        
        # Configurar filtros si se especifican novelas
        where_filter = None
        if novelas_filtro and novelas_filtro != "todas":
            if len(novelas_filtro) == 1:
                where_filter = {"novela": {"$eq": novelas_filtro[0]}}
            else:
                where_filter = {"novela": {"$in": novelas_filtro}}
        
        # Busco en Chroma los vectores más similares
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Extraigo y devuelvo los textos relevantes con metadatos
        contexts = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            contexts.append({
                'texto': doc,
                'novela': metadata['novela'],
                'chunk_index': metadata['chunk_index'],
                'distancia': distance
            })
        
        return contexts
    
    def get_system_prompt(self, tipo_consulta="general"):
        """Función de system prompt adaptable según el tipo de consulta"""
        base_prompt = """Eres un experto en literatura española del siglo XIX especializado en análisis literario. 
Tienes acceso a un corpus de 10 novelas clásicas españolas de este período."""
        
        if tipo_consulta == "especifica":
            return base_prompt + """
            
Responde preguntas específicas sobre novelas individuales de manera precisa y detallada.
Cita siempre la novela de la que proviene la información."""
            
        elif tipo_consulta == "comparativa":
            return base_prompt + """
            
Analiza y compara información entre múltiples novelas del corpus.
Identifica patrones, similitudes y diferencias entre las obras.
Proporciona análisis literario fundamentado y bien estructurado."""
            
        else:  # general
            return base_prompt + """
            
Responde basándote únicamente en la información del corpus proporcionado.
Si la pregunta requiere análisis de múltiples novelas, proporciona una síntesis comparativa.
Si es sobre una novela específica, enfócate en esa obra."""

# Coordinador del Agente
class AgentCoordinator:
    def __init__(self):
        self.agent = NovelasAgent()
        self.client = groq.Client(api_key=os.environ.get("GROQ_API_KEY"))
    
    def determinar_tipo_consulta(self, query, novelas_detectadas):
        """Determina el tipo de consulta para ajustar el prompt del sistema"""
        if novelas_detectadas == "todas":
            return "general"
        elif len(novelas_detectadas) == 1:
            return "especifica"
        else:
            return "comparativa"
    
    def process_query(self, query, top_k=5):
        """Procesa la consulta y genera la respuesta usando el agente especializado"""
        # Detectar qué novelas son relevantes
        novelas_objetivo = self.agent.detectar_novelas_objetivo(query)
        
        # Determinar tipo de consulta
        tipo_consulta = self.determinar_tipo_consulta(query, novelas_objetivo)
        
        # Realizar búsqueda
        contexts = self.agent.search(query, top_k, novelas_objetivo)
        
        # Construir prompt con contexto
        if contexts:
            # Agrupar contextos por novela para mejor organización
            contextos_por_novela = {}
            for ctx in contexts:
                novela = ctx['novela']
                if novela not in contextos_por_novela:
                    contextos_por_novela[novela] = []
                contextos_por_novela[novela].append(ctx['texto'])
            
            # Construir el prompt con información organizada
            contexto_info = []
            for novela, textos in contextos_por_novela.items():
                # Limpiar nombre de novela para display
                novela_display = novela.replace('_', ' ').replace('pg', '').replace('UTF8', '').strip()
                contexto_info.append(f"\n--- {novela_display} ---")
                contexto_info.extend(textos)
            
            prompt = f"""
Con base en la siguiente información del corpus de literatura española del siglo XIX:

{chr(10).join(contexto_info)}

Responde a la pregunta: {query}

Instrucciones adicionales:
- Si mencionas información específica, indica de qué novela proviene
- Si la pregunta es comparativa, estructura tu respuesta comparando las obras relevantes
- Si es una pregunta general, sintetiza la información de manera coherente
"""
        else:
            prompt = f"""
No se encontró información relevante en el corpus para responder: {query}

Por favor, reformula tu pregunta o intenta con términos más específicos relacionados con:
- Las novelas del corpus (La de Bringas, Tristana, La Gaviota, etc.)
- Temas literarios (amor, sociedad, personajes, estilo, etc.)
- Aspectos narrativos (argumento, protagonistas, final, etc.)
"""
        
        # Obtener prompt del sistema según el tipo de consulta
        system_prompt = self.agent.get_system_prompt(tipo_consulta)
        
        # Generar la respuesta
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content

# Interfaz de Streamlit
def main():
    st.title("  Chatbot RAG especialista en novelas españolas del siglo XIX")
    st.write("Explora y analiza un corpus de 10 novelas clásicas españolas")
    
    # Sidebar con información del corpus
    with st.sidebar:
        st.header("📖 Corpus de Novelas")
        st.write("""
        **Novelas disponibles:**
        - La de Bringas
        - El sombrero de tres picos  
        - Tristana
        - La gaviota
        - Pepita Jiménez
        - Su único hijo
        - La desheredada
        - Peñas arriba
        - Platero y yo
        - Los pazos de Ulloa
        """)
        
        st.header("💡 Ejemplos de consultas")
        st.write("""
        **Específicas:**
        - ¿Quién es el protagonista de Tristana?
        - ¿Cómo termina La gaviota?
        
        **Comparativas:**
        - ¿Qué diferencias hay entre Pepita Jiménez y Tristana?
        - ¿Cómo se representa el amor en estas novelas?
        
        **Generales:**
        - ¿Qué temas sociales aparecen en el corpus?
        - ¿Cuáles son los estilos narrativos más comunes?
        """)
    
    # Inicializar el coordinador de agentes
    try:
        coordinator = AgentCoordinator()
        
        # Inicializar historial de chat si no existe
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Mostrar el historial de chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Entrada para nueva consulta
        if prompt := st.chat_input("¿Qué quieres saber sobre las novelas del corpus?"):
            # Agregar pregunta del usuario al historial
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generar la respuesta
            with st.chat_message("assistant"):
                with st.spinner("Analizando corpus literario..."):
                    try:
                        # Procesar la consulta a través del coordinador de agentes
                        answer = coordinator.process_query(prompt)
                        
                        # Mostrar la respuesta
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error al procesar la consulta: {str(e)}")
                        answer = "Lo siento, ha ocurrido un error al procesar tu consulta. Asegúrate de que la base de datos Chroma esté disponible."
            
            # Agregar respuesta al historial
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
    except Exception as e:
        st.error(f"Error al inicializar la aplicación: {str(e)}")
        st.write("Asegúrate de que:")
        st.write("- La base de datos Chroma esté creada (ejecuta generar_embeddings_chroma.py)")
        st.write("- Las variables de entorno estén configuradas (GROQ_API_KEY)")
        st.write("- El directorio ./chroma_novelas_db exista y contenga la colección 'novelas_corpus'")

if __name__ == "__main__":
    main()
