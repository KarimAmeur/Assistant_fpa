# PATCH POUR STREAMLIT CLOUD - À placer en tout début de app.py
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # pysqlite3 n'est pas disponible, continuer avec sqlite3 normal
    pass

import streamlit as st
import os
import zipfile
import shutil
from pathlib import Path

# Configuration de la page - DOIT être en début de script
st.set_page_config(
    page_title="Assistant FPA - Ingénierie de Formation",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des APIs - Utilise les secrets Streamlit
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "")
except Exception as e:
    st.error(f"Erreur de configuration des secrets: {e}")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Configurer le token HuggingFace
if HUGGINGFACE_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN

# Imports après configuration
try:
    from langchain_community.vectorstores import Chroma
    from langchain_mistralai import ChatMistralAI
    from mistralai.client import MistralClient
    from prompting import (
        retrieve_documents,
        generate_context_response,
        generate_example_training_plan,
        generate_pedagogical_engineering_advice,
        reformulate_competencies_apc,
        generate_structured_training_scenario
    )
    from user_rag_page import user_rag_page
except ImportError as e:
    st.error(f"Erreur d'importation: {e}")
    st.stop()

# Définition des couleurs
COLORS = {
    "primary": "#1D5B68",
    "secondary": "#E6525E", 
    "light_blue": "#94B7BD",
    "very_light_blue": "#DDE7E9",
    "dark_gray": "#3F3F3F",
    "light_gray": "#F5F5F6",
    "background": "#1A1A1A",
    "text": "#FFFFFF"
}

# Configuration CSS personnalisée
def local_css():
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
        }}
        
        h1, h2, h3 {{
            color: {COLORS["light_blue"]};
            font-family: 'Helvetica Neue', sans-serif;
        }}
        
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["light_blue"]};
        }}
        
        .stSelectbox>div>div>div {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            border-radius: 6px;
            border: 1px solid {COLORS["light_blue"]};
        }}
        
        .stButton>button {{
            background-color: {COLORS["primary"]};
            color: {COLORS["text"]};
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s;
        }}
        
        .stButton>button:hover {{
            background-color: {COLORS["light_blue"]};
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
        }}
        
        .scenario-card, .user-message, .assistant-message {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid {COLORS["secondary"]};
            margin-bottom: 10px;
        }}
        
        .upload-box {{
            background-color: {COLORS["dark_gray"]};
            border: 2px dashed {COLORS["light_blue"]};
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        }}
        
        .banner {{
            background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["light_blue"]});
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }}
        
        .info-box {{
            background-color: {COLORS["very_light_blue"]};
            color: {COLORS["dark_gray"]};
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

# Appliquer le CSS
local_css()

# NOUVELLE FONCTION : Nettoyage automatique ChromaDB
def clean_corrupted_chromadb(db_path):
    """Nettoie automatiquement une base ChromaDB corrompue"""
    try:
        st.warning("🔧 Détection d'une base ChromaDB incompatible...")
        st.info("🗑️ Suppression automatique de l'ancienne base...")
        
        # Supprimer le dossier corrompu
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        st.success("✅ Base corrompue supprimée avec succès !")
        st.info("📋 Veuillez re-uploader votre fichier chromadb_formation.zip")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors du nettoyage: {e}")
        return False

# Fonctions de gestion de la base vectorielle
@st.cache_resource
def extract_database_if_needed():
    """Décompresse automatiquement la base vectorielle si nécessaire"""
    
    db_path = "chromadb_formation"
    zip_path = "chromadb_formation.zip"
    
    # Si le dossier existe déjà et n'est pas vide, pas besoin de décompresser
    if os.path.exists(db_path) and os.listdir(db_path):
        st.success("✅ Base vectorielle déjà disponible")
        return True
    
    # Si le zip existe, le décompresser
    if os.path.exists(zip_path):
        st.info("📦 Décompression de la base vectorielle...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            st.success("✅ Base vectorielle décompressée avec succès")
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la décompression: {e}")
            return False
    
    return False

def database_upload_interface():
    """Interface d'upload de la base vectorielle"""
    
    st.markdown("""
    <div class="upload-box">
        <h3>📤 Upload de votre base vectorielle</h3>
        <p>La base vectorielle ChromaDB est nécessaire pour le fonctionnement de l'assistant.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisir le fichier chromadb_formation.zip",
        type="zip",
        help="Uploadez votre base vectorielle compressée au format ZIP"
    )
    
    if uploaded_file is not None:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("💾 Sauvegarde du fichier...")
            progress_bar.progress(25)
            
            with open("chromadb_formation.zip", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            status_text.text("📦 Décompression en cours...")
            progress_bar.progress(50)
            
            with zipfile.ZipFile("chromadb_formation.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            progress_bar.progress(100)
            status_text.text("✅ Base vectorielle installée avec succès!")
            
            st.success("🎉 Votre base vectorielle a été installée ! L'application va redémarrer...")
            st.balloons()
            
            import time
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'installation: {e}")
            st.info("💡 Assurez-vous que le fichier ZIP contient bien le dossier 'chromadb_formation'")
    
    return False

# Fonctions de mise en cache avec nouveaux imports
class MistralEmbeddings:
    """
    Wrapper LangChain pour Mistral Embed (1024 dims).
    Identique à celui utilisé dans rag_formation.py
    """
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        from mistralai.client import MistralClient
        self.client = MistralClient(api_key=api_key)
        self.model = model

    def embed_documents(self, texts):
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = self.client.embeddings(model=self.model, input=batch)
                embeddings.extend([d.embedding for d in resp.data])
            except Exception as e:
                st.error(f"Erreur embedding lot {i//batch_size+1}: {e}")
                embeddings.extend([[0.0]*1024 for _ in batch])
        return embeddings

    def embed_query(self, text: str):
        try:
            resp = self.client.embeddings(model=self.model, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            st.error(f"Erreur embedding requête: {e}")
            return [0.0]*1024

@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embedding Mistral - IDENTIQUE à rag_formation.py"""
    try:
        if not MISTRAL_API_KEY:
            st.error("❌ Clé API Mistral manquante pour les embeddings")
            return None
            
        return MistralEmbeddings(api_key=MISTRAL_API_KEY)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle d'embedding Mistral: {e}")
        return None

@st.cache_resource
def load_vector_store():
    """Charge la base vectorielle avec mise en cache et gestion d'erreurs"""
    try:
        embeddings = load_embedding_model()
        if embeddings is None:
            return None
            
        db_path = "chromadb_formation"
        if not os.path.exists(db_path):
            st.error("❌ Base vectorielle 'chromadb_formation' non trouvée")
            return None
            
        try:
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            # Test de fonctionnement
            vectorstore.similarity_search("test", k=1)
            return vectorstore
            
        except Exception as chroma_error:
            error_msg = str(chroma_error).lower()
            
            # Détecter l'erreur de colonne manquante
            if "no such column: collections.topic" in error_msg:
                st.error("❌ Base ChromaDB incompatible détectée")
                
                # Nettoyage automatique
                if clean_corrupted_chromadb(db_path):
                    return "needs_reupload"
                else:
                    return None
            else:
                st.error(f"❌ Erreur ChromaDB: {chroma_error}")
                return None
                
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la base vectorielle: {e}")
        return None

@st.cache_resource
def create_mistral_llm():
    """Crée l'instance Mistral avec mise en cache"""
    try:
        if not MISTRAL_API_KEY:
            st.error("❌ Clé API Mistral manquante")
            return None
            
        return ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model="mistral-large-latest",
            temperature=0.1,
            max_tokens=4000
        )
    except Exception as e:
        st.error(f"❌ Erreur lors de la création du modèle Mistral: {e}")
        return None

# Initialisation automatique du système
def initialize_system():
    """Initialise le système avec gestion automatique de la base et des erreurs"""
    
    # Vérifier/décompresser la base d'abord
    if not extract_database_if_needed():
        return None, None, "database_missing"
    
    with st.spinner("🚀 Initialisation de l'Assistant FPA..."):
        # Charger la base vectorielle
        vectorstore = load_vector_store()
        
        # Vérifier si on doit re-uploader
        if vectorstore == "needs_reupload":
            return None, None, "database_missing"
        
        # Charger le modèle LLM
        llm = create_mistral_llm()
        
        if vectorstore is None:
            return None, None, "vectorstore_error"
        if llm is None:
            return None, None, "llm_error"
            
        return vectorstore, llm, "success"

# Vérification et initialisation
if 'initialized' not in st.session_state:
    vectorstore, llm, status = initialize_system()
    st.session_state.vectorstore = vectorstore
    st.session_state.llm = llm
    st.session_state.initialization_status = status
    st.session_state.conversation_history = []
    st.session_state.scenarisation_history = []
    st.session_state.initialized = True

# Gestion des erreurs d'initialisation
if st.session_state.initialization_status == "database_missing":
    st.markdown("""
    <div class="banner">
        <h1>🎓 Assistant FPA - Ingénierie de Formation</h1>
        <p>Configuration initiale requise</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ Base vectorielle non trouvée ou incompatible")
    st.info("📋 Veuillez uploader votre base vectorielle pour commencer à utiliser l'assistant.")
    
    if not database_upload_interface():
        st.stop()

elif st.session_state.initialization_status in ["vectorstore_error", "llm_error"]:
    st.error("❌ Erreur lors de l'initialisation du système")
    
    # Interface de diagnostic
    st.subheader("🔧 Diagnostic")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Tester API Mistral"):
            try:
                from mistralai.client import MistralClient
                client = MistralClient(api_key=MISTRAL_API_KEY)
                models = client.list_models()
                st.success("✅ API Mistral accessible")
            except Exception as e:
                st.error(f"❌ Erreur API Mistral: {e}")
    
    with col2:
        if st.button("🤗 Tester token HuggingFace"):
            try:
                if HUGGINGFACE_TOKEN:
                    from huggingface_hub import whoami
                    info = whoami(token=HUGGINGFACE_TOKEN)
                    st.success(f"✅ Token HF valide: {info['name']}")
                else:
                    st.warning("⚠️ Token HuggingFace non configuré")
            except Exception as e:
                st.error(f"❌ Token HF invalide: {e}")
    
    with col3:
        if st.button("📁 Vérifier base vectorielle"):
            if os.path.exists("chromadb_formation"):
                files = os.listdir("chromadb_formation")
                st.info(f"📂 Dossier trouvé avec {len(files)} fichiers")
            else:
                st.error("❌ Dossier 'chromadb_formation' non trouvé")
    
    if st.button("🔄 Réessayer l'initialisation"):
        st.session_state.initialized = False
        st.rerun()
    
    st.stop()

# Page principale de chat
def main_chat_page():
    """Page principale de chat avec l'assistant FPA"""
    
    st.markdown("""
    <div class="banner">
        <h1>🎓 Assistant FPA - Ingénierie de Formation</h1>
        <p>Votre partenaire intelligent pour la conception et l'amélioration de vos formations professionnelles</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conteneur principal
    chat_container = st.container()
    
    with chat_container:
        # Afficher l'historique de conversation
        for message in st.session_state.conversation_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>Vous:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant FPA:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)

        # Input pour le message de l'utilisateur
        if prompt := st.chat_input("Posez votre question sur la formation professionnelle"):
            # Ajouter le message de l'utilisateur à l'historique
            st.session_state.conversation_history.append({
                'role': 'user', 
                'content': prompt
            })

            # Afficher le message de l'utilisateur
            st.markdown(f"""
            <div class="user-message">
                <strong>Vous:</strong><br>
                {prompt}
            </div>
            """, unsafe_allow_html=True)

            # Générer la réponse
            with st.status("🔍 Recherche en cours...", expanded=True) as status:
                st.write("📚 Recherche des documents pertinents...")
                retrieved_docs = retrieve_documents(
                    st.session_state.vectorstore, 
                    prompt
                )
                
                st.write("🧠 Génération de la réponse...")
                response = generate_context_response(
                    st.session_state.llm, 
                    prompt, 
                    retrieved_docs,
                    st.session_state.conversation_history
                )
                
                status.update(label="✅ Recherche terminée", state="complete", expanded=False)
                
            # Afficher la réponse
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Assistant FPA:</strong><br>
                {response}
            </div>
            """, unsafe_allow_html=True)

            # Ajouter la réponse à l'historique
            st.session_state.conversation_history.append({
                'role': 'assistant', 
                'content': response
            })

            # Option d'affichage des documents sources
            with st.expander("📚 Documents sources"):
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"""
                    <div class="scenario-card">
                        <h4>Document {i}</h4>
                        <p><span style="background: {COLORS['primary']}; color: white; padding: 2px 8px; border-radius: 4px;">Score: {doc['score']:.2f}</span></p>
                        <p><strong>Titre:</strong> {doc['title']}</p>
                        <hr>
                        {doc['content']}
                    </div>
                    """, unsafe_allow_html=True)

    # Sidebar avec outils
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #1D5B68, #94B7BD); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 24px; font-weight: bold; color: white;">FPA</div>
        <h3>Assistant Formation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🛠️ Outils supplémentaires")

    if st.sidebar.button("📝 Exemple de plan de formation"):
        with st.spinner("📝 Génération d'un exemple de plan..."):
            exemple_plan = generate_example_training_plan(st.session_state.llm)
            st.markdown(f"""
            <div class="scenario-card">
                <h2>📝 Exemple de Plan de Formation</h2>
                <div class="info-box">
                    Ce plan peut servir de modèle pour vos propres formations.
                </div>
                {exemple_plan}
            </div>
            """, unsafe_allow_html=True)

    if st.sidebar.button("🔍 Aide à l'ingénierie pédagogique"):
        with st.spinner("🔍 Génération de conseils..."):
            aide_ingenierie = generate_pedagogical_engineering_advice(st.session_state.llm)
            st.markdown(f"""
            <div class="scenario-card">
                <h2>🔍 Conseils d'Ingénierie Pédagogique</h2>
                <div class="info-box">
                    Conseils pour améliorer vos méthodes d'ingénierie pédagogique.
                </div>
                {aide_ingenierie}
            </div>
            """, unsafe_allow_html=True)

# Page de scénarisation
def scenarisation_page():
    """Page de scénarisation de formation"""
    
    st.markdown("""
    <div class="banner">
        <h1>🎯 Scénarisation de Formation</h1>
        <p>Créez des scénarios pédagogiques adaptés à vos objectifs</p>
    </div>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("""
        <div class="scenario-card">
            <h3>📋 Paramètres du scénario</h3>
        """, unsafe_allow_html=True)
        
        input_type = st.selectbox(
            "Type d'entrée",
            ["Titre", "Programme", "Compétences"]
        )
        
        input_data = st.text_area(f"Contenu de {input_type.lower()}", 
            height=150,
            placeholder=f"Saisissez ici votre {input_type.lower()} de formation..."
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="scenario-card">
            <h3>⏱️ Durée de formation</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            duration_hours = st.number_input("Heures", min_value=0, max_value=40, value=3, step=1)
        with col2:
            duration_minutes = st.number_input("Minutes supplémentaires", min_value=0, max_value=59, value=30, step=5)
        
        total_duration_minutes = (duration_hours * 60) + duration_minutes
        
        st.markdown(f"""
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <span style="background: {COLORS['primary']}; color: white; padding: 5px 10px; border-radius: 5px; font-size: 1rem;">
                ⏱️ Durée totale: {duration_hours}h{duration_minutes if duration_minutes > 0 else ''} ({total_duration_minutes} minutes)
            </span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✨ Générer le scénario de formation", use_container_width=True):
            if input_data:
                user_content = f"""
                <div>
                    <p><strong>Type d'entrée:</strong> {input_type}</p>
                    <p><strong>Contenu:</strong> {input_data}</p>
                    <p><strong>Durée:</strong> {duration_hours}h{duration_minutes if duration_minutes > 0 else ''} ({total_duration_minutes} minutes)</p>
                </div>
                """
                
                st.session_state.scenarisation_history.append({
                    'role': 'user', 
                    'content': user_content
                })
                
                st.markdown(f"""
                <div class="user-message">
                    <strong>Votre demande:</strong><br>
                    {user_content}
                </div>
                """, unsafe_allow_html=True)
                
                with st.status("🎯 Création de votre scénario de formation...", expanded=True) as status:
                    input_type_lower = input_type.lower()
                    
                    if input_type_lower == 'competences':
                        st.write("🔄 Reformulation des compétences selon l'approche par compétences...")
                        reformulated_competencies = reformulate_competencies_apc(
                            st.session_state.llm,
                            st.session_state.vectorstore,
                            input_data
                        )
                        input_data = reformulated_competencies
                    
                    st.write("📝 Génération du scénario pédagogique...")
                    scenario = generate_structured_training_scenario(
                        st.session_state.llm,
                        st.session_state.vectorstore,
                        input_data,
                        input_type_lower,
                        total_duration_minutes
                    )
                    
                    status.update(label="✅ Scénario terminé!", state="complete", expanded=False)
                
                st.markdown(f"""
                <div class="assistant-message">
                    <h3>📋 Votre Scénario de Formation</h3>
                    <div class="info-box">
                        Ce scénario a été généré en fonction de vos paramètres.
                    </div>
                    {scenario}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.scenarisation_history.append({
                    'role': 'assistant', 
                    'content': scenario
                })
            else:
                st.warning("⚠️ Veuillez saisir un contenu pour générer le scénario.")
                
    with right_col:
        st.markdown("""
        <div class="scenario-card">
            <h3>💡 Guide de scénarisation</h3>
            <p>Pour créer un scénario de formation efficace:</p>
            <ol>
                <li><strong>Choisissez un type d'entrée</strong></li>
                <li><strong>Définissez le contenu</strong> avec détails</li>
                <li><strong>Ajustez la durée</strong> selon vos contraintes</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Onglets de navigation
tab1, tab2, tab3 = st.tabs(["💬 Assistant FPA", "🎯 Scénarisation", "📚 RAG Personnel"])

with tab1:
    main_chat_page()

with tab2:
    scenarisation_page()

with tab3:
    user_rag_page()