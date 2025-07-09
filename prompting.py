import logging
import traceback
import csv
from io import StringIO
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma

import os
try:
    import streamlit as st
    # Récupérer les tokens depuis les secrets Streamlit
    HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", os.getenv("HUGGINGFACE_TOKEN", ""))
    MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY", ""))
    
    # Configurer le token HuggingFace
    if HUGGINGFACE_TOKEN:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN
except:
    # Fallback si streamlit n'est pas disponible (utilisation locale)
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    if HUGGINGFACE_TOKEN:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def extract_query_essence(query: str) -> str:
    """Version simplifiée SANS spacy - Compatible Streamlit Cloud"""
    try:
        # Mots vides français simples (sans spacy)
        stop_words = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'me', 'même', 'te', 'si', 'la', 'du', 'des', 'les', 'au', 'aux',
            'donne', 'moi', 'explique', 'dis', 'trouve', 'comment', 'quoi', 'où', 'quand',
            'pourquoi', 'pourrais', 'peux', 'voudrais', 'veux', 'cherche', 'montre'
        }
        
        # Nettoyage simple du texte
        words = query.lower().split()
        
        # Filtrer les mots vides et garder les mots importants
        filtered_words = [
            word.strip('.,?!:;') 
            for word in words 
            if len(word) > 2 and word.lower() not in stop_words
        ]
        
        # Si on a des mots filtrés, les retourner
        if filtered_words:
            essence = " ".join(filtered_words)
        else:
            # Sinon, retourner la requête originale
            essence = query
        
        print(f"Requête originale: '{query}'")
        print(f"Essence extraite: '{essence}'")
        
        return essence
        
    except Exception as e:
        print(f"Erreur lors de l'extraction (version simple): {e}")
        return query

def retrieve_documents(
    vectorstore, 
    query: str, 
    max_results: int = 5,
    min_content_length: int = 100
) -> List[Dict[str, Any]]:
    """Récupère les documents pertinents pour une requête."""
    if not vectorstore:
        print("Base de données vectorielle non disponible.")
        return []

    try:
        processed_query = extract_query_essence(query)
        
        results = vectorstore.similarity_search_with_score(
            query=processed_query, 
            k=max_results * 2
        )
        
        if not results:
            print("Aucun document trouvé.")
            return []
        
        filtered_results = [(doc, score) for doc, score in results 
                           if len(doc.page_content) >= min_content_length]
        filtered_results = filtered_results[:max_results]
            
        retrieved_docs = []
        for doc, raw_score in filtered_results:
            retrieved_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': raw_score,
                'title': doc.metadata.get('titre', doc.metadata.get('filename', 'Document sans titre'))
            })

        print(f"\n{'='*80}")
        print(f"RÉSULTATS DE RECHERCHE")
        print(f"Requête originale: '{query}'")
        print(f"Requête optimisée: '{processed_query}'")
        print(f"Nombre de documents trouvés: {len(retrieved_docs)}")
        print(f"{'='*80}")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDOCUMENT {i+1}")
            print(f"Titre: {doc['title']}")
            print(f"Score brut: {doc['score']}")
            print(f"Métadonnées: {doc['metadata']}")
            print(f"Longueur du contenu: {len(doc['content'])} caractères")
            print(f"{'-'*40}\nCONTENU:")
            
            content_preview = doc['content'][:1000]
            formatted_preview = ""
            for line in content_preview.split('\n'):
                formatted_preview += line + "\n"
            
            print(f"{formatted_preview}{'...' if len(doc['content']) > 1000 else ''}")
            print(f"{'-'*80}")

        return retrieved_docs

    except Exception as e:
        print(f"Erreur lors de la récupération des documents: {e}")
        traceback.print_exc()
        return []

def generate_context_response(
    llm, 
    query: str, 
    retrieved_docs: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None
    ) -> str:
    """Génère une réponse basée sur les documents récupérés."""
    if not retrieved_docs:
        return "Je n'ai pas trouvé de documents pertinents pour répondre à votre question. Pourriez-vous reformuler ou préciser votre demande?"
    
    context = ""
    
    max_docs_to_include = min(5, len(retrieved_docs))
    total_chars = 0
    max_chars_per_doc = 2000
    max_total_chars = 8000
    
    for i, doc in enumerate(retrieved_docs[:max_docs_to_include]):
        doc_header = f"\n\nDOCUMENT {i+1} [Score: {doc['score']:.2f}/100, Titre: {doc['title']}]\n"
        
        meta_text = ""
        if doc['metadata']:
            relevant_meta = {k: v for k, v in doc['metadata'].items() 
                           if k in ['auteur', 'date', 'source', 'categorie', 'titre', 'filename'] and v}
            if relevant_meta:
                meta_text = "Métadonnées: " + ", ".join([f"{k}: {v}" for k, v in relevant_meta.items()]) + "\n"
        
        header_size = len(doc_header) + len(meta_text)
        content_size = min(max_chars_per_doc, len(doc['content']))
        
        if total_chars + header_size + content_size > max_total_chars:
            context += f"\n\n[Note: {len(retrieved_docs) - i} documents supplémentaires non inclus en raison de limitations de taille]"
            break
        
        content_preview = doc['content'][:content_size]
        context += doc_header + meta_text + f"Contenu: {content_preview}"
        
        if len(doc['content']) > content_size:
            content_percentage = int((content_size / len(doc['content'])) * 100)
            context += f"\n[Note: {content_percentage}% du contenu affiché, {len(doc['content']) - content_size} caractères omis]"
        
        total_chars += header_size + content_size
    
    history_text = ""
    if conversation_history and len(conversation_history) > 0:
        history_text = "HISTORIQUE DE LA CONVERSATION:\n"
        for i, msg in enumerate(conversation_history[-5:]):
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            history_text += f"[{i+1}] {role}: {msg['content']}\n\n"
    
    full_prompt = f"""
    Tu es un assistant d'information spécialisé dans la création de réponses détaillées, précises et nuancées sur la formation professionnelle pour adultes.
    
    {history_text}
    
    CONTEXTE DES DOCUMENTS RÉCUPÉRÉS:
    {context}
    
    QUESTION ACTUELLE: "{query}"
    
    CONSIGNES POUR LA GÉNÉRATION DE LA RÉPONSE:
    1. Analyse en profondeur les documents fournis et identifie les informations les plus pertinentes
    2. Développe une réponse exhaustive qui couvre tous les aspects de la question
    3. Structure ta réponse de manière logique et hiérarchisée (introduction, développement, conclusion)
    4. Utilise des sous-titres si nécessaire pour organiser les informations complexes
    5. Inclus des exemples concrets et des illustrations quand c'est pertinent
    6. Cite explicitement les documents sources en indiquant "Selon le document X..."
    7. Si les documents contiennent des fragments incomplets, essaie de reconstruire le sens à partir des différents extraits
    8. Si les informations sont contradictoires entre les documents, présente ces nuances
    9. Si les documents ne permettent pas de répondre complètement, indique précisément les limites
    10. Adopte un ton professionnel, précis et nuancé
    11. Évite les simplifications excessives et présente la complexité du sujet quand c'est nécessaire
    
    Ta réponse doit être la plus informative et complète possible tout en restant pertinente et spécialisée dans le domaine de la formation professionnelle pour adultes.
    """
    
    try:
        logging.info(f"Génération de réponse pour la requête: '{query}'")
        
        # Appel à l'API Mistral via LangChain
        response = llm.invoke(full_prompt)
        
        # Extraire le contenu de la réponse
        if hasattr(response, 'content'):
            result = response.content
        elif isinstance(response, dict) and 'content' in response:
            result = response['content']
        elif isinstance(response, str):
            result = response
        else:
            result = str(response)
        
        logging.info(f"Réponse générée avec succès ({len(result)} caractères)")
        return result
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse: {e}")
        logging.exception("Détails de l'erreur:")
        return "Je n'ai pas pu générer une réponse complète en raison d'une erreur technique. Pourriez-vous reformuler votre question ou réessayer plus tard?"

def generate_example_training_plan(llm) -> str:
    """Génère un exemple de plan de formation."""
    exemple_prompt = """
    Génère un exemple de plan de formation professionnelle détaillé pour adultes en formation continue, 
    incluant:
    - Contexte de la formation
    - Objectifs pédagogiques formulés selon l'approche par compétences
    - Public cible
    - Méthodes et modalités pédagogiques
    - Contenu et progression
    - Évaluation des compétences
    - Ressources nécessaires
    
    Assure-toi que l'exemple soit concret, applicable et respecte les standards de l'ingénierie pédagogique.
    """
    
    try:
        response = llm.invoke(exemple_prompt)
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        logging.error(f"Erreur lors de la génération du plan de formation: {e}")
        return "Impossible de générer un exemple de plan de formation."

def generate_pedagogical_engineering_advice(llm) -> str:
    """Génère des conseils d'ingénierie pédagogique."""
    aide_prompt = """
    Donne des conseils professionnels pour construire 
    une démarche d'ingénierie pédagogique efficace pour la formation professionnelle d'adultes, 
    en détaillant:
    - Les étapes clés du processus d'ingénierie pédagogique
    - Les bonnes pratiques à suivre
    - Les méthodes d'analyse des besoins
    - Les outils d'évaluation des compétences
    - Les stratégies d'adaptation aux différents publics
    - Les erreurs courantes à éviter
    
    Focalise-toi sur des conseils pratiques et applicables immédiatement.
    """
    
    try:
        response = llm.invoke(aide_prompt)
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        logging.error(f"Erreur lors de la génération des conseils d'ingénierie: {e}")
        return "Impossible de générer des conseils d'ingénierie pédagogique."

def reformulate_competencies_apc(
    llm, 
    vectorstore, 
    initial_competencies: str
) -> str:
    """Reformule les compétences selon l'Approche Par Compétences (APC)."""
    retrieved_docs = retrieve_documents(
        vectorstore, 
        f"Approche Par Compétences TARDIF, reformulation de compétences, {initial_competencies}"
    )
    
    context = "\n\n".join([
        f"Document {i+1} (Score: {doc['score']:.2f}):\n{doc['content']}" 
        for i, doc in enumerate(retrieved_docs)
    ])
    
    reformulation_prompt = f"""
    Contexte des documents récupérés sur l'Approche Par Compétences:
    {context}
    
    Compétences initiales à reformuler:
    {initial_competencies}
    
    Instructions pour la reformulation selon l'Approche Par Compétences (APC) de TARDIF:
    1. Transformer les compétences en énoncés précis et observables
    2. Utiliser la structure : Verbe d'action + Objet + Contexte + Critères de performance
    3. Assurer que chaque compétence soit :
       - Réaliste et atteignable
       - Mesurable et évaluable
       - Contextualisée dans un environnement professionnel
       - Alignée avec les standards professionnels du domaine
    4. Décomposer les compétences complexes en composantes spécifiques
    5. Mettre en évidence le résultat attendu plutôt que le processus d'apprentissage
    6. Inclure des indicateurs de performance pour chaque compétence
    
    Compétences reformulées selon l'APC:
    """
    
    try:
        response = llm.invoke(reformulation_prompt)
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        logging.error(f"Erreur lors de la reformulation des compétences: {e}")
        return "Impossible de reformuler les compétences selon l'APC."

def generate_structured_training_scenario(llm, vectorstore, input_data, input_type, duration_minutes=210):
    """Génère un scénario de formation structuré."""
    csv_structure = """DURÉE\tHORAIRES\tCONTENU\tOBJECTIFS PÉDAGOGIQUES\tMETHODE\tREPARTITION DES APPRENANTS\tACTIVITES\t\tRESSOURCES et MATERIEL\tEVALUATION\t
\t\t\t\t\tFormateur\tApprenants\t\tType\tSujet
20 min\t9h00-9h20\tIntroduction à la formation\tN/A\ttransmissive\tgroupe entier\tprésentation du formateur, du programme et des objectifs, méthodologie et modalités d'évaluation\técoute active, questions\tprésentation PowerPoint, liste des participants\tN/A\tN/A
25 min\t9h20-9h45\tÉvaluation diagnostique\tIdentifier le niveau initial des participants\tactive\tindividuel puis groupe entier\tdistribution et explication du questionnaire, supervision\tréalisation du test de positionnement\tquestionnaire d'évaluation, stylos\tdiagnostique\tconnaissances préalables
35 min\t9h45-10h20\tConcepts fondamentaux\tComprendre les principes de base du sujet en identifiant les éléments clés dans un contexte professionnel\ttransmissive puis interrogative\tgroupe entier\tprésentation des concepts, questionnement ciblé\tprise de notes, réponses aux questions\tsupport de cours, vidéoprojecteur\tformative\tcompréhension des concepts
15 min\t10h20-10h35\tPause\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A
40 min\t10h35-11h15\tActivité pratique en sous-groupes\tAppliquer les concepts théoriques dans une situation professionnelle en respectant les critères de qualité définis\tactive\tpetits groupes de 3-4\tprésentation des consignes, facilitation, accompagnement des groupes\tétude de cas, résolution collaborative de problèmes\tdocument de travail, fiches consignes, paperboard\tformative\tapplication pratique
30 min\t11h15-11h45\tMise en commun et analyse\tSynthétiser et analyser les solutions produites en identifiant les points forts et axes d'amélioration\tparticipative\tgroupe entier\tanimation des échanges, apport de compléments, clarifications\tprésentation des productions, partage d'expérience, questions\tpaperboard, feutres\tformative\tcapacité d'analyse et de synthèse"""
    
    retrieved_docs = retrieve_documents(
        vectorstore,
        f"Formation sur {input_data}, méthodologie, Approche Par Compétences, APC, TARDIF, méthodes pédagogiques",
        max_results=10  
    )
    
    context = "\n\n".join([doc["content"] for doc in retrieved_docs[:5]])
    
    csv_reader = csv.reader(StringIO(csv_structure), delimiter='\t')
    headers = next(csv_reader, None)
    
    if not headers:
        logging.error("Structure CSV invalide: impossible de lire les en-têtes")
        return "Impossible de générer un scénario: structure CSV invalide"
    
    prompt = f"""
    Tu es un expert en ingénierie de formation professionnelle pour adultes (FPA) spécialisé dans la conception de scénarios pédagogiques selon l'Approche Par Compétences (APC).
    
    # DEMANDE SPÉCIFIQUE
    Je souhaite que tu crées un scénario de formation complet et détaillé sur le sujet suivant : "{input_data}".
    Le type d'entrée fourni est : {input_type}.
    La durée totale de la formation est de {duration_minutes} minutes.
    
    # INFORMATIONS CONTEXTUELLES
    Voici des informations pertinentes sur ce sujet, les méthodes pédagogiques et l'Approche Par Compétences (APC) :
    {context}
    
    # STRUCTURE REQUISE POUR LE SCÉNARIO
    Tu dois utiliser EXACTEMENT la structure CSV suivante pour ton scénario de formation :
    {csv_structure}
    
    Les colonnes de ce tableau sont : {', '.join(headers)}
    
    J'ai inclus plusieurs lignes d'exemples pour illustrer le format attendu. Observe bien la structure de ces exemples pour comprendre :
    - Comment formuler les objectifs selon l'APC (avec verbe d'action + contexte + critères)
    - Comment varier les méthodes pédagogiques
    - Comment détailler les activités du formateur et des apprenants
    - Comment préciser les types d'évaluation
    
    Inspire-toi de ce format mais crée un contenu entièrement original pour {input_data} en respectant les principes de l'APC.
    
    Cette structure est uniquement un cadre - TU NE DOIS PAS REPRENDRE LE CONTENU d'exemples précédents.
    Ton scénario doit porter EXCLUSIVEMENT sur {input_data} et non sur un autre sujet.
    
    # CONSIGNES POUR LA CRÉATION DU SCÉNARIO
    1. ANALYSE PRÉALABLE:
       - Identifie clairement le public cible pour une formation sur {input_data}
       - Détermine les prérequis nécessaires pour ce type de formation
       - Respecte strictement la durée totale de {duration_minutes} minutes
    
    2. OBJECTIFS PÉDAGOGIQUES SELON L'APC:
       - Formule 3-5 objectifs pédagogiques spécifiques à {input_data}
       - Utilise la structure TARDIF: Verbe d'action + Objet + Contexte + Critères de performance
       - Assure-toi que les objectifs soient observables, mesurables et contextualisés
       - Décompose les compétences complexes en composantes spécifiques
    
    3. STRUCTURE DU SCÉNARIO:
       - Remplis chaque colonne du tableau CSV selon les en-têtes fournis
       - Divise la durée totale de {duration_minutes} minutes en séquences pédagogiques cohérentes
       - Inclus une pause de 15 minutes au milieu si la formation dure plus de 2 heures
       - Prévois un temps d'introduction et de conclusion
       - Calcule précisément les horaires en partant d'une heure de début (9h00 par défaut)
    
    4. MÉTHODES PÉDAGOGIQUES VARIÉES:
       - Propose une variété de méthodes actives adaptées à l'apprentissage de {input_data}
       - Intègre au moins 4 méthodes différentes parmi: apprentissage par l'action, classe inversée, apprentissage par problèmes, jeux de rôles, co-construction, études de cas, etc.
       - Alterne entre méthodes pédagogiques transmissives, actives et expérientielles
       - Inclus des travaux pratiques spécifiques à {input_data}
    
    5. ÉVALUATION DES COMPÉTENCES:
       - Propose des méthodes d'évaluation formative et/ou sommative
       - Aligne les évaluations sur les objectifs formulés selon l'APC
       - Inclus des critères de réussite mesurables
    
    6. RESSOURCES:
       - Indique les ressources et le matériel nécessaires pour chaque activité
    
    # FORMAT DE RÉPONSE
    Avant le tableau, présente brièvement:
    1. Le titre précis de la formation sur {input_data}
    2. Le public cible pertinent pour cette formation
    3. Les prérequis nécessaires
    4. Les objectifs généraux de la formation formulés selon l'APC
    
    Ensuite, présente ton scénario sous forme d'un tableau au format Markdown qui respecte EXACTEMENT la structure CSV fournie. 
    Assure-toi que les durées des séquences s'additionnent EXACTEMENT pour atteindre {duration_minutes} minutes.
    """
    
    try:
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        logging.error(f"Erreur lors de la génération du scénario de formation: {e}")
        return "Impossible de générer un scénario de formation."