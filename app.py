import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import torch
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('stopwords')
french_stopwords = stopwords.words('french')
# --- Charger les données (à adapter si besoin) ---
products = pd.read_csv("packs.csv")
interactions = pd.read_csv("interactions.csv")
users = pd.read_csv("users.csv")

# 🧹 Filtrer uniquement les vues et achats
df = interactions[interactions['type_interaction'].isin(['view', 'purchase'])].copy()

# 🔄 Associer chaque interaction à une catégorie de pack
df = df.merge(products[['pack_id', 'categorie']], left_on='product_id', right_on='pack_id')

# 🧮 Créer une matrice utilisateur x catégorie
user_categorie_matrix = pd.crosstab(df['user_id'], df['categorie'])

# 🔗 Calculer la matrice de corrélation entre catégories
correlation_matrix = user_categorie_matrix.corr()



# 🔍 Fonction de recommandation basée sur les corrélations
def recommander_categories_corrélés(categorie_consultée, correlation_matrix, top_n=3):
    if categorie_consultée not in correlation_matrix.columns:
        return []
    correlations = correlation_matrix[categorie_consultée].drop(categorie_consultée)
    top_categories = correlations.sort_values(ascending=False).head(top_n).index.tolist()
    return top_categories

# Préparation des textes
#products["content"] = products["nom_pack"] + " " + products["description"]+ " " +products["type"]
products["content"] = ( (products["nom_pack"] + " ") * 5 + products["categorie"]*2 + products["description"] + " " + products["type"] )

# Vectorisation TF-IDF

vectorizer_packs = TfidfVectorizer(stop_words=french_stopwords)
tfidf_packs = vectorizer_packs.fit_transform(products["content"])


# --- Préparation des interactions filtrées ---
interaction_types = ['purchase', 'view', 'add_to_cart', 'favorite']
interactions_filtered = interactions[interactions['type_interaction'].isin(interaction_types)]



# --- 🔥 Produits populaires ---
def recommend_popular(top_n=5):
    top_packs = interactions_filtered['product_id'].value_counts().head(top_n).index
    return products[products['pack_id'].isin(top_packs)][['pack_id', 'nom_pack', 'type', 'prix_total', 'description']]
#contet-based
def rechercher(requete, data, tfidf_matrix, top_n=1):
    vecteur = vectorizer_packs.transform([requete]) 
    similarites = cosine_similarity(vecteur, tfidf_matrix).flatten()
    indices = similarites.argsort()[::-1][:top_n]
    # Garder uniquement ceux avec similarité > 0
    indices_filtrés = [i for i in indices if similarites[i] > 0]
    resultat = data.iloc[indices_filtrés].copy()
    resultat["similarité"] = similarites[indices_filtrés]
    return resultat

# --- Streamlit UI ---

def afficher_packs(packs, titre):
    st.subheader(titre)
    cols = st.columns(len(packs))
    for col, (_, row) in zip(cols, packs.iterrows()):
        with col:
          #f"/content/{image_name}"
            #st.image("Fêtes de naissance ou baby showers.png", use_container_width=True)

            st.image(row.get("image_url", ""), use_container_width=True, caption=row["nom_pack"])

            st.markdown(f"**{row['nom_pack']}**")
            st.markdown(f"**Type :** {row['type']}")
            st.markdown(f"**Prix :** {row['prix_total']} TND")
            if "similarité" in row:
                st.markdown(f"**Similarité :** {row['similarité']:.2f}")
            st.markdown(f"_{row['description'][:200]}..._")
def get_packs_interacted_by_user(user_id, interactions_df, products_df):
    # Obtenir les IDs des produits avec lesquels l'utilisateur a interagi
    product_ids = interactions_df[interactions_df['user_id'] == user_id]['product_id'].unique()
    
    # Récupérer les packs correspondants dans la table produits
    packs = products_df[products_df['pack_id'].isin(product_ids)].copy()
    
    return packs

def reset_requete():
    st.session_state.requete_text = ""  # clear the input box



st.sidebar.header("Paramètres")
    # --- Initialiser l'état de session pour chaque utilisateur ---
if "vues_utilisateur" not in st.session_state:
    st.session_state.vues_utilisateur = {}  # {user_id: [pack_ids]}

user_id = st.sidebar.selectbox("Sélectionner un utilisateur", users['user_id'],key="selected_user",
on_change=reset_requete)

if user_id not in st.session_state.vues_utilisateur:
    st.session_state.vues_utilisateur[user_id] = []  # Liste vide pour ce user

    
top_n = st.sidebar.slider("Nombre de recommandations", min_value=1, max_value=20, value=5)

tab1, tab2, = st.tabs(["👤 Utilisateur", "📈 Admin"])



with tab2:
    tab21, tab22 = st.tabs([ "📈 Admin", "🤖 Suggestions intelligentes"])
    with tab21:
       
    
    # 📈 Interactions par type
        st.markdown("### 📈 Répartition des interactions")
        interaction_counts = interactions_filtered['type_interaction'].value_counts()
        st.bar_chart(interaction_counts)

    # 👤 Utilisateurs les plus actifs
        st.markdown("### 👥 Utilisateurs les plus actifs")
        top_users = interactions_filtered['user_id'].value_counts().head(5)
        st.table(top_users)
        # 🎨 Affichage de la heatmap
        st.subheader("🔗 Corrélation entre les catégories de packs consultés")
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
    with tab22:
        st.header("Suggestions intelligentes basées sur la catégorie préférée")
    # Filtrer interactions utilisateur
        df_user = df[df['user_id'] == user_id]

        if df_user.empty:
            st.warning("Aucune interaction trouvée pour cet utilisateur.")
        else:
        # Catégorie préférée selon interactions
            categorie_preferee = df_user['categorie'].value_counts().idxmax()
            st.markdown(f"🎯 Catégorie préférée de l'utilisateur **{user_id}** : `{categorie_preferee}`")

        # Recommandation de catégories corrélées
            categories_suggérées = recommander_categories_corrélés(categorie_preferee, correlation_matrix)

            st.markdown("## 🧠 Catégories similaires recommandées :")
            for cat in categories_suggérées:
                st.markdown(f"- {cat}")

        # Afficher les packs liés
            packs_suggérés = products[products['categorie'].isin(categories_suggérées)]
            afficher_packs(packs_suggérés, f"🎁 Packs similaires à la catégorie préférée de {user_id}")


with tab1:
    st.header("Interface utilisateur")
    # Recommandation LightFM, recherche...



    # 🔎 Recherche personnalisée par mots-clés
    st.markdown("## 🔍 Recommandation par mots-clés")
    requete = st.text_input("Entrez vos préférences ou mots-clés :", key="requete_text")

    if requete.strip():  # Vérifie que la requête n'est pas vide ou que des espaces
        result_packs = rechercher(requete, products, tfidf_packs, top_n=5)
        if not result_packs.empty:
            # 👉 Ajouter seulement le premier pack dans les vues
            premier_pack_id = result_packs.iloc[0]['pack_id']
            if premier_pack_id not in st.session_state.vues_utilisateur[user_id]:
                st.session_state.vues_utilisateur[user_id].append(premier_pack_id)

            afficher_packs(result_packs, f"📦 Packs suggérés")
        else:
            st.warning("Aucun pack trouvé pour cette requête.")

    

    recs_pop = recommend_popular(top_n)
    afficher_packs(recs_pop, "🔥 Packs les plus populaires")
    
    packs_interacted_by_user = get_packs_interacted_by_user(user_id, interactions, products)
    #if st.session_state.vues_utilisateur[user_id] or not packs_interacted_by_user.empty :
    if st.session_state.vues_utilisateur[user_id] or not packs_interacted_by_user.empty:
      st.markdown("### 👁️ Packs consultés + vus récemment")
      historique_df = products[products["pack_id"].isin(st.session_state.vues_utilisateur[user_id])]
      packs_concat = pd.concat([historique_df, packs_interacted_by_user]).drop_duplicates(subset='pack_id')

      afficher_packs(historique_df, "Historique personnalisé")

    afficher_packs(packs_suggérés, f"🎁 Packs similaires à la catégorie préférée de {user_id} : `{categorie_preferee}`")
   
