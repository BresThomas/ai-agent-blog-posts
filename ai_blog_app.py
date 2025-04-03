import os
import re
import json
import hashlib
import requests
import streamlit as st
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import git
import yaml
from dateutil import parser as date_parser

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


###############################################################################
# CONSTANTES
###############################################################################
CACHE_DIR = "cache_html"
INDEX_FILE = "index_data.json"
MAX_WORKERS = 10

os.makedirs(CACHE_DIR, exist_ok=True)

###############################################################################
# GESTION FICHIER D'INDEX
###############################################################################
def load_index_data() -> dict:
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_index_data(data: dict):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

###############################################################################
# DETECTER UN FORMAT DE DATE
###############################################################################
def detect_date_format(date_str: str) -> str:
    """
    Essaie de détecter le format de date dans le YAML d'exemple.
    Si échec, on renvoie un format par défaut.
    (Cette fonction est rudimentaire et peut être améliorée selon vos besoins.)
    """
    # Liste de formats communs que vous voulez supporter
    possible_formats = [
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        # etc.
    ]
    for fmt in possible_formats:
        try:
            datetime.strptime(date_str, fmt)
            return fmt
        except:
            pass
    # Si on ne trouve rien, on met un format par défaut
    return "%Y-%m-%d %H:%M:%S %z"

###############################################################################
# TELECHARGEMENT AVEC CACHE
###############################################################################
def url_to_filename(url: str) -> str:
    h = hashlib.md5(url.encode('utf-8')).hexdigest()
    return f"{h}.html"

def fetch_html_content_cached(url: str, lastmod: Optional[str], old_lastmod: Optional[str]) -> str:
    filepath = os.path.join(CACHE_DIR, url_to_filename(url))
    
    need_download = False
    if old_lastmod != lastmod:
        need_download = True
    elif not os.path.exists(filepath):
        need_download = True

    if need_download:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                html = resp.text
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html)
            else:
                html = ""
        except:
            html = ""
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")

###############################################################################
# PARSER SITEMAP
###############################################################################
def parse_sitemap(sitemap_url: str) -> List[dict]:
    results = []
    try:
        resp = requests.get(sitemap_url, timeout=10)
        soup = BeautifulSoup(resp.content, "xml")
        url_tags = soup.find_all("url")
        for tag in url_tags:
            loc = tag.find("loc")
            mod = tag.find("lastmod")
            if loc:
                results.append({
                    "url": loc.text.strip(),
                    "lastmod": mod.text.strip() if mod else None
                })
    except:
        pass
    return results

###############################################################################
# CONSTRUCTION ARBRE
###############################################################################
from urllib.parse import urlparse

def get_domain_and_path(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path or "/"
    if path == "":
        path = "/"
    return domain, path

def build_url_tree(entries: List[dict]) -> dict:
    tree = {}
    for e in entries:
        url = e["url"]
        lastmod = e["lastmod"]
        domain, path = get_domain_and_path(url)

        if domain not in tree:
            tree[domain] = {"children": {}, "urls": {}}

        parts = path.strip("/").split("/")
        if parts == ['']:
            parts = []

        current_node = tree[domain]
        for i, seg in enumerate(parts):
            if i == len(parts) - 1:
                if seg == '':
                    seg = '(root)'
                current_node["urls"][seg] = {"url": url, "lastmod": lastmod}
            else:
                if seg == '':
                    seg = '(root)'
                if seg not in current_node["children"]:
                    current_node["children"][seg] = {"children": {}, "urls": {}}
                current_node = current_node["children"][seg]
    return tree

def collect_urls(node: dict) -> List[str]:
    urls = []
    for meta in node["urls"].values():
        urls.append(meta["url"])
    for child_node in node["children"].values():
        urls.extend(collect_urls(child_node))
    return urls

def render_domain(domain: str, node: dict, selected_urls: set):
    with st.expander(domain, expanded=False):
        render_subtree_recursive(node, indent="", selected_urls=selected_urls, path=[domain])

def collect_selected_children(node: dict, selected_urls: set) -> List[str]:
    selected = []
    for seg, meta in node.get("urls", {}).items():
        if meta["url"] in selected_urls:
            selected.append(seg if seg != "(root)" else "/")
    for _, child_node in node.get("children", {}).items():
        selected.extend(collect_selected_children(child_node, selected_urls))
    return selected

def render_subtree_recursive(node: dict, indent: str, selected_urls: set, path: List[str]):
    for folder_name, child_node in node.get("children", {}).items():
        display_name = "/" if folder_name == "(root)" else folder_name
        current_path = path + [display_name]
        folder_key = "/".join(current_path)
        if folder_key not in st.session_state:
            st.session_state[folder_key] = False

        folder_checked = st.checkbox(
            f"{indent}📁 {folder_key}",
            value=st.session_state[folder_key],
            key=folder_key
        )
        
        folder_urls = collect_urls(child_node)
        if folder_checked:
            for url in folder_urls:
                selected_urls.add(url)
        else:
            for url in folder_urls:
                selected_urls.discard(url)

        new_indent = indent + "    "
        render_subtree_recursive(child_node, new_indent, selected_urls, current_path)

    for seg, meta in node.get("urls", {}).items():
        label = meta["url"]
        url = meta["url"]
        checked = url in selected_urls
        new_val = st.checkbox(f"{indent}📄 {label}", value=checked, key=url)
        if new_val:
            selected_urls.add(url)
        else:
            selected_urls.discard(url)

###############################################################################
# INDEXATION
###############################################################################
def incremental_index_selected(urls: List[dict]) -> List[str]:
    index_data = load_index_data()

    def process_entry(entry):
        url = entry["url"]
        lastmod = entry["lastmod"]
        old_lastmod = index_data.get(url, None)
        return fetch_html_content_cached(url, lastmod, old_lastmod)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        texts = list(executor.map(process_entry, urls))

    for e, txt in zip(urls, texts):
        index_data[e["url"]] = e["lastmod"]
    save_index_data(index_data)
    return texts

###############################################################################
# RAG UTILS
###############################################################################
def tfidf_top_k_pages(pages_content: List[str], query: str, k=5) -> List[int]:
    if not pages_content:
        return []
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(pages_content)
    query_vec = vectorizer.transform([query])
    scores = (X * query_vec.T).toarray().ravel()
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices[:k]

def embed_chunks_and_retrieve_top(chunks: List[str], query: str, top_k: int = 3) -> List[str]:
    if not chunks:
        return []
    embedder = OpenAIEmbeddings()
    query_emb = embedder.embed_query(query)

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    chunk_embs = embedder.embed_documents(chunks)
    sims = [cos_sim(query_emb, emb) for emb in chunk_embs]
    sorted_indices = np.argsort(sims)[::-1]
    top_indices = sorted_indices[:top_k]
    return [chunks[i] for i in top_indices]

def get_rag_context_on_demand(all_pages_text: List[str], user_query: str,
                              top_pages=5, top_chunks=3) -> str:
    if not all_pages_text:
        return ""
    page_indices = tfidf_top_k_pages(all_pages_text, user_query, k=top_pages)
    selected_texts = [all_pages_text[i] for i in page_indices]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_chunks = []
    for txt in selected_texts:
        splitted = splitter.split_text(txt)
        all_chunks.extend(splitted)

    best_chunks = embed_chunks_and_retrieve_top(all_chunks, user_query, top_k=top_chunks)
    return "\n\n".join(best_chunks)

###############################################################################
# PROMPTS + PIPELINE
###############################################################################
def format_prompt_and_invoke(llm: ChatOpenAI, prompt_text: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return response.content

def remove_leading_number(title: str) -> str:
    return re.sub(r'^\d+\.?\s*', '', title).strip()

def create_slug_from_title(title: str) -> str:
    words = re.split(r'\W+', title.lower())
    words = [w for w in words if w]
    return '-'.join(words)

# 1) Title
title_template = (
    "You are a creative blog strategist. Given the user niche or topic: '{niche}'\n"
    "And the user's additional comment or instruction: '{comment}'\n"
    "Propose 5 different, catchy blog post titles. Be concise."
)
def generate_titles(llm: ChatOpenAI, niche: str, comment: str):
    prompt_text = title_template.format(niche=niche, comment=comment)
    raw_output = format_prompt_and_invoke(llm, prompt_text)
    return [
        remove_leading_number(line.strip()) 
        for line in raw_output.split("\n") 
        if line.strip()
    ]

# 2) Outline
outline_template = (
    "You have two sources of context:\n\n"
    "1) The RAG context from the website:\n{rag_context}\n\n"
    "2) An external text to inspire how to approach the subject:\n{external_text}\n\n"
    "Using only the useful and relevant information that enhances our article, and ignoring any non-essential details "
    "(for example, names of YouTube channels, names of speakers, etc.), create a detailed blog post outline for the title: {chosen_title}. "
    "The outline should include headings, subheadings, and bullet points to clearly organize the ideas."
)
def generate_outline(llm: ChatOpenAI, chosen_title: str, rag_context: str, external_text: str) -> str:
    prompt_text = outline_template.format(
        chosen_title=chosen_title,
        rag_context=rag_context,
        external_text=external_text
    )
    return format_prompt_and_invoke(llm, prompt_text)

# 3) Draft
draft_template = (
    "Write a first draft based on this outline:\n{outline}\n\n"
    "Incorporate relevant info from:\n{rag_context}\n\n"
    "Also, be inspired by:\n{external_text}\n\n"
    "Write in a {brand_voice} tone, ~1000 words, with markdown formatting."
)
def write_draft(llm: ChatOpenAI, outline: str, brand_voice: str, rag_context: str, external_text: str) -> str:
    prompt_text = draft_template.format(
        outline=outline,
        brand_voice=brand_voice,
        rag_context=rag_context,
        external_text=external_text
    )
    return format_prompt_and_invoke(llm, prompt_text)

# 4) SEO
seo_template = (
    "Optimize this draft for SEO:\n{draft}\n\n"
    "Incorporate these keywords naturally: {keywords}. "
    "Provide:\n1. Optimized content\n2. Suggested meta title (prefix with 'Suggested meta title: ')\n"
    "3. Suggested meta description (prefix with 'Suggested meta description: ')"
)
def optimize_for_seo(llm: ChatOpenAI, draft: str, keywords: List[str]):
    prompt_text = seo_template.format(draft=draft, keywords=", ".join(keywords))
    raw_output = format_prompt_and_invoke(llm, prompt_text)
    
    meta_title = None
    meta_desc = None
    content_lines = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if line.startswith("Suggested meta title:"):
            meta_title = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("Suggested meta description:"):
            meta_desc = line.split(":", 1)[1].strip().strip('"')
        elif line:
            content_lines.append(line)
    return "\n".join(content_lines), meta_title, meta_desc

# 5) Fact-check
edit_template = (
    "Edit and fact-check this article:\n{draft}\n\n"
    "1. Fix grammar/spelling\n2. Flag uncited claims\n3. Improve clarity\n"
    "Return ONLY the edited text."
)
def edit_and_fact_check(llm: ChatOpenAI, draft: str):
    prompt_text = edit_template.format(draft=draft)
    return format_prompt_and_invoke(llm, prompt_text)

# 6) Excerpt & Tags
excerpt_template = (
    "Generate for this article:\n{final_article}\n\n"
    "1. A 1-2 sentence excerpt\n2. 5 tags\n\n"
    "Format:\nExcerpt: <text>\nTags: comma, separated, list"
)
def generate_excerpt_and_tags(llm: ChatOpenAI, final_article: str):
    prompt_text = excerpt_template.format(final_article=final_article)
    raw_output = format_prompt_and_invoke(llm, prompt_text)
    excerpt_match = re.search(r'Excerpt:\s*(.+?)(?=\nTags:|$)', raw_output, re.DOTALL)
    tags_match = re.search(r'Tags:\s*(.+)', raw_output)
    
    excerpt = excerpt_match.group(1).strip() if excerpt_match else ""
    tags = [t.strip() for t in tags_match.group(1).split(",")] if tags_match else []
    return excerpt, tags[:5]


###############################################################################
# GESTION DYNAMIQUE DU YAML
###############################################################################
def build_dynamic_yaml_fields(chosen_title: str):
    """
    - On affiche des inputs pour tous les champs sauf :
        excerpt, tags, categories, lang, date
      car ceux-là doivent être gérés automatiquement
    - Si 'title' est présent, on met la valeur = chosen_title par défaut
    - On stocke toutes les valeurs dans st.session_state["user_yaml_inputs"]
    - On détecte le format de date pour usage ultérieur
    """
    if "parsed_yaml_example" not in st.session_state:
        return

    data = st.session_state["parsed_yaml_example"]
    if "user_yaml_inputs" not in st.session_state:
        st.session_state["user_yaml_inputs"] = {}
    user_inputs = st.session_state["user_yaml_inputs"]

    # Initialiser la détection de format de date
    if "yaml_date_format" not in st.session_state:
        st.session_state["yaml_date_format"] = "%Y-%m-%d"  # fallback

    st.markdown("#### Fill or adjust the YAML fields below (some are automatic):")

    for field_name, field_value in data.items():
        field_lower = field_name.lower()
        
        # EXCLUDE these from manual input
        if field_lower in ["excerpt", "tags", "categories", "lang"]:
            continue
        
        # date => on ne fait pas de champ input. On détecte le format
        if field_lower == "date":
            # Essayer de détecter le format
            date_str = str(field_value)
            st.session_state["yaml_date_format"] = detect_date_format(date_str)
            # On ne propose pas de saisie, tout sera auto-généré
            continue

        # title => on fait un input prérempli avec chosen_title
        if field_lower == "title":
            user_inputs[field_name] = st.text_input(field_name, value=chosen_title)
            continue

        # Autres cas => simple text_input
        user_inputs[field_name] = st.text_input(field_name, value=str(field_value))


def build_front_matter_for_lang(
    lang: str,
    chosen_title: str,
    excerpt: str,
    tags: List[str],
    meta_title: Optional[str],
    meta_desc: Optional[str]
):
    """
    Construit le YAML final en se basant sur:
    - st.session_state["parsed_yaml_example"]
    - st.session_state["user_yaml_inputs"]
    - Date format stocké dans st.session_state["yaml_date_format"]
    - Remplace excerpt, tags, lang, date automatiquement
    - Remplace le title par meta_title ou chosen_title si 'title' est présent
    """
    if "parsed_yaml_example" not in st.session_state:
        return ""
    if "user_yaml_inputs" not in st.session_state:
        return ""

    data = dict(st.session_state["parsed_yaml_example"])  # copie
    user_inputs = st.session_state["user_yaml_inputs"]
    date_format = st.session_state.get("yaml_date_format", "%Y-%m-%d")

    # 1) Appliquer tout ce que l'utilisateur a saisi
    for k, v in user_inputs.items():
        # Sauf s'il s'agit de date, excerpt, etc. => on gère plus bas
        if k.lower() in ["date", "excerpt", "tags", "categories", "lang"]:
            continue
        # On met la valeur
        data[k] = v

    # 2) Gérer la date => on met la date du jour dans le format détecté
    now = datetime.now()
    try:
        data["date"] = now.strftime(date_format)
    except:
        # fallback si le format est incompatible
        data["date"] = now.strftime("%Y-%m-%d %H:%M:%S")

    # 3) Gérer excerpt & tags => on force
    data["excerpt"] = meta_desc or excerpt
    data["tags"] = tags

    # 4) Gérer lang => on force la langue
    data["lang"] = lang

    # 5) Gérer le title => si présent
    found_title_key = None
    for key in data.keys():
        if key.lower() == "title":
            found_title_key = key
            break
    if found_title_key:
        data[found_title_key] = meta_title or chosen_title

    # 6) On reconstruit en YAML
    front_matter = yaml.dump(data, sort_keys=False, allow_unicode=True)
    return f"---\n{front_matter}---\n"


###############################################################################
# FONCTIONS I/O
###############################################################################
def store_local_markdown(repo_path: str, content_folder: str, language: str, slug: str, md_content: str):
    folder_path = os.path.join(repo_path, content_folder, language)
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"{slug}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    return file_path

def git_commit_and_push(repo_path: str, commit_message: str = "Add new blog post"):
    repo = git.Repo(repo_path)
    repo.git.add("--all")
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    origin.push()

###############################################################################
# PIPELINE
###############################################################################
def run_article_pipeline(
    llm: ChatOpenAI,
    chosen_title: str,
    brand_voice: str,
    keywords: List[str],
    rag_context: str,
    external_text: str
):
    base_title = remove_leading_number(chosen_title)
    slug = create_slug_from_title(base_title)
    
    outline = generate_outline(llm, base_title, rag_context, external_text)
    draft = write_draft(llm, outline, brand_voice, rag_context, external_text)
    optimized_content, meta_title, meta_desc = optimize_for_seo(llm, draft, keywords)
    final_draft = edit_and_fact_check(llm, optimized_content)
    excerpt, tags = generate_excerpt_and_tags(llm, final_draft)
    
    return {
        "slug": slug,
        "outline": outline,
        "final_draft": final_draft,
        "excerpt": excerpt,
        "tags": tags,
        "meta_title": meta_title,
        "meta_desc": meta_desc
    }


###############################################################################
# STREAMLIT APP
###############################################################################
def main():
    st.title("AI Blog Creator")

    #
    # 1) SITEMAP & INDEX
    #
    sitemap_url = st.text_input("Sitemap URL", "https://example.com/sitemap.xml")
    if st.button("Charger la sitemap"):
        st.session_state.sitemap_entries = parse_sitemap(sitemap_url)
        st.session_state.url_tree = build_url_tree(st.session_state.sitemap_entries)
        st.success(f"Sitemap chargée ! Nombre de pages : {len(st.session_state.sitemap_entries)}")

    if 'url_tree' not in st.session_state:
        st.warning("Veuillez d'abord charger la sitemap.")
        return

    st.markdown("### Sélectionnez les URLs à indexer")
    if 'selected_urls' not in st.session_state:
        st.session_state.selected_urls = set()

    for domain, node in st.session_state.url_tree.items():
        render_domain(domain, node, st.session_state.selected_urls)

    st.markdown(f"**URLs sélectionnées :** {len(st.session_state.selected_urls)}")

    if st.button("Indexer la sélection"):
        selected_dicts = []
        for e in st.session_state.sitemap_entries:
            if e["url"] in st.session_state.selected_urls:
                selected_dicts.append(e)
        
        if not selected_dicts:
            st.warning("Aucune URL sélectionnée.")
            return
        
        with st.spinner("Indexation en cours..."):
            all_texts = incremental_index_selected(selected_dicts)
        st.session_state.all_pages_text = all_texts
        st.success(f"Indexation terminée. {len(all_texts)} pages traitées.")

    if 'all_pages_text' not in st.session_state:
        st.info("Une fois l'indexation terminée, passez à la suite.")
        return

    #
    # 2) GENERER TITRES
    #
    st.markdown("## Génération de Titres")
    external_text = st.text_area("Optional external text", "")
    niche = st.text_input("Blog Topic (Niche)", "AI Content Generation")
    comment = st.text_area("Additional Instructions", "")
    brand_voice = st.selectbox("Brand Voice", ["Professional", "Casual", "Technical", "Friendly"])
    keywords = st.text_input("SEO Keywords (comma-separated)", "AI, Content Generation, SEO")
    languages = st.multiselect("Languages", ["en", "fr", "es", "de"], default=["en"])

    if st.button("Générer des Titres"):
        llm_titles = ChatOpenAI(temperature=0.7)
        st.session_state.titles = generate_titles(llm_titles, niche, comment)

    if 'titles' in st.session_state:
        chosen_title = st.selectbox("Choisissez un Titre", st.session_state.titles)
        st.session_state.chosen_title = remove_leading_number(chosen_title)

    #
    # 3) PARSE YAML
    #
    st.markdown("## YAML Template Configuration")
    sample_yaml = st.text_area("Paste a sample YAML front matter here", value="""---
layout: post
title: "My Post Title"
date: 2024-07-18 01:00:00 +0200
categories: update
lang: en
author: Damien
code: blog
---
""")

    if st.button("Parse YAML"):
        content = sample_yaml.strip()
        if content.startswith("---"):
            content = content[3:].strip()
        if content.endswith("---"):
            content = content[:-3].strip()
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                data = {}
            st.session_state["parsed_yaml_example"] = data
            st.success("YAML parsed successfully.")
        except Exception as e:
            st.error(f"Error parsing YAML: {e}")

    # Afficher les inputs pour les champs, sauf date/excerpt/tags/categories/lang
    if "parsed_yaml_example" in st.session_state and "chosen_title" in st.session_state:
        build_dynamic_yaml_fields(chosen_title=st.session_state.chosen_title)

    #
    # 4) GENERATION DE L'ARTICLE
    #
    st.markdown("## Générer l'article complet")
    repo_path = st.text_input("Repository Path", os.path.expanduser("~/blog-content"))
    content_folder = st.text_input("Content Folder", "src/content")

    if ('chosen_title' in st.session_state) and ('parsed_yaml_example' in st.session_state):
        if st.button("Générer l'article complet"):
            llm = ChatOpenAI(temperature=0.3, model="gpt-4")
            user_query = (niche + " " + comment).strip()
            
            with st.spinner("Fetching RAG context..."):
                rag_context = get_rag_context_on_demand(
                    all_pages_text=st.session_state.all_pages_text,
                    user_query=user_query,
                    top_pages=5,
                    top_chunks=3
                )
            
            with st.spinner("Generating article..."):
                results = run_article_pipeline(
                    llm=llm,
                    chosen_title=st.session_state.chosen_title,
                    brand_voice=brand_voice.lower(),
                    keywords=[k.strip() for k in keywords.split(",")],
                    rag_context=rag_context,
                    external_text=external_text
                )
            st.session_state["results"] = results
            st.success("Article generated!")

            # On construit un YAML + le draft pour la première langue en guise d'aperçu
            if languages:
                preview_lang = languages[0]
            else:
                preview_lang = "en"

            fm_preview = build_front_matter_for_lang(
                lang=preview_lang,
                chosen_title=st.session_state.chosen_title,
                excerpt=results["excerpt"],
                tags=results["tags"],
                meta_title=results["meta_title"],
                meta_desc=results["meta_desc"]
            )
            combined = fm_preview + "\n" + results["final_draft"]
            st.session_state["final_article_preview"] = combined

        if "results" in st.session_state and "final_article_preview" in st.session_state:
            st.markdown("### Final Article")
            st.session_state["final_article_preview"] = st.text_area(
                "You can edit the final article (front matter + content) here:",
                value=st.session_state["final_article_preview"],
                height=500
            )

            if st.button("Enregistrer et Pousser sur Git"):
                results = st.session_state["results"]
                slug = results["slug"]
                created_files = []
                llm = ChatOpenAI(temperature=0.3, model="gpt-4")

                for lang in set(languages):
                    # Si != en, on traduit
                    if lang != "en":
                        translation_prompt = f"Translate this text to {lang}:\n\n{results['final_draft']}"
                        translated_draft = format_prompt_and_invoke(llm, translation_prompt)

                        translation_prompt2 = f"Translate this text to {lang}:\n\n{results['excerpt']}"
                        translated_excerpt = format_prompt_and_invoke(llm, translation_prompt2)

                        translated_tags = []
                        for t in results["tags"]:
                            t_ = format_prompt_and_invoke(llm, f"Translate this to {lang}:\n\n{t}")
                            translated_tags.append(t_)

                        # Reconstruit le front matter pour cette langue
                        fm = build_front_matter_for_lang(
                            lang=lang,
                            chosen_title=st.session_state.chosen_title,
                            excerpt=translated_excerpt,
                            tags=translated_tags,
                            meta_title=results["meta_title"],
                            meta_desc=results["meta_desc"]
                        )
                        md_content = fm + "\n" + translated_draft
                    else:
                        # Utilise l'aperçu modifié par l'utilisateur
                        md_content = st.session_state["final_article_preview"]

                    fpath = store_local_markdown(repo_path, content_folder, lang, slug, md_content)
                    created_files.append(fpath)

                try:
                    git_commit_and_push(repo_path)
                    st.success(f"Success! Files stored: {created_files}. Committed & pushed to Git.")
                except Exception as e:
                    st.error(f"Git error: {str(e)}")

    else:
        st.info("Choisissez un titre et parsez le YAML pour débloquer la génération complète.")


if __name__ == "__main__":
    main()
