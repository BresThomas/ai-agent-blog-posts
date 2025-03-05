# ai_blog_app.py

import os
import re
import streamlit as st
from typing import List, Tuple
from datetime import date

import git
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

## streamlit run ai_blog_app.py

###############################################################################
# UTILS
###############################################################################

def remove_leading_number(title: str) -> str:
    return re.sub(r'^\d+\.?\s*', '', title).strip()

def create_slug_from_title(title: str) -> str:
    words = re.split(r'\W+', title.lower())
    words = [w for w in words if w]
    return '-'.join(words)

def escape_yaml_string(text: str) -> str:
    return text.replace('"', r'\"').replace('\n', ' ').strip()

###############################################################################
# TITLE GENERATION
###############################################################################

title_prompt = PromptTemplate(
    input_variables=["niche", "comment"],
    template=(
        "You are a creative blog strategist. Given the user niche or topic: '{niche}'\n"
        "And the user's additional comment or instruction: '{comment}'\n"
        "Propose 5 different, catchy blog post titles. Be concise and avoid numbering in the final text if possible."
    )
)

def generate_titles(llm: ChatOpenAI, niche: str, comment: str = "") -> List[str]:
    chain = LLMChain(llm=llm, prompt=title_prompt)
    raw_output = chain.run(niche=niche, comment=comment)
    return [remove_leading_number(line.strip()) for line in raw_output.split("\n") if line.strip()]

###############################################################################
# PIPELINE STEPS
###############################################################################

# 1) Outline & Structure
outline_prompt = PromptTemplate(
    input_variables=["chosen_title"],
    template=(
        "Create a detailed blog post outline for the title: {chosen_title}. "
        "Include headings, subheadings, and bullet points."
    )
)

def generate_outline(llm: ChatOpenAI, chosen_title: str) -> str:
    return LLMChain(llm=llm, prompt=outline_prompt).run(chosen_title=chosen_title)

# 2) Draft-Writing
draft_prompt = PromptTemplate(
    input_variables=["outline", "brand_voice"],
    template=(
        "Write a first draft for the blog post based on this outline:\n{outline}\n\n"
        "Write in a {brand_voice} tone. Aim for 800-1200 words with proper markdown formatting."
    )
)

def write_draft(llm: ChatOpenAI, outline: str, brand_voice: str) -> str:
    return LLMChain(llm=llm, prompt=draft_prompt).run(outline=outline, brand_voice=brand_voice)

# 3) SEO Optimization
seo_prompt = PromptTemplate(
    input_variables=["draft", "keywords"],
    template=(
        "Optimize this draft for SEO:\n{draft}\n\n"
        "Incorporate these keywords naturally: {keywords}. "
        "Provide:\n1. Optimized content\n2. Suggested meta title (prefix with 'Suggested meta title: ')\n"
        "3. Suggested meta description (prefix with 'Suggested meta description: ')"
    )
)

def optimize_for_seo(llm: ChatOpenAI, draft: str, keywords: List[str]) -> Tuple[str, str, str]:
    raw_output = LLMChain(llm=llm, prompt=seo_prompt).run(
        draft=draft, 
        keywords=", ".join(keywords)
    )
    
    meta_title = None
    meta_description = None
    content_lines = []
    
    for line in raw_output.split("\n"):
        line = line.strip()
        if line.startswith("Suggested meta title:"):
            meta_title = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("Suggested meta description:"):
            meta_description = line.split(":", 1)[1].strip().strip('"')
        elif line:
            content_lines.append(line)
    
    return "\n".join(content_lines), meta_title, meta_description

# 4) Fact-Check & Edit
edit_prompt = PromptTemplate(
    input_variables=["draft"],
    template=(
        "Edit and fact-check this article:\n{draft}\n\n"
        "1. Fix grammar/spelling\n2. Flag uncited claims\n3. Improve clarity\n"
        "Return ONLY the edited text."
    )
)

def edit_and_fact_check(llm: ChatOpenAI, draft: str) -> str:
    return LLMChain(llm=llm, prompt=edit_prompt).run(draft=draft)

# 5) Generate excerpt & tags
excerpt_tags_prompt = PromptTemplate(
    input_variables=["final_article"],
    template=(
        "Generate for this article:\n{final_article}\n\n"
        "1. A 1-2 sentence excerpt\n2. 5 tags\n\n"
        "Format:\nExcerpt: <text>\nTags: comma, separated, list"
    )
)

def generate_excerpt_and_tags(llm: ChatOpenAI, final_article: str) -> Tuple[str, List[str]]:
    response = LLMChain(llm=llm, prompt=excerpt_tags_prompt).run(final_article=final_article)
    excerpt_match = re.search(r'Excerpt:\s*(.+?)(?=\nTags:|$)', response, re.DOTALL)
    tags_match = re.search(r'Tags:\s*(.+)', response)
    
    excerpt = excerpt_match.group(1).strip() if excerpt_match else ""
    tags = [t.strip() for t in tags_match.group(1).split(",")] if tags_match else []
    return excerpt, tags[:5]

# 6) Translation
translate_prompt = PromptTemplate(
    input_variables=["text", "target_language"],
    template=(
        "Translate to {target_language}:\n{text}\n\n"
        "Preserve meaning, style, and formatting."
    )
)

def translate_text(llm: ChatOpenAI, text: str, target_language: str) -> str:
    return LLMChain(llm=llm, prompt=translate_prompt).run(
        text=text, 
        target_language=target_language
    )

# 7) Markdown Creation
def create_markdown_content(
    title: str,
    content: str,
    excerpt: str,
    cover_image_url: str,
    tags: List[str],
    meta_title: str = None,
    meta_description: str = None
) -> str:
    front_matter = {
        "title": escape_yaml_string(meta_title or title),
        "date": date.today().isoformat(),
        "excerpt": escape_yaml_string(meta_description or excerpt),
        "coverImage": cover_image_url,
        "tags": [escape_yaml_string(tag) for tag in tags]
    }
    
    yaml_lines = []
    for k, v in front_matter.items():
        if isinstance(v, list):
            # Build the formatted list without nested f-strings
            formatted_list = ", ".join('"' + item + '"' for item in v)
            yaml_lines.append(f'{k}: [{formatted_list}]')
        else:
            yaml_lines.append(f'{k}: "{v}"')
    
    return f"---\n" + "\n".join(yaml_lines) + f"\n---\n\n{content}"


# 8) File Handling
def store_local_markdown(
    repo_path: str,
    content_folder: str,
    language: str,
    slug: str,
    md_content: str
) -> str:
    lang_folder = "en" if language == "en" else f"{language}"
    folder_path = os.path.join(repo_path, content_folder, lang_folder)
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
# MAIN PIPELINE
###############################################################################

def run_article_pipeline(
    llm: ChatOpenAI,
    chosen_title: str,
    brand_voice: str,
    keywords: List[str],
    cover_image: str,
    repo_path: str,
    content_folder: str,
    languages: List[str]
) -> dict:
    base_title = remove_leading_number(chosen_title)
    slug = create_slug_from_title(base_title)
    
    # Generate content
    outline = generate_outline(llm, base_title)
    draft = write_draft(llm, outline, brand_voice)
    optimized_content, meta_title, meta_desc = optimize_for_seo(llm, draft, keywords)
    final_draft = edit_and_fact_check(llm, optimized_content)
    excerpt, tags = generate_excerpt_and_tags(llm, final_draft)
    
    # Handle translations
    created_files = []
    for lang in set(languages + ["en"]):  # Always include English
        if lang == "en":
            md_content = create_markdown_content(
                title=base_title,
                content=final_draft,
                excerpt=excerpt,
                cover_image_url=cover_image,
                tags=tags,
                meta_title=meta_title,
                meta_description=meta_desc
            )
        else:
            trans_title = translate_text(llm, base_title, lang)
            trans_content = translate_text(llm, final_draft, lang)
            trans_excerpt = translate_text(llm, excerpt, lang)
            trans_tags = [translate_text(llm, tag, lang) for tag in tags]
            
            md_content = create_markdown_content(
                title=trans_title,
                content=trans_content,
                excerpt=trans_excerpt,
                cover_image_url=cover_image,
                tags=trans_tags
            )
        
        fpath = store_local_markdown(repo_path, content_folder, lang, slug, md_content)
        created_files.append(fpath)
    
    return {
        "slug": slug,
        "outline": outline,
        "final_draft": final_draft,
        "created_files": created_files,
        "published_urls": [
            f"/blog/{slug}" if lang == "en" else f"/{lang}/blog/{slug}"
            for lang in languages
        ]
    }

###############################################################################
# STREAMLIT UI
###############################################################################

def main():
    st.title("AI Blog Creator with Proper Markdown Handling")
    
    # Configuration
    repo_path = st.text_input("Repository Path", os.path.expanduser("~/blog-content"))
    content_folder = st.text_input("Content Folder", "src/content")
    niche = st.text_input("Blog Topic", "AI Content Generation")
    comment = st.text_area("Additional Instructions")
    cover_image = st.text_input("Cover Image URL", "https://example.com/default-image.jpg")
    
    # Title Generation
    if st.button("Generate Titles"):
        llm = ChatOpenAI(temperature=0.7)
        st.session_state.titles = generate_titles(llm, niche, comment)
    
    if 'titles' in st.session_state:
        chosen_title = st.selectbox("Select Title", st.session_state.titles)
        st.session_state.chosen_title = remove_leading_number(chosen_title)
    
    # Article Settings
    if 'chosen_title' in st.session_state:
        brand_voice = st.selectbox("Brand Voice", ["Professional", "Casual", "Technical", "Friendly"])
        keywords = st.text_input("SEO Keywords (comma-separated)", "AI, Content Generation, SEO, Marketing")
        languages = st.multiselect("Languages", ["en", "es", "fr", "de"], default=["en"])
        
        if st.button("Generate Full Article"):
            llm = ChatOpenAI(temperature=0.3, model="gpt-4")
            
            with st.spinner("Generating article..."):
                results = run_article_pipeline(
                    llm=llm,
                    chosen_title=st.session_state.chosen_title,
                    brand_voice=brand_voice.lower(),
                    keywords=[k.strip() for k in keywords.split(",")],
                    cover_image=cover_image,
                    repo_path=repo_path,
                    content_folder=content_folder,
                    languages=languages
                )
                
            st.session_state.results = results
            st.success(f"Article generated at {results['created_files'][0]}")
            
            with st.expander("View Outline"):
                st.markdown(results["outline"])
                
            with st.expander("View Final Draft"):
                st.markdown(results["final_draft"])
            
            if st.button("Commit to Git"):
                try:
                    git_commit_and_push(repo_path)
                    st.success("Successfully committed and pushed!")
                except Exception as e:
                    st.error(f"Git error: {str(e)}")

if __name__ == "__main__":
    main()