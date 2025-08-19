#!/usr/bin/env python3
"""
PDF to HTML Converter - Version Intégrée avec Transfert de Style Neuronal
Application Streamlit complète pour convertir des PDF en HTML optimisé SEO/AEO,
avec la capacité d'appliquer un style visuel et de personnaliser le prompt d'analyse IA.
Utilise les modèles IA de pointe : GPT-4o et Claude 3.5 Sonnet.
"""

# --- Imports Standard et Essentiels ---
import streamlit as st
import asyncio
import json
import tempfile
import base64
from pathlib import Path
from datetime import datetime
import logging
import re
import io
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import colorsys

# --- Configuration de la Page Streamlit (doit être la première commande st) ---
st.set_page_config(
    page_title="PDF to HTML Style Converter",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dépendances Majeures (avec gestion d'erreurs) ---
try:
    import aiohttp
    from bs4 import BeautifulSoup
    import tinycss2
    import webcolors
    import fitz  # PyMuPDF
    from PIL import Image
    import numpy as np
    from sklearn.cluster import KMeans
    import openai
    from anthropic import AsyncAnthropic
    from jinja2 import Template
except ImportError as e:
    st.error(f"Une bibliothèque essentielle est manquante : {e}")
    st.error("Veuillez installer toutes les dépendances requises avec la commande suivante :")
    st.code("pip install streamlit aiohttp beautifulsoup4 tinycss2 webcolors PyMuPDF Pillow numpy scikit-learn openai anthropic jinja2", language="bash")
    st.stop()

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constante pour le prompt par défaut ---
DEFAULT_PROMPT = """Analyse le contenu du document PDF suivant et génère une structure sémantique et SEO optimisée.
Texte : {text_sample}

Retourne UNIQUEMENT un objet JSON valide avec la structure exacte suivante :
{
  "title": "Un titre SEO concis et percutant (max 60 caractères).",
  "meta_description": "Une méta-description engageante (max 160 caractères).",
  "h1": "Le titre principal de la page (H1).",
  "sections": [
    {
      "title": "Le titre d'une section (pour un H2).",
      "content": "Un paragraphe résumé de la section."
    }
  ],
  "keywords": ["Une", "liste", "de", "5 à 7", "mots-clés", "pertinents"],
  "faq": [
    {
      "question": "Une question fréquente sur le sujet.",
      "answer": "La réponse à cette question."
    }
  ]
}
"""

# --- SECTION 1: BIBLIOTHÈQUE D'ANALYSE DE STYLE ---

@dataclass
class StyleFingerprint:
    """Empreinte stylistique extraite d'une page web."""
    color_palette: List[str]
    typography: Dict[str, Any]
    design_mood: str = "N/A"
    confidence_score: float = 0.0

class WebStyleAnalyzer:
    """
    Analyseur de style web. Extrait les couleurs et la typographie d'une URL
    pour créer une empreinte stylistique.
    """
    _COLOR_HEX_PATTERN = re.compile(r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})\b')
    _COLOR_RGB_PATTERN = re.compile(r'rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)')
    _FONT_FAMILY_PATTERN = re.compile(r'font-family\s*:\s*([^;]+)', re.IGNORECASE)

    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (compatible; StyleAnalyzer/1.0)'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()

    async def analyze_reference_page(self, url: str) -> StyleFingerprint:
        logger.info(f"Début de l'analyse de style pour l'URL : {url}")
        html_content, css_contents = await self._fetch_page_with_resources(url)
        soup = BeautifulSoup(html_content, 'html.parser')
        color_palette = self._extract_color_palette(css_contents, soup)
        typography = self._analyze_typography(css_contents)
        return StyleFingerprint(
            color_palette=color_palette,
            typography=typography,
            confidence_score=self._calculate_confidence(color_palette, typography)
        )

    async def _fetch_page_with_resources(self, url: str) -> Tuple[str, List[str]]:
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                html_content = await response.text(encoding='utf-8', errors='replace')
        except aiohttp.ClientError as e:
            raise ValueError(f"Impossible de récupérer la page principale {url}: {e}")

        soup = BeautifulSoup(html_content, 'html.parser')
        css_links = [link.get('href') for link in soup.find_all('link', rel='stylesheet') if link.get('href')]
        tasks = [self._fetch_resource(urljoin(url, href)) for href in css_links[:15]]
        css_results = await asyncio.gather(*tasks)
        all_css = [res for res in css_results if res]
        all_css.extend(style.string for style in soup.find_all('style') if style.string)
        return html_content, all_css

    async def _fetch_resource(self, url: str) -> Optional[str]:
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                return await response.text(encoding='utf-8', errors='replace')
        except Exception as e:
            logger.warning(f"Impossible de récupérer la ressource {url}: {e}")
            return None

    def _extract_color_palette(self, css_contents: List[str], soup: BeautifulSoup) -> List[str]:
        colors = set()
        text_to_scan = "\n".join(css_contents) + "".join(el['style'] for el in soup.find_all(style=True))
        colors.update(m.group(0).lower() for m in self._COLOR_HEX_PATTERN.finditer(text_to_scan))
        for m in self._COLOR_RGB_PATTERN.finditer(text_to_scan):
            try:
                colors.add(webcolors.rgb_to_hex(tuple(map(int, m.groups()))))
            except ValueError: pass
        if not colors: return []
        hex_colors_for_clustering = [webcolors.hex_to_rgb(c) for c in colors if len(c) == 7]
        if len(hex_colors_for_clustering) < 3: return sorted(list(colors))[:10]
        pixels = np.array(hex_colors_for_clustering)
        n_clusters = min(8, len(pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(pixels)
        return sorted([webcolors.rgb_to_hex(tuple(map(int, center))) for center in kmeans.cluster_centers_])

    def _analyze_typography(self, css_contents: List[str]) -> Dict[str, Any]:
        fonts = set()
        for css in css_contents:
            for match in self._FONT_FAMILY_PATTERN.finditer(css):
                for family in match.group(1).split(','):
                    font = family.strip().strip("'\"").lower()
                    if font and font not in ['inherit', 'initial', 'sans-serif', 'serif', 'monospace']:
                        fonts.add(font.capitalize())
        return {'font_families': sorted(list(fonts))}

    def _calculate_confidence(self, palette: List, typo: Dict) -> float:
        score = 0.5 if len(palette) >= 3 else 0.0
        score += 0.5 if len(typo.get('font_families', [])) >= 1 else 0.0
        return score

    def _is_light(self, hex_color: str) -> bool:
        try:
            h = hex_color.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            return ((rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000) > 140
        except: return True

    def generate_transfer_css(self, fingerprint: StyleFingerprint) -> str:
        if fingerprint.confidence_score < 0.4: return "/* Confiance d'analyse de style trop faible. */"
        css = "/* --- Feuille de Style Transférée --- */\n:root {\n"
        if fingerprint.color_palette:
            primary = fingerprint.color_palette[0]
            css += f"    --theme-primary: {primary};\n"
            css += f"    --theme-secondary: {fingerprint.color_palette[1] if len(fingerprint.color_palette) > 1 else '#6c757d'};\n"
            css += f"    --theme-bg: #ffffff;\n    --theme-text: #212529;\n"
            css += f"    --theme-text-on-primary: {'#212529' if self._is_light(primary) else '#f8f9fa'};\n"
            css += f"    --theme-link: {fingerprint.color_palette[2] if len(fingerprint.color_palette) > 2 else primary};\n"
        css += "}\n\nbody.style-transferred {\n"
        if fingerprint.typography.get('font_families'):
            font_stack = ", ".join([f"'{f}'" for f in fingerprint.typography['font_families'][:2]])
            css += f"    font-family: {font_stack}, sans-serif;\n"
        css += "    background-color: var(--theme-bg);\n    color: var(--theme-text);\n}\n\n"
        css += ".style-transferred h1, .style-transferred h2, .style-transferred h3 { color: var(--theme-primary); }\n"
        css += ".style-transferred a { color: var(--theme-link); }\n"
        css += ".style-transferred .container { background: var(--theme-bg); }\n"
        css += ".style-transferred .header { border-bottom-color: var(--theme-primary); }\n"
        css += ".style-transferred .faq-section { border-left-color: var(--theme-secondary); }\n"
        return css

    def apply_style_transfer(self, target_html: str, fingerprint: StyleFingerprint) -> str:
        logger.info("Application du transfert de style.")
        css_rules = self.generate_transfer_css(fingerprint)
        soup = BeautifulSoup(target_html, 'html.parser')
        for style_tag in soup.find_all('style'): style_tag.decompose()
        new_style_tag = soup.new_tag('style')
        new_style_tag.string = css_rules
        if soup.head: soup.head.append(new_style_tag)
        else:
            head = soup.new_tag('head')
            soup.insert(0, head)
            head.append(new_style_tag)
        if soup.body: soup.body['class'] = soup.body.get('class', []) + ['style-transferred']
        return str(soup)


# --- SECTION 2: LOGIQUE DE CONVERSION PDF ---

@dataclass
class ConversionResult:
    html_content: str
    seo_metrics: Dict[str, Any]
    processing_time: float
    word_count: int
    style_fingerprint: Optional[StyleFingerprint] = None

class PDFConverterApp:
    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.openai_client = openai.AsyncOpenAI(api_key=openai_key) if openai_key else None
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None
        self.api_keys_provided = bool(openai_key or anthropic_key)

    def extract_pdf_content(self, pdf_file) -> Optional[Dict[str, Any]]:
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            content = {'text': "".join(page.get_text() for page in doc), 'pages': doc.page_count}
            doc.close()
            content['word_count'] = len(content['text'].split())
            return content
        except Exception as e:
            st.error(f"Erreur lors de l'extraction du contenu PDF : {e}")
            return None
    
    async def analyze_content_with_ai(self, content: Dict[str, Any], custom_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.api_keys_provided:
            st.error("Une clé API (OpenAI ou Anthropic) est requise. Veuillez en fournir une.")
            return None

        text_sample = content['text'][:8000]
        final_prompt = custom_prompt.format(text_sample=text_sample)

        try:
            if self.openai_client:
                st.info("Utilisation de l'API OpenAI (GPT-4o)...")
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": final_prompt}]
                )
                return json.loads(response.choices[0].message.content)
            elif self.anthropic_client:
                st.info("Utilisation de l'API Anthropic (Claude 3.5 Sonnet)...")
                message = await self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": final_prompt}]
                )
                return json.loads(message.content[0].text)
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API IA : {e}")
            return None
        return None

    def generate_html(self, analysis: Dict[str, Any]) -> str:
        template_str = """
<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ analysis.title }}</title><meta name="description" content="{{ analysis.meta_description }}">
<meta name="keywords" content="{{ analysis.get('keywords', []) | join(', ') }}">
<script type="application/ld+json">{"@context":"https://schema.org","@type":"Article","headline":"{{ analysis.h1 }}","description":"{{ analysis.meta_description }}","author":{"@type":"Organization","name":"PDF Converter"},"datePublished":"{{ current_date }}"}</script>
<style>body{font-family:sans-serif;line-height:1.6;max-width:900px;margin:0 auto;padding:20px;background:#fdfdff}.container{background:white;padding:30px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,.08)}.header{border-bottom:2px solid #007bff;padding-bottom:15px;margin-bottom:25px}h1,h2,h3{color:#333}.faq-section{background:#f8f9fa;padding:20px;border-radius:8px;margin-top:30px;border-left:4px solid #17a2b8}.faq-item{margin-bottom:15px}.faq-question{font-weight:bold}</style>
</head><body><div class="container"><header class="header"><h1>{{ analysis.h1 }}</h1></header><main>
{% for section in analysis.get('sections', []) %}<section><h2>{{ section.title }}</h2><p>{{ section.content | replace('\\n', '<br>') }}</p></section>{% endfor %}
</main>{% if analysis.get('faq', []) %}<section class="faq-section"><h2>Questions fréquentes</h2>
{% for item in analysis.get('faq', []) %}<div class="faq-item"><div class="faq-question">{{ item.question }}</div><div>{{ item.answer }}</div></div>{% endfor %}
</section>{% endif %}</div></body></html>"""
        return Template(template_str).render(analysis=analysis, current_date=datetime.now().strftime("%Y-%m-%d"))
    
    async def convert(self, pdf_file, style_reference_url: Optional[str], custom_prompt: str) -> Optional[ConversionResult]:
        start_time = datetime.now()
        content = self.extract_pdf_content(pdf_file)
        if not content: return None
        
        analysis = await self.analyze_content_with_ai(content, custom_prompt)
        if not analysis: return None
        
        html_content = self.generate_html(analysis)
        
        fingerprint_result = None
        if style_reference_url:
            with st.spinner(f"Analyse du style de {style_reference_url}..."):
                try:
                    async with WebStyleAnalyzer() as style_analyzer:
                        fingerprint = await style_analyzer.analyze_reference_page(style_reference_url)
                        fingerprint_result = fingerprint
                        if fingerprint.confidence_score > 0:
                            st.success(f"Empreinte de style capturée (Confiance: {fingerprint.confidence_score:.0%})")
                            html_content = style_analyzer.apply_style_transfer(html_content, fingerprint)
                        else:
                            st.warning("Impossible d'extraire une empreinte de style fiable de l'URL.")
                except Exception as e:
                    st.error(f"Erreur lors du transfert de style : {e}")

        return ConversionResult(
            html_content=html_content,
            seo_metrics={'title_length': len(analysis.get('title', '')), 'description_length': len(analysis.get('meta_description', ''))},
            processing_time=(datetime.now() - start_time).total_seconds(),
            word_count=content['word_count'],
            style_fingerprint=fingerprint_result
        )

# --- SECTION 3: INTERFACE UTILISATEUR STREAMLIT ---

def create_download_link(content: str, filename: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}" style="text-decoration:none;background-color:#007bff;color:white;padding:10px 15px;border-radius:5px;">📥 Télécharger HTML</a>'

def display_color_palette(palette: List[str]):
    if not palette: return
    st.subheader("Palette de couleurs extraite")
    swatches = "".join(f"<div style='text-align:center;margin:5px;'><div style='width:70px;height:70px;background-color:{c};border:1px solid #ddd;border-radius:8px;'></div><div style='font-size:12px;margin-top:4px;'>{c}</div></div>" for c in palette)
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;'>{swatches}</div>", unsafe_allow_html=True)

def main():
    st.markdown("""<style>.main-header{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:2rem;border-radius:10px;color:white;text-align:center;margin-bottom:2rem}.stTabs [data-baseweb="tab-list"]{gap:24px}</style>""", unsafe_allow_html=True)
    st.markdown('<div class="main-header"><h1>🎨 PDF to HTML Style Converter</h1><p>Conversion intelligente avec optimisation SEO et transfert de style via IA</p></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🔑 Clés API (Requis)")
        openai_key = st.text_input("Clé API OpenAI", type="password", help="Priorisé si les deux clés sont fournies.")
        anthropic_key = st.text_input("Clé API Anthropic (Claude)", type="password")
        st.markdown("---")
        st.header("🎨 Transfert de Style")
        enable_style_transfer = st.checkbox("Activer le transfert de style", value=False)
        style_reference_url = st.text_input("URL de référence", placeholder="https://votre-site.com", disabled=not enable_style_transfer)
        st.markdown("---")
        st.header("🤖 Prompt IA Personnalisé")
        custom_prompt = st.text_area(
            "Modifiez le prompt pour l'analyse de contenu",
            placeholder=DEFAULT_PROMPT,
            height=300,
            help="Votre prompt doit inclure la variable {text_sample} pour que le contenu du PDF y soit inséré."
        )
        final_prompt = custom_prompt if custom_prompt.strip() else DEFAULT_PROMPT


    st.subheader("📤 1. Choisissez un fichier PDF")
    uploaded_file = st.file_uploader("Sélectionnez un fichier PDF", type=['pdf'], label_visibility="collapsed")
    
    if uploaded_file:
        st.subheader("🚀 2. Résultat de la Conversion")
        converter = PDFConverterApp(openai_key, anthropic_key)
        with st.spinner("Analyse du PDF et conversion en cours..."):
            try:
                result = asyncio.run(converter.convert(uploaded_file, style_reference_url if enable_style_transfer else None, custom_prompt=final_prompt))
                if result:
                    st.success(f"Conversion réussie en {result.processing_time:.2f} secondes !")
                    tabs = st.tabs(["🌐 Aperçu", "💻 Code Source", "📊 Métriques"])
                    with tabs[0]: st.components.v1.html(result.html_content, height=600, scrolling=True)
                    with tabs[1]:
                        st.code(result.html_content, language='html')
                        st.markdown(create_download_link(result.html_content, "converted_document.html"), unsafe_allow_html=True)
                    with tabs[2]:
                        st.metric("Nombre de mots", f"{result.word_count}")
                        st.metric("Longueur Titre SEO", f"{result.seo_metrics['title_length']} car.")
                        st.metric("Longueur Méta-Desc.", f"{result.seo_metrics['description_length']} car.")
                        if result.style_fingerprint and result.style_fingerprint.color_palette:
                            st.markdown("---")
                            display_color_palette(result.style_fingerprint.color_palette)
            except Exception as e:
                st.error(f"Une erreur inattendue est survenue : {e}")
                logger.exception("Erreur détaillée dans main()")

if __name__ == "__main__":
    main()