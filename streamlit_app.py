#!/usr/bin/env python3
"""
PDF to HTML Converter - Version Intégrée avec Transfert de Style Neuronal
Application Streamlit complète pour convertir des PDF en HTML optimisé SEO/AEO,
avec la capacité d'appliquer un style visuel extrait d'une page web de référence.
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


# --- SECTION 1: BIBLIOTHÈQUE D'ANALYSE DE STYLE (Intégration du 2ème script) ---

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
        self.openai_client = openai.AsyncOpenAI(api_key=openai_key) if openai_key else None
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None
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
        """Analyse complète d'une page de référence."""
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
        """Récupère la page et ses ressources CSS."""
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                html_content = await response.text(encoding='utf-8', errors='replace')
        except aiohttp.ClientError as e:
            raise ValueError(f"Impossible de récupérer la page principale {url}: {e}")

        soup = BeautifulSoup(html_content, 'html.parser')
        css_links = [link.get('href') for link in soup.find_all('link', rel='stylesheet') if link.get('href')]
        
        tasks = []
        for href in css_links[:15]:  # Limiter à 15 fichiers CSS
            css_url = urljoin(url, href)
            tasks.append(self._fetch_resource(css_url))
        
        css_results = await asyncio.gather(*tasks)
        
        all_css = [res for res in css_results if res]
        all_css.extend(style.string for style in soup.find_all('style') if style.string)
        
        return html_content, all_css

    async def _fetch_resource(self, url: str) -> Optional[str]:
        """Récupère une ressource CSS."""
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                return await response.text(encoding='utf-8', errors='replace')
        except Exception as e:
            logger.warning(f"Impossible de récupérer la ressource {url}: {e}")
            return None

    def _extract_color_palette(self, css_contents: List[str], soup: BeautifulSoup) -> List[str]:
        """Extraction de la palette de couleurs."""
        colors = set()
        text_to_scan = "\n".join(css_contents)
        for element in soup.find_all(style=True):
            text_to_scan += ";" + element['style']

        colors.update(m.group(0).lower() for m in self._COLOR_HEX_PATTERN.finditer(text_to_scan))
        for m in self._COLOR_RGB_PATTERN.finditer(text_to_scan):
            try:
                colors.add(webcolors.rgb_to_hex((int(m.group(1)), int(m.group(2)), int(m.group(3)))))
            except ValueError:
                pass

        if not colors:
            return []

        hex_colors_for_clustering = [webcolors.hex_to_rgb(c) for c in colors if len(c) == 7]
        if len(hex_colors_for_clustering) < 3:
            return sorted(list(colors))[:10]

        pixels = np.array(hex_colors_for_clustering)
        n_clusters = min(8, len(pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(pixels)
        dominant_colors = [webcolors.rgb_to_hex(tuple(map(int, center))) for center in kmeans.cluster_centers_]
        return sorted(dominant_colors)

    def _analyze_typography(self, css_contents: List[str]) -> Dict[str, Any]:
        """Analyse la typographie."""
        fonts = set()
        for css in css_contents:
            for match in self._FONT_FAMILY_PATTERN.finditer(css):
                for family in match.group(1).split(','):
                    font = family.strip().strip("'\"").lower()
                    if font and font not in ['inherit', 'initial', 'sans-serif', 'serif', 'monospace']:
                        fonts.add(font.capitalize())
        return {'font_families': sorted(list(fonts))}

    def _calculate_confidence(self, palette: List, typo: Dict) -> float:
        """Calcule un score de confiance simple."""
        score = 0.0
        if len(palette) >= 3:
            score += 0.5
        if len(typo.get('font_families', [])) >= 1:
            score += 0.5
        return score

    def _is_light(self, hex_color: str) -> bool:
        """Détermine si une couleur est claire ou foncée pour le contraste."""
        try:
            h = hex_color.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
            return brightness > 140
        except:
            return True

    def generate_transfer_css(self, fingerprint: StyleFingerprint) -> str:
        """Génère une feuille de style CSS à partir de l'empreinte."""
        if fingerprint.confidence_score < 0.4:
            return "/* Confiance d'analyse de style trop faible. */"

        css = "/* --- Feuille de Style Transférée --- */\n"
        
        # Variables de couleur
        css += ":root {\n"
        if fingerprint.color_palette:
            primary_color = fingerprint.color_palette[0]
            text_color = '#212529' if self._is_light(primary_color) else '#f8f9fa'
            bg_color = '#ffffff' if self._is_light(primary_color) else '#212529'
            
            css += f"    --theme-primary: {primary_color};\n"
            css += f"    --theme-secondary: {fingerprint.color_palette[1] if len(fingerprint.color_palette) > 1 else '#6c757d'};\n"
            css += f"    --theme-bg: {bg_color};\n"
            css += f"    --theme-text: {text_color};\n"
            css += f"    --theme-link: {fingerprint.color_palette[2] if len(fingerprint.color_palette) > 2 else primary_color};\n"
        css += "}\n\n"

        # Styles de base
        css += "body.style-transferred {\n"
        if fingerprint.typography.get('font_families'):
            font_stack = ", ".join([f"'{f}'" for f in fingerprint.typography['font_families'][:2]])
            css += f"    font-family: {font_stack}, sans-serif;\n"
        css += "    background-color: var(--theme-bg);\n"
        css += "    color: var(--theme-text);\n"
        css += "}\n\n"

        # Titres
        css += ".style-transferred h1, .style-transferred h2, .style-transferred h3 {\n"
        css += "    color: var(--theme-primary);\n"
        css += "}\n\n"
        
        # Liens
        css += ".style-transferred a {\n"
        css += "    color: var(--theme-link);\n"
        css += "}\n\n"
        
        # Remplacement des styles du template de base
        css += ".style-transferred .container { background: var(--theme-bg); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }\n"
        css += ".style-transferred .header { border-bottom-color: var(--theme-primary); }\n"
        css += ".style-transferred .faq-section { border-left-color: var(--theme-secondary); }\n"
        
        return css

    def apply_style_transfer(self, target_html: str, fingerprint: StyleFingerprint) -> str:
        """Applique l'empreinte stylistique à un HTML cible."""
        logger.info("Application du transfert de style.")
        css_rules = self.generate_transfer_css(fingerprint)
        soup = BeautifulSoup(target_html, 'html.parser')

        # Supprimer les anciens styles pour éviter les conflits
        for style_tag in soup.find_all('style'):
            style_tag.decompose()

        # Ajouter les nouvelles règles CSS
        new_style_tag = soup.new_tag('style')
        new_style_tag.string = css_rules
        if soup.head:
            soup.head.append(new_style_tag)
        else:
            # Créer <head> s'il n'existe pas
            head = soup.new_tag('head')
            soup.insert(0, head)
            head.append(new_style_tag)

        # Appliquer une classe au body pour activer les nouveaux styles
        if soup.body:
            soup.body['class'] = soup.body.get('class', []) + ['style-transferred']

        return str(soup)


# --- SECTION 2: LOGIQUE DE CONVERSION PDF (Coeur du 1er script) ---

@dataclass
class ConversionResult:
    """Résultat de la conversion."""
    html_content: str
    seo_metrics: Dict[str, Any]
    processing_time: float
    word_count: int

class PDFConverterApp:
    """Classe principale de l'application Streamlit."""
    
    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.openai_client = openai.AsyncOpenAI(api_key=openai_key) if openai_key else None
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None
        self.api_keys_provided = bool(openai_key or anthropic_key)

    def extract_pdf_content(self, pdf_file) -> Optional[Dict[str, Any]]:
        """Extrait le contenu texte, les métadonnées et les tableaux d'un PDF."""
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            content = {
                'text': "".join(page.get_text() for page in pdf_document),
                'pages': pdf_document.page_count,
                'metadata': pdf_document.metadata,
                'tables': []
            }
            # ... (logique d'extraction de tableaux plus avancée si nécessaire) ...
            pdf_document.close()
            content['word_count'] = len(content['text'].split())
            return content
        except Exception as e:
            st.error(f"Erreur lors de l'extraction du contenu PDF : {e}")
            return None
    
    async def analyze_content_with_ai(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le contenu avec une IA pour générer une structure SEO."""
        if not self.api_keys_provided:
            st.info("Mode simulation (pas de clé API). L'analyse IA est simulée.")
            return self._simulate_ai_analysis(content)
        
        text_sample = content['text'][:4000]
        prompt = f"""
        Analyse le contenu de ce document PDF et génère une structure sémantique et SEO optimisée.
        Texte: "{text_sample}"
        
        Retourne un JSON valide avec les clés suivantes :
        - "title": Un titre SEO concis et percutant (max 60 caractères).
        - "meta_description": Une méta-description engageante (max 160 caractères).
        - "h1": Le titre principal de la page (H1).
        - "sections": Une liste d'objets, chacun avec "title" (pour un H2) et "content" (un paragraphe résumé de la section).
        - "keywords": Une liste de 5 à 7 mots-clés pertinents.
        - "faq": Une liste de 2 à 3 questions fréquentes (objets avec "question" et "answer").
        """
        
        try:
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4
                )
                return json.loads(response.choices[0].message.content)
            elif self.anthropic_client:
                # La logique pour Claude serait ici, potentiellement avec un formattage différent
                st.warning("L'analyse avec Claude n'est pas entièrement implémentée pour le JSON. Utilisation de la simulation.")
                return self._simulate_ai_analysis(content)
        except Exception as e:
            st.warning(f"Erreur de l'API IA, passage en mode simulation : {e}")
            return self._simulate_ai_analysis(content)

    def _simulate_ai_analysis(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Simulation d'analyse IA pour la démo sans API."""
        text = content['text']
        first_words = ' '.join(text.split()[:10])
        title = f"Document : {first_words[:40]}..."
        return {
            'title': title[:60],
            'meta_description': f"Document de {content['word_count']} mots, converti et structuré automatiquement.",
            'h1': title,
            'sections': [{'title': 'Résumé Principal', 'content': text[:800] + '...'}],
            'keywords': ['document', 'pdf', 'conversion', 'html', 'analyse'],
            'faq': [{'question': 'Quel est le sujet de ce document ?', 'answer': 'Ce document a été généré automatiquement.'}]
        }
    
    def generate_html(self, content: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Génère le HTML final à partir d'un template Jinja2."""
        html_template = Template("""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ analysis.title }}</title>
    <meta name="description" content="{{ analysis.meta_description }}">
    <meta name="keywords" content="{{ analysis.keywords | join(', ') }}">
    <script type="application/ld+json">
    {
        "@context": "https://schema.org", "@type": "Article", "headline": "{{ analysis.h1 }}",
        "description": "{{ analysis.meta_description }}", "author": {"@type": "Organization", "name": "PDF Converter"},
        "datePublished": "{{ current_date }}"
    }
    </script>
    <style>
        body { font-family: sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; background: #fdfdff; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .header { border-bottom: 2px solid #007bff; padding-bottom: 15px; margin-bottom: 25px; }
        h1, h2, h3 { color: #333; }
        .faq-section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 30px; border-left: 4px solid #17a2b8; }
        .faq-item { margin-bottom: 15px; } .faq-question { font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header"><h1>{{ analysis.h1 }}</h1></header>
        <main>
            {% for section in analysis.sections %}
            <section><h2>{{ section.title }}</h2><p>{{ section.content | replace('\\n', '<br>') }}</p></section>
            {% endfor %}
        </main>
        {% if analysis.faq %}
        <section class="faq-section">
            <h2>Questions fréquentes</h2>
            {% for item in analysis.faq %}
            <div class="faq-item">
                <div class="faq-question">{{ item.question }}</div><div>{{ item.answer }}</div>
            </div>
            {% endfor %}
        </section>
        {% endif %}
    </div>
</body>
</html>
        """)
        return html_template.render(
            content=content, analysis=analysis, current_date=datetime.now().strftime("%Y-%m-%d")
        )
    
    async def convert(self, pdf_file, style_reference_url: Optional[str] = None) -> Optional[ConversionResult]:
        """Processus de conversion complet, incluant le transfert de style optionnel."""
        start_time = datetime.now()
        
        content = self.extract_pdf_content(pdf_file)
        if not content: return None
        
        analysis = await self.analyze_content_with_ai(content)
        
        html_content = self.generate_html(content, analysis)
        
        # --- Étape de Transfert de Style ---
        if style_reference_url:
            with st.spinner(f"Analyse du style de {style_reference_url}..."):
                try:
                    # Utilise les mêmes clés API que pour l'analyse de contenu
                    openai_key = self.openai_client.api_key if self.openai_client else None
                    anthropic_key = self.anthropic_client.api_key if self.anthropic_client else None
                    
                    async with WebStyleAnalyzer(openai_key, anthropic_key) as style_analyzer:
                        fingerprint = await style_analyzer.analyze_reference_page(style_reference_url)
                        if fingerprint.confidence_score > 0:
                            st.success(f"Empreinte de style capturée (Confiance: {fingerprint.confidence_score:.0%})")
                            html_content = style_analyzer.apply_style_transfer(html_content, fingerprint)
                        else:
                            st.warning("Impossible d'extraire une empreinte de style fiable de l'URL.")
                except Exception as e:
                    st.error(f"Erreur lors du transfert de style : {e}")

        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ConversionResult(
            html_content=html_content,
            seo_metrics={
                'title_length': len(analysis.get('title', '')),
                'description_length': len(analysis.get('meta_description', '')),
            },
            processing_time=processing_time,
            word_count=content['word_count']
        )

# --- SECTION 3: INTERFACE UTILISATEUR STREAMLIT ---

def create_download_link(content: str, filename: str) -> str:
    """Crée un lien de téléchargement pour le contenu HTML."""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}" style="text-decoration: none; background-color: #007bff; color: white; padding: 10px 15px; border-radius: 5px;">📥 Télécharger le fichier HTML</a>'

def main():
    """Fonction principale qui exécute l'interface Streamlit."""
    # Styles CSS pour l'interface
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 10px; color: white;
            text-align: center; margin-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
    </style>
    """, unsafe_allow_html=True)

    # En-tête
    st.markdown('<div class="main-header"><h1>🎨 PDF to HTML Style Converter</h1><p>Conversion intelligente avec optimisation SEO et transfert de style via IA</p></div>', unsafe_allow_html=True)
    
    # Barre latérale pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🔑 Clés API (Optionnel)")
        openai_key = st.text_input("Clé API OpenAI", type="password", help="Requis pour une analyse de contenu de haute qualité.")
        anthropic_key = st.text_input("Clé API Anthropic", type="password", disabled=True)
        
        st.markdown("---")
        
        # --- NOUVELLE SECTION POUR LE TRANSFERT DE STYLE ---
        st.header("🎨 Transfert de Style")
        enable_style_transfer = st.checkbox("Activer le transfert de style", value=False)
        style_reference_url = ""
        if enable_style_transfer:
            style_reference_url = st.text_input(
                "URL de référence pour le style",
                placeholder="https://votre-site.com",
                help="L'application extraira les couleurs et polices de cette URL pour les appliquer au document."
            )

    # Zone principale pour l'upload
    st.subheader("📤 1. Choisissez un fichier PDF")
    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier PDF à convertir",
        type=['pdf'],
        label_visibility="collapsed"
    )
    
    # Traitement du fichier
    if uploaded_file:
        st.subheader("🚀 2. Résultat de la Conversion")
        
        converter = PDFConverterApp(openai_key, anthropic_key)
        
        with st.spinner("Analyse du PDF et conversion en cours..."):
            try:
                # Lancer la conversion asynchrone
                result = asyncio.run(converter.convert(uploaded_file, style_reference_url if enable_style_transfer else None))
                
                if result:
                    st.success(f"Conversion réussie en {result.processing_time:.2f} secondes !")
                    
                    # Affichage des résultats dans des onglets
                    tab1, tab2, tab3 = st.tabs(["🌐 Aperçu du Résultat", "💻 Code Source HTML", "📊 Métriques"])
                    
                    with tab1:
                        st.components.v1.html(result.html_content, height=600, scrolling=True)
                    
                    with tab2:
                        st.code(result.html_content, language='html')
                        st.markdown(create_download_link(result.html_content, "converted_document.html"), unsafe_allow_html=True)
                    
                    with tab3:
                        st.metric("Nombre de mots", f"{result.word_count}")
                        st.metric("Longueur du Titre SEO", f"{result.seo_metrics['title_length']} caractères")
                        st.metric("Longueur de la Méta-Description", f"{result.seo_metrics['description_length']} caractères")

            except Exception as e:
                st.error(f"Une erreur inattendue est survenue : {e}")

if __name__ == "__main__":
    main()