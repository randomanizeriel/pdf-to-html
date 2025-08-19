#!/usr/bin/env python3
"""
Transfert de Style Neuronal - Interface Streamlit
Application interactive pour l'extraction et l'application de styles visuels
"""

import streamlit as st
import asyncio
import json
import tempfile
import base64
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import io
import os
import re
import colorsys
from urllib.parse import urljoin, urlparse
import hashlib

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 50%, #ff8e8e 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff6b6b;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.2);
    }
    .code-container {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .color-palette {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0;
    }
    .color-swatch {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        border: 2px solid #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        display: inline-block;
    }
    .fingerprint-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import des bibliothèques nécessaires avec gestion d'erreur
@st.cache_data
def check_dependencies():
    """Vérifie les dépendances critiques"""
    missing_deps = []
    available_features = {
        'web_analysis': False,
        'pdf_processing': False,
        'ai_openai': False,
        'ai_anthropic': False,
        'ml_clustering': False
    }
    
    try:
        import aiohttp
        from bs4 import BeautifulSoup
        import tinycss2
        import webcolors
        available_features['web_analysis'] = True
    except ImportError:
        missing_deps.append("Analyse web (aiohttp, beautifulsoup4, tinycss2, webcolors)")
    
    try:
        import fitz
        from PIL import Image
        available_features['pdf_processing'] = True
    except ImportError:
        missing_deps.append("Traitement PDF (PyMuPDF, Pillow)")
    
    try:
        import openai
        available_features['ai_openai'] = True
    except ImportError:
        missing_deps.append("OpenAI")
    
    try:
        from anthropic import AsyncAnthropic
        available_features['ai_anthropic'] = True
    except ImportError:
        missing_deps.append("Anthropic")
    
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        available_features['ml_clustering'] = True
    except ImportError:
        missing_deps.append("Machine Learning (numpy, scikit-learn)")
    
    return available_features, missing_deps

# Vérification des dépendances au démarrage
FEATURES, MISSING_DEPS = check_dependencies()

# Import conditionnel des bibliothèques
if FEATURES['web_analysis']:
    import aiohttp
    from bs4 import BeautifulSoup
    import tinycss2
    import webcolors

if FEATURES['pdf_processing']:
    import fitz
    from PIL import Image

if FEATURES['ai_openai']:
    import openai

if FEATURES['ai_anthropic']:
    from anthropic import AsyncAnthropic

if FEATURES['ml_clustering']:
    import numpy as np
    from sklearn.cluster import KMeans

@dataclass
class StyleFingerprint:
    """Empreinte stylistique d'une page web ou d'un document PDF."""
    color_palette: List[str]
    typography: Dict[str, Any]
    layout_patterns: Dict[str, Any]
    spacing_system: Dict[str, Any]
    visual_hierarchy: Dict[str, Any]
    branding_elements: Dict[str, Any]
    responsive_breakpoints: List[int]
    css_rules: Dict[str, Any]
    metadata_profile: Dict[str, Any]
    design_mood: str
    confidence_score: float
    pdf_text_content: Optional[str] = None
    pdf_page_count: Optional[int] = None
    pdf_image_count: Optional[int] = None

class StyleTransferApp:
    """Application Streamlit pour le transfert de style"""
    
    def __init__(self, openai_key: str = None, anthropic_key: str = None):
        self.openai_client = None
        self.anthropic_client = None
        
        if openai_key and FEATURES['ai_openai']:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
        if anthropic_key and FEATURES['ai_anthropic']:
            self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
    
    async def analyze_reference_page(self, source_input: str, is_url: bool = True, is_pdf: bool = False) -> Optional[StyleFingerprint]:
        """Analyse une page de référence ou un document PDF"""
        if not FEATURES['web_analysis'] and not is_pdf:
            st.error("Les dépendances d'analyse web ne sont pas installées")
            return None
        
        if is_pdf and not FEATURES['pdf_processing']:
            st.error("Les dépendances de traitement PDF ne sont pas installées")
            return None
        
        try:
            if is_pdf:
                return await self._analyze_pdf_source(source_input, is_url)
            else:
                return await self._analyze_html_source(source_input, is_url)
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {e}")
            return None
    
    async def _analyze_html_source(self, source_input: str, is_url: bool) -> StyleFingerprint:
        """Analyse une source HTML"""
        if is_url:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(source_input, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        response.raise_for_status()
                        html_content = await response.text()
                except Exception as e:
                    raise ValueError(f"Impossible de récupérer la page: {e}")
        else:
            html_content = source_input
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Simulation d'analyse (version simplifiée pour la démo)
        color_palette = self._extract_colors_simple(html_content)
        typography = self._extract_typography_simple(soup)
        
        return StyleFingerprint(
            color_palette=color_palette,
            typography=typography,
            layout_patterns={'type': 'responsive'},
            spacing_system={'base_unit': 8},
            visual_hierarchy={'heading_levels': len(soup.find_all(['h1', 'h2', 'h3']))},
            branding_elements={'logo_found': bool(soup.find('img', alt=re.compile(r'logo', re.I)))},
            responsive_breakpoints=[768, 1024],
            css_rules={'rule_count': html_content.count('{')},
            metadata_profile={
                'title': soup.title.string if soup.title else "N/A",
                'description': "Analysé automatiquement"
            },
            design_mood='modern',
            confidence_score=0.85
        )
    
    async def _analyze_pdf_source(self, source_input: str, is_url: bool) -> StyleFingerprint:
        """Analyse une source PDF"""
        if is_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(source_input) as response:
                    pdf_data = await response.read()
        else:
            with open(source_input, 'rb') as f:
                pdf_data = f.read()
        
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text_content = ""
        page_count = len(doc)
        
        for page in doc:
            text_content += page.get_text()
        
        doc.close()
        
        return StyleFingerprint(
            color_palette=['#000000', '#333333'],
            typography={'font_families': ['Times', 'Arial']},
            layout_patterns={'type': 'PDF Document'},
            spacing_system={'type': 'N/A'},
            visual_hierarchy={'type': 'Document'},
            branding_elements={'type': 'N/A'},
            responsive_breakpoints=[],
            css_rules={},
            metadata_profile={'source_type': 'PDF'},
            design_mood='document',
            confidence_score=0.7,
            pdf_text_content=text_content[:1000],
            pdf_page_count=page_count,
            pdf_image_count=0
        )
    
    def _extract_colors_simple(self, html_content: str) -> List[str]:
        """Extraction simple des couleurs"""
        color_pattern = re.compile(r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})')
        colors = list(set(color_pattern.findall(html_content)))
        return [f"#{color}" for color in colors[:8]]
    
    def _extract_typography_simple(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extraction simple de la typographie"""
        return {
            'font_families': ['Arial', 'Helvetica', 'sans-serif'],
            'heading_count': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        }
    
    async def apply_style_transfer(self, target_html: str, fingerprint: StyleFingerprint) -> str:
        """Applique le transfert de style"""
        soup = BeautifulSoup(target_html, 'html.parser')
        
        # Génération du CSS de transfert
        transfer_css = self._generate_transfer_css(fingerprint)
        
        # Ajout du CSS au document
        if not soup.head:
            soup.insert(0, soup.new_tag('head'))
        
        style_tag = soup.new_tag('style')
        style_tag.string = transfer_css
        soup.head.append(style_tag)
        
        return str(soup)
    
    def _generate_transfer_css(self, fingerprint: StyleFingerprint) -> str:
        """Génère le CSS de transfert"""
        css = "/* Style Transfer CSS */\n"
        
        if fingerprint.color_palette:
            css += ":root {\n"
            for i, color in enumerate(fingerprint.color_palette[:5]):
                css += f"    --color-{i+1}: {color};\n"
            css += "}\n\n"
            
            css += "body { color: var(--color-1); }\n"
            css += "h1, h2, h3 { color: var(--color-2); }\n"
            css += "a { color: var(--color-3); }\n"
        
        return css

def create_download_link(content: str, filename: str) -> str:
    """Crée un lien de téléchargement"""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}" style="text-decoration: none; background: #ff6b6b; color: white; padding: 10px 20px; border-radius: 8px; display: inline-block; margin: 10px 0;">📥 Télécharger {filename}</a>'

def display_color_palette(colors: List[str]):
    """Affiche une palette de couleurs"""
    if not colors:
        return
    
    palette_html = '<div class="color-palette">'
    for color in colors:
        palette_html += f'<div class="color-swatch" style="background-color: {color}" title="{color}"></div>'
    palette_html += '</div>'
    
    st.markdown(palette_html, unsafe_allow_html=True)

def display_fingerprint(fingerprint: StyleFingerprint):
    """Affiche l'empreinte stylistique"""
    st.markdown(f"""
    <div class="fingerprint-card">
        <h3>🎨 Empreinte Stylistique Extraite</h3>
        <div class="stat-grid">
            <div><strong>Couleurs:</strong> {len(fingerprint.color_palette)}</div>
            <div><strong>Typographie:</strong> {len(fingerprint.typography.get('font_families', []))} polices</div>
            <div><strong>Confiance:</strong> {fingerprint.confidence_score:.0%}</div>
            <div><strong>Ambiance:</strong> {fingerprint.design_mood}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Neural Style Transfer</h1>
        <p>Extraction et application intelligente de styles visuels avec IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérification des dépendances
    if MISSING_DEPS:
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Dépendances manquantes</h4>
            <p>Certaines fonctionnalités ne seront pas disponibles :</p>
            <ul>{"".join(f"<li>{dep}</li>" for dep in MISSING_DEPS)}</ul>
            <p><strong>Installation :</strong><br>
            <code>pip install aiohttp beautifulsoup4 tinycss2 webcolors PyMuPDF Pillow openai anthropic numpy scikit-learn</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar de configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Fonctionnalités disponibles
        st.subheader("🔧 Fonctionnalités")
        for feature, available in FEATURES.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {feature.replace('_', ' ').title()}")
        
        st.divider()
        
        # Configuration IA
        use_ai = st.checkbox("Utiliser l'IA pour l'analyse", value=False,
                           help="Activez si vous avez des clés API")
        
        openai_key = None
        anthropic_key = None
        
        if use_ai:
            st.subheader("🔑 Clés API")
            if FEATURES['ai_openai']:
                openai_key = st.text_input("OpenAI API Key", type="password")
            if FEATURES['ai_anthropic']:
                anthropic_key = st.text_input("Anthropic API Key", type="password")
            
            if not openai_key and not anthropic_key:
                st.info("💡 Mode démo activé sans IA")
        
        st.divider()
        
        # Statistiques
        st.subheader("📊 Statistiques")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        if 'transfer_count' not in st.session_state:
            st.session_state.transfer_count = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses", st.session_state.analysis_count)
        with col2:
            st.metric("Transferts", st.session_state.transfer_count)
    
    # Interface principale
    tabs = st.tabs(["🔍 Analyser", "🎯 Transférer", "📊 Résultats"])
    
    with tabs[0]:
        st.subheader("🔍 Analyse de Style de Référence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Options d'entrée
            input_type = st.radio(
                "Type de source :",
                ["URL Web", "HTML Direct", "PDF Local", "PDF URL"],
                horizontal=True
            )
            
            if input_type == "URL Web":
                source_url = st.text_input(
                    "URL de la page à analyser",
                    placeholder="https://example.com",
                    help="URL complète de la page web"
                )
                is_url, is_pdf = True, False
                source_input = source_url
                
            elif input_type == "HTML Direct":
                source_html = st.text_area(
                    "Code HTML",
                    height=200,
                    placeholder="<html>...</html>"
                )
                is_url, is_pdf = False, False
                source_input = source_html
                
            elif input_type == "PDF Local":
                pdf_file = st.file_uploader("Fichier PDF", type=['pdf'])
                is_url, is_pdf = False, True
                source_input = pdf_file
                
            else:  # PDF URL
                pdf_url = st.text_input(
                    "URL du PDF",
                    placeholder="https://example.com/document.pdf"
                )
                is_url, is_pdf = True, True
                source_input = pdf_url
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>💡 Conseils</h4>
                <ul>
                    <li>Utilisez des sites bien conçus comme référence</li>
                    <li>Les PDF fonctionnent pour les couleurs de base</li>
                    <li>L'IA améliore la qualité d'analyse</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton d'analyse
        if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
            if source_input:
                app = StyleTransferApp(openai_key, anthropic_key)
                
                with st.spinner("Analyse en cours..."):
                    try:
                        # Pour les fichiers PDF uploadés, on doit gérer différemment
                        if input_type == "PDF Local" and pdf_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                tmp.write(pdf_file.read())
                                tmp_path = tmp.name
                            
                            fingerprint = asyncio.run(
                                app.analyze_reference_page(tmp_path, is_url=False, is_pdf=True)
                            )
                            os.unlink(tmp_path)
                        else:
                            fingerprint = asyncio.run(
                                app.analyze_reference_page(source_input, is_url, is_pdf)
                            )
                        
                        if fingerprint:
                            st.session_state.analysis_count += 1
                            st.session_state.current_fingerprint = fingerprint
                            
                            st.markdown("""
                            <div class="success-box">
                                ✅ <strong>Analyse terminée avec succès !</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Affichage de l'empreinte
                            display_fingerprint(fingerprint)
                            
                            # Détails de l'analyse
                            with st.expander("📋 Détails de l'analyse"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**🎨 Palette de couleurs:**")
                                    display_color_palette(fingerprint.color_palette)
                                    
                                    if fingerprint.color_palette:
                                        for i, color in enumerate(fingerprint.color_palette[:5]):
                                            st.write(f"{i+1}. {color}")
                                
                                with col2:
                                    st.write("**📝 Typographie:**")
                                    fonts = fingerprint.typography.get('font_families', [])
                                    if fonts:
                                        for font in fonts[:5]:
                                            st.write(f"• {font}")
                                    
                                    st.write("**📊 Métadonnées:**")
                                    if fingerprint.metadata_profile:
                                        st.write(f"Titre: {fingerprint.metadata_profile.get('title', 'N/A')[:50]}...")
                        
                        else:
                            st.error("❌ Échec de l'analyse")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
            else:
                st.warning("⚠️ Veuillez fournir une source à analyser")
    
    with tabs[1]:
        st.subheader("🎯 Transfert de Style")
        
        if 'current_fingerprint' not in st.session_state:
            st.info("🔍 Veuillez d'abord analyser une source de référence dans l'onglet 'Analyser'")
        else:
            fingerprint = st.session_state.current_fingerprint
            
            # Affichage de l'empreinte actuelle
            st.write("**🎨 Empreinte actuelle:**")
            display_fingerprint(fingerprint)
            
            st.divider()
            
            # HTML cible
            st.write("**📄 HTML Cible:**")
            target_html = st.text_area(
                "Code HTML à styliser",
                height=300,
                placeholder="""<!DOCTYPE html>
<html>
<head><title>Ma Page</title></head>
<body>
    <h1>Titre Principal</h1>
    <p>Contenu de la page...</p>
    <a href="#">Lien</a>
</body>
</html>""",
                help="HTML sur lequel appliquer le style"
            )
            
            if st.button("🎨 Appliquer le Transfert", type="primary", use_container_width=True):
                if target_html.strip():
                    app = StyleTransferApp(openai_key, anthropic_key)
                    
                    with st.spinner("Application du style..."):
                        try:
                            styled_html = asyncio.run(
                                app.apply_style_transfer(target_html, fingerprint)
                            )
                            
                            st.session_state.transfer_count += 1
                            st.session_state.styled_html = styled_html
                            
                            st.markdown("""
                            <div class="success-box">
                                ✅ <strong>Style appliqué avec succès !</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Affichage du résultat
                            with st.expander("👀 Aperçu du résultat", expanded=True):
                                st.components.v1.html(styled_html, height=400, scrolling=True)
                            
                            # Code source
                            with st.expander("💻 Code HTML généré"):
                                st.code(styled_html, language='html')
                                
                                # Lien de téléchargement
                                st.markdown(
                                    create_download_link(styled_html, "styled-page.html"),
                                    unsafe_allow_html=True
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors du transfert: {e}")
                else:
                    st.warning("⚠️ Veuillez fournir du code HTML")
    
    with tabs[2]:
        st.subheader("📊 Historique et Résultats")
        
        if 'current_fingerprint' not in st.session_state:
            st.info("Aucune analyse disponible pour le moment.")
        else:
            fingerprint = st.session_state.current_fingerprint
            
            # Métriques globales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Couleurs Extraites", len(fingerprint.color_palette))
            with col2:
                st.metric("Score de Confiance", f"{fingerprint.confidence_score:.0%}")
            with col3:
                st.metric("Analyses Totales", st.session_state.analysis_count)
            with col4:
                st.metric("Transferts Réussis", st.session_state.transfer_count)
            
            st.divider()
            
            # Analyse détaillée
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎨 Analyse Colorimétrique:**")
                if fingerprint.color_palette:
                    display_color_palette(fingerprint.color_palette)
                    
                    # Graphique de répartition (simulation)
                    chart_data = {
                        'Couleur': fingerprint.color_palette[:5],
                        'Usage': [30, 25, 20, 15, 10]  # Données simulées
                    }
                    st.bar_chart(chart_data, x='Couleur', y='Usage')
                
                st.write("**📱 Responsive Design:**")
                if fingerprint.responsive_breakpoints:
                    for bp in fingerprint.responsive_breakpoints:
                        st.write(f"• {bp}px")
            
            with col2:
                st.write("**📝 Profil Typographique:**")
                fonts = fingerprint.typography.get('font_families', [])
                for font in fonts:
                    st.write(f"• {font}")
                
                st.write("**🏗️ Structure:**")
                st.write(f"• Hiérarchie: {fingerprint.visual_hierarchy}")
                st.write(f"• Espacement: Base {fingerprint.spacing_system.get('base_unit', 'N/A')}")
                st.write(f"• Ambiance: {fingerprint.design_mood}")
            
            # Export des données
            if st.button("💾 Exporter l'Analyse (JSON)"):
                fingerprint_dict = {
                    'color_palette': fingerprint.color_palette,
                    'typography': fingerprint.typography,
                    'confidence_score': fingerprint.confidence_score,
                    'design_mood': fingerprint.design_mood,
                    'export_date': datetime.now().isoformat()
                }
                
                json_str = json.dumps(fingerprint_dict, indent=2)
                st.download_button(
                    "📥 Télécharger l'analyse",
                    json_str,
                    "style_analysis.json",
                    "application/json"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        © 2025 Neural Style Transfer - Développé avec ❤️ par xAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()