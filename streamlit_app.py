#!/usr/bin/env python3
""" Transfert de Style Neuronal HTML et PDF - Système Avancé
Extraction et application intelligente des styles visuels entre pages web et documents PDF.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
import base64
import hashlib
import colorsys
from urllib.parse import urljoin, urlparse
import math

# Bibliothèques d'analyse web
import aiohttp
from bs4 import BeautifulSoup
import tinycss2
import webcolors

# PDF processing (PyMuPDF for fitz)
import fitz # PyMuPDF
# CORRECTION 1: Import manquant pour le traitement d'images
from PIL import Image

# IA et analyse
import openai
from anthropic import AsyncAnthropic # Retiré si non utilisé
import numpy as np
from sklearn.cluster import KMeans

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StyleFingerprint:
    """Empreinte stylistique d'une page web ou d'un document PDF"""
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
    # Ajout de champs spécifiques pour l'analyse PDF
    pdf_text_content: Optional[str] = None
    pdf_page_count: Optional[int] = None
    pdf_image_count: Optional[int] = None

@dataclass
class VisualElement:
    """Éléments visuels analysés (simplifié pour cet exemple)"""
    selector: str
    element_type: str
    styles: Dict[str, str]
    computed_styles: Dict[str, str]
    visual_weight: float
    semantic_role: str
    position_context: Dict[str, Any]

class WebStyleAnalyzer:
    """Analyseur de styles web et de documents PDF avec IA"""

    # Constantes pour les patterns Regex de couleurs (pré-compilées)
    _COLOR_HEX_PATTERN = re.compile(r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})\b')
    _COLOR_RGB_PATTERN = re.compile(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')
    _COLOR_RGBA_PATTERN = re.compile(r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)')
    _COLOR_HSL_PATTERN = re.compile(r'hsl\(\s*(\d+)\s*,\s*(\d+%)\s*,\s*(\d+%)\s*\)')
    _COLOR_HSLA_PATTERN = re.compile(r'hsla\(\s*(\d+)\s*,\s*(\d+%)\s*,\s*(\d+%)\s*,\s*([0-9.]+)\s*\)')
    _COLOR_KEYWORDS_PATTERN = re.compile(r'\b(transparent|currentColor|inherit|initial|unset|revert)\b', re.IGNORECASE)
    # CORRECTION 2: L'attribut de webcolors est en minuscules (css3_names_to_hex)
    _COLOR_NAMES_PATTERN = re.compile(r'\b(' + '|'.join(webcolors.css3_names_to_hex.keys()) + r')\b', re.IGNORECASE)

    # Constantes pour les patterns Regex de typographie
    _FONT_FAMILY_PATTERN = re.compile(r'font-family\s*:\s*([^;]+)', re.IGNORECASE)
    _FONT_SIZE_PATTERN = re.compile(r'font-size\s*:\s*([^;]+)', re.IGNORECASE)
    _FONT_WEIGHT_PATTERN = re.compile(r'font-weight\s*:\s*([^;]+)', re.IGNORECASE)
    _LINE_HEIGHT_PATTERN = re.compile(r'line-height\s*:\s*([^;]+)', re.IGNORECASE)

    # Constantes pour les patterns Regex d'espacement
    _MARGIN_PATTERN = re.compile(r'margin[^:]*:\s*([^;]+)', re.IGNORECASE)
    _PADDING_PATTERN = re.compile(r'padding[^:]*:\s*([^;]+)', re.IGNORECASE)
    _NUMERIC_SPACING_PATTERN = re.compile(r'(\d+(?:\.\d+)?)(px|em|rem|%)?')
    _UTILITY_SPACING_CLASS_PATTERN = re.compile(r'^(m|p)[tbrlaexy]?-(\d+(?:\.\d+)?)$', re.IGNORECASE)

    # Constante pour les valeurs d'espacement par défaut
    _DEFAULT_SPACING_SCALE = [8.0, 16.0, 24.0, 32.0, 48.0]

    # Constantes pour les patterns d'exclusion de classes CSS (pré-compilées)
    _EXCLUDED_CLASS_PATTERNS = [
        re.compile(r'^col-', re.IGNORECASE), re.compile(r'^row', re.IGNORECASE), re.compile(r'^container', re.IGNORECASE), re.compile(r'^d-flex', re.IGNORECASE),
        re.compile(r'^justify-content-', re.IGNORECASE), re.compile(r'^align-items-', re.IGNORECASE),
        re.compile(r'^(m|p)[tbrlaexy]?-', re.IGNORECASE), re.compile(r'^(w|h)-', re.IGNORECASE), re.compile(r'^text-', re.IGNORECASE),
        re.compile(r'^bg-', re.IGNORECASE), re.compile(r'^font-', re.IGNORECASE), re.compile(r'^border-', re.IGNORECASE),
        re.compile(r'^shadow-', re.IGNORECASE), re.compile(r'clearfix', re.IGNORECASE), re.compile(r'float-', re.IGNORECASE),
        re.compile(r'sr-only', re.IGNORECASE), re.compile(r'^grid', re.IGNORECASE), re.compile(r'^flex', re.IGNORECASE),
        re.compile(r'active', re.IGNORECASE), re.compile(r'selected', re.IGNORECASE), re.compile(r'hidden', re.IGNORECASE),
        re.compile(r'visible', re.IGNORECASE), re.compile(r'item', re.IGNORECASE), re.compile(r'list', re.IGNORECASE),
        re.compile(r'wrapper', re.IGNORECASE), re.compile(r'content', re.IGNORECASE), re.compile(r'block', re.IGNORECASE)
    ]


    def __init__(self, ai_config: Dict[str, str]):
        self.openai_client = openai.AsyncOpenAI(api_key=ai_config.get('openai_api_key'))
        self.anthropic_client = AsyncAnthropic(api_key=ai_config.get('anthropic_api_key')) # Actuellement non utilisé directement dans ce code
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (compatible; StyleAnalyzer/1.0)'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def analyze_reference_page(self, source_input: str, is_url: bool = True, is_pdf: bool = False,
                                   extract_pdf_style: bool = False, extract_pdf_text: bool = False) -> StyleFingerprint:
        """
        Analyse complète d'une page de référence ou d'un document PDF.
        :param source_input: URL, contenu HTML ou chemin/URL du fichier PDF.
        :param is_url: Indique si source_input est une URL (HTML ou PDF).
        :param is_pdf: Indique si source_input (ou l'URL) pointe vers un fichier PDF.
        :param extract_pdf_style: Si is_pdf est True, extraire les styles du PDF.
        :param extract_pdf_text: Si is_pdf est True, extraire le texte du PDF.
        """
        logger.info(f"Début analyse source: {source_input} (URL: {is_url}, PDF: {is_pdf})")

        html_content = ""
        page_resources = {'css': [], 'js': [], 'images': []}
        pdf_text_content = None
        pdf_page_count = None
        pdf_image_count = None

        if is_pdf:
            # Gestion de l'entrée PDF (local ou URL)
            pdf_path_or_url = source_input
            pdf_data = None

            if is_url:
                logger.info(f"Téléchargement du PDF depuis l'URL: {pdf_path_or_url}")
                try:
                    async with self.session.get(pdf_path_or_url) as response:
                        response.raise_for_status()
                        pdf_data = await response.read()
                except aiohttp.ClientError as e:
                    logger.error(f"Erreur HTTP ou réseau lors du téléchargement du PDF {pdf_path_or_url}: {e}")
                    raise ValueError(f"Impossible de télécharger le PDF {pdf_path_or_url}.") from e
                except asyncio.TimeoutError as e:
                    logger.error(f"Délai d'attente dépassé lors du téléchargement du PDF {pdf_path_or_url}: {e}")
                    raise ValueError(f"Délai d'attente dépassé lors du téléchargement du PDF {pdf_path_or_url}.") from e
            else:
                logger.info(f"Chargement du PDF depuis le chemin: {pdf_path_or_url}")
                try:
                    with open(pdf_path_or_url, 'rb') as f:
                        pdf_data = f.read()
                except FileNotFoundError:
                    logger.error(f"Fichier PDF non trouvé: {pdf_path_or_url}")
                    raise
                except IOError as e:
                    logger.error(f"Erreur de lecture du fichier PDF {pdf_path_or_url}: {e}")
                    raise

            if pdf_data:
                pdf_analysis_results = await self._analyze_pdf_document(
                    pdf_data, extract_pdf_style, extract_pdf_text
                )
                # Mettre à jour les résultats de l'empreinte stylistique avec les données PDF
                color_palette = pdf_analysis_results['color_palette']
                typography = pdf_analysis_results['typography']
                pdf_text_content = pdf_analysis_results['text_content']
                pdf_page_count = pdf_analysis_results['page_count']
                pdf_image_count = pdf_analysis_results['image_count']

                # Les autres champs seront des placeholders ou N/A pour le PDF
                layout = {'pdf_document': True, 'details': 'N/A pour PDF direct'}
                spacing = {'pdf_document': True, 'details': 'N/A pour PDF direct'}
                hierarchy = {'pdf_document': True, 'details': 'N/A pour PDF direct'}
                branding = {'pdf_document': True, 'details': 'N/A pour PDF direct'}
                responsive = [] # N/A pour PDF
                css_rules = {'pdf_document': True, 'details': 'N/A pour PDF (pas de CSS au sens HTML)'}
                metadata = {'pdf_document': True, 'details': 'Métadonnées PDF via PyMuPDF'}

                # Calcul du score de confiance spécifique pour PDF
                confidence = self._calculate_pdf_analysis_confidence(pdf_analysis_results)

                return StyleFingerprint(
                    color_palette=color_palette,
                    typography=typography,
                    layout_patterns=layout,
                    spacing_system=spacing,
                    visual_hierarchy=hierarchy,
                    branding_elements=branding,
                    responsive_breakpoints=responsive,
                    css_rules=css_rules,
                    metadata_profile=metadata,
                    design_mood="document", # Mood par défaut pour un PDF
                    confidence_score=confidence,
                    pdf_text_content=pdf_text_content,
                    pdf_page_count=pdf_page_count,
                    pdf_image_count=pdf_image_count
                )

        else: # C'est une page HTML (URL ou contenu brut)
            if is_url:
                html_content, page_resources = await self._fetch_page_with_resources(source_input)
            else:
                html_content = source_input
                page_resources = {'css': [], 'js': [], 'images': []}
            
            soup = BeautifulSoup(html_content, 'html.parser')

            all_css_content_list = self._get_all_css_content(soup, page_resources)

            analysis_tasks = await asyncio.gather(
                self._extract_color_palette(all_css_content_list, soup), # Passe aussi la soup pour les styles inline
                self._analyze_typography(all_css_content_list, soup),
                self._extract_layout_patterns(soup),
                self._analyze_spacing_system(all_css_content_list, soup),
                self._detect_visual_hierarchy(soup),
                self._extract_branding_elements(soup),
                self._analyze_responsive_design(all_css_content_list, soup),
                self._extract_metadata_profile(soup)
            )
            color_palette, typography, layout, spacing, hierarchy, branding, responsive, metadata = analysis_tasks

            css_rules = self._extract_css_rules(all_css_content_list)

            design_mood = await self._analyze_design_mood_with_ai(soup, {
                'colors': color_palette,
                'typography': typography,
                'layout': layout
            })

            confidence = self._calculate_analysis_confidence({
                'color_palette': color_palette,
                'typography': typography,
                'layout_patterns': layout,
                'spacing_system': spacing,
                'visual_hierarchy': hierarchy,
                'branding_elements': branding,
                'responsive_breakpoints': responsive,
                'css_rules': css_rules,
                'metadata_profile': metadata
            })

            return StyleFingerprint(
                color_palette=color_palette,
                typography=typography,
                layout_patterns=layout,
                spacing_system=spacing,
                visual_hierarchy=hierarchy,
                branding_elements=branding,
                responsive_breakpoints=responsive,
                css_rules=css_rules,
                metadata_profile=metadata,
                design_mood=design_mood,
                confidence_score=confidence,
                pdf_text_content=None, # Non applicable pour HTML
                pdf_page_count=None,
                pdf_image_count=None
            )

    def _get_all_css_content(self, soup: BeautifulSoup, resources: Dict[str, Any]) -> List[str]:
        """Collecte tout le contenu CSS (inline <style> et externe) dans une liste de chaînes."""
        all_css_list = []
        for css_resource in resources.get('css', []):
            all_css_list.append(css_resource['content'])
        for style in soup.find_all('style'):
            if style.string:
                all_css_list.append(style.string)
        return all_css_list

    async def _fetch_page_with_resources(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Récupère la page et ses ressources CSS/JS"""
        resources = {'css': [], 'js': [], 'images': []}
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"URL de base invalide ou mal formée: {url}")

        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                html_content = await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Erreur HTTP ou réseau lors de la récupération de la page {url}: {e}")
            raise ValueError(f"Impossible de récupérer la page {url} en raison d'une erreur réseau ou HTTP.") from e
        except asyncio.TimeoutError as e:
            logger.error(f"Délai d'attente dépassé lors de la récupération de la page {url}: {e}")
            raise ValueError(f"Délai d'attente dépassé lors de la récupération de la page {url}.") from e

        soup = BeautifulSoup(html_content, 'html.parser')

        css_links = soup.find_all('link', {'rel': 'stylesheet'})
        for link in css_links:
            href = link.get('href')
            if not href: continue
            if href.startswith('#'):
                logger.debug(f"Ignored fragment-only CSS link: {href}")
                continue

            css_url = urljoin(url, href)
            parsed_css_url = urlparse(css_url)

            if parsed_css_url.scheme not in ['http', 'https'] or not parsed_css_url.netloc:
                logger.warning(f"URL CSS externe non valide ou non HTTP/HTTPS après jointure: {href} -> {css_url}. Ignoré.")
                continue

            try:
                async with self.session.get(css_url) as css_response:
                    css_response.raise_for_status()
                    css_content = await css_response.text()
                    resources['css'].append({'url': css_url, 'content': css_content})
            except aiohttp.ClientError as e: logger.warning(f"Erreur HTTP/réseau lors de la récupération de la ressource CSS {css_url}: {e}")
            except asyncio.TimeoutError: logger.warning(f"Délai d'attente dépassé lors de la récupération de la ressource CSS {css_url}. Ignoré.")
            except Exception as e: logger.warning(f"Erreur inattendue lors de la récupération CSS {css_url}: {e}")

        return html_content, resources

    # Helper pour convertir HSL/HSLA en RGB puis Hex
    def _hsl_to_hex(self, h: int, s: str, l: str, a: Optional[float] = 1.0) -> str:
        s_val = float(s.strip('%')) / 100
        l_val = float(l.strip('%')) / 100
        
        h_norm = h % 360
        s_norm = max(0.0, min(1.0, s_val))
        l_norm = max(0.0, min(1.0, l_val))

        r, g, b = colorsys.hls_to_rgb(h_norm / 360, l_norm, s_norm)
        
        if a < 1.0:
            r = r * a + 1 * (1 - a)
            g = g * a + 1 * (1 - a)
            b = b * a + 1 * (1 - a)

        r_int = max(0, min(255, int(r * 255)))
        g_int = max(0, min(255, int(g * 255)))
        b_int = max(0, min(255, int(b * 255)))
        return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


    async def _extract_color_palette(self, all_css_content_list: List[str], soup: BeautifulSoup = None, pdf_image_data_list: List[Dict] = None) -> List[str]:
        """
        Extraction intelligente de la palette de couleurs.
        Peut extraire des couleurs à partir de contenu CSS/HTML ou d'images PDF rendues.
        """
        colors = set()
        
        # CORRECTION 4 & 5: Logique de traitement des images PDF refactorisée
        if pdf_image_data_list:
            logger.info("Extraction de la palette de couleurs à partir d'images PDF.")
            for img_data in pdf_image_data_list:
                try:
                    # Créer une image PIL à partir des données brutes du pixmap
                    image = Image.frombytes(
                        img_data["colorspace"],
                        (img_data["width"], img_data["height"]),
                        img_data["samples"]
                    )
                    
                    # Redimensionner l'image pour l'analyse de couleurs afin d'accélérer le traitement
                    image.thumbnail((200, 200), Image.LANCZOS)
                    # Convertir l'image en un tableau numpy pour KMeans
                    # S'assurer que l'image est en RGB (pas RGBA) pour KMeans
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    np_image = np.array(image)

                    # Remodeler l'image pour le clustering
                    pixels = np_image.reshape(-1, 3)
                    n_colors = min(5, len(pixels)) # Essayer d'extraire 5 couleurs dominantes
                    if n_colors > 0:
                        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                        kmeans.fit(pixels)
                        for center in kmeans.cluster_centers_:
                            r, g, b = map(int, center)
                            colors.add(f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'extraction des couleurs d'une image PDF: {e}")
            
        else: # Traitement HTML/CSS
            logger.info("Extraction de la palette de couleurs à partir de CSS/HTML.")
            combined_css_and_inline_styles = "\n".join(all_css_content_list)
            
            # Ajouter les styles des attributs 'style' des éléments HTML à la chaîne combinée
            if soup: # S'assurer que soup est fourni pour l'analyse HTML
                for element in soup.find_all(attrs={'style': True}):
                    combined_css_and_inline_styles += f"\n{element.get('style', '')}"

            # Extraction de toutes les couleurs de la chaîne combinée
            for pattern in [self._COLOR_HEX_PATTERN, self._COLOR_RGB_PATTERN, 
                            self._COLOR_RGBA_PATTERN, self._COLOR_HSL_PATTERN, 
                            self._COLOR_HSLA_PATTERN, self._COLOR_NAMES_PATTERN]:
                for match in pattern.finditer(combined_css_and_inline_styles):
                    if pattern == self._COLOR_HEX_PATTERN:
                        colors.add(match.group(0).lower())
                    elif pattern == self._COLOR_RGB_PATTERN:
                        r, g, b = map(int, match.groups())
                        colors.add(f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}")
                    elif pattern == self._COLOR_RGBA_PATTERN:
                        r, g, b = map(int, match.groups()[:3])
                        a = float(match.group(4))
                        if a < 1.0:
                            r_blended = int(r * a + 255 * (1 - a))
                            g_blended = int(g * a + 255 * (1 - a))
                            b_blended = int(b * a + 255 * (1 - a))
                            colors.add(f"#{max(0, min(255, r_blended)):02x}{max(0, min(255, g_blended)):02x}{max(0, min(255, b_blended)):02x}")
                        else:
                            colors.add(f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}")
                    elif pattern == self._COLOR_HSL_PATTERN:
                        h, s, l = match.groups()
                        try: colors.add(self._hsl_to_hex(int(h), s, l, 1.0))
                        except ValueError: logger.warning(f"Impossible de parser HSL: {match.group(0)}")
                    elif pattern == self._COLOR_HSLA_PATTERN:
                        h, s, l, a = match.groups()
                        try: colors.add(self._hsl_to_hex(int(h), s, l, float(a)))
                        except ValueError: logger.warning(f"Impossible de parser HSLA: {match.group(0)}")
                    elif pattern == self._COLOR_NAMES_PATTERN:
                        color_name_lower = match.group(0).lower()
                        # CORRECTION 2 (bis): L'attribut est en minuscules
                        if color_name_lower in webcolors.css3_names_to_hex:
                            colors.add(webcolors.css3_names_to_hex[color_name_lower])

        color_list = list(colors)
        hex_colors_for_clustering = [c for c in color_list if self._COLOR_HEX_PATTERN.match(c)]

        if len(hex_colors_for_clustering) > 2:
            optimized_palette = await self._optimize_color_palette(hex_colors_for_clustering)
            return optimized_palette[:12]
        return hex_colors_for_clustering[:12] if hex_colors_for_clustering else color_list[:12]


    async def _optimize_color_palette(self, colors: List[str]) -> List[str]:
        """Optimise la palette de couleurs avec clustering"""
        try:
            rgb_colors = []
            for color_str in colors:
                if self._COLOR_HEX_PATTERN.match(color_str):
                    hex_color = color_str[1:]
                    if len(hex_color) == 3: hex_color = ''.join([c*2 for c in hex_color])
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    rgb_colors.append([r, g, b])
            
            if len(rgb_colors) < 3: return colors[:8]

            rgb_array = np.array(rgb_colors)
            n_clusters = min(8, len(rgb_colors))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(rgb_array)

            optimized_colors = []
            for center in kmeans.cluster_centers_:
                r, g, b = map(int, center)
                hex_color = f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"
                optimized_colors.append(hex_color)
            return optimized_colors
        except Exception as e:
            logger.warning(f"Erreur optimisation palette: {e}")
            return colors[:8]

    async def _analyze_typography(self, all_css_content_list: List[str], soup: BeautifulSoup, pdf_document: Optional[fitz.Document] = None) -> Dict[str, Any]:
        """Analyse complète de la typographie"""
        typography = {
            'font_families': {},
            'font_sizes': [],
            'font_weights': [],
            'line_heights': [],
            'text_hierarchy': {},
            'font_loading': []
        }

        if pdf_document:
            logger.info("Analyse de la typographie à partir du PDF.")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                # Extrait le texte par blocs, ce qui inclut des infos de police
                text_blocks = page.get_text("dict")["blocks"]
                for block in text_blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                font_name = span["font"]
                                font_size = span["size"]
                                
                                typography['font_families'][font_name] = typography['font_families'].get(font_name, 0) + 1
                                typography['font_sizes'].append(f"{font_size}pt")
                                
            typography['font_sizes'] = list(dict.fromkeys(typography['font_sizes']))
            # Pour le PDF, font_weights, line_heights, text_hierarchy, font_loading sont moins pertinents ou trop complexes
            typography['font_weights'] = ["N/A pour PDF"]
            typography['line_heights'] = ["N/A pour PDF"]
            typography['text_hierarchy'] = {"details": "Analyse de hiérarchie limitée pour PDF"}
            typography['font_loading'] = ["N/A pour PDF"]
            
        else: # Analyse HTML/CSS
            logger.info("Analyse de la typographie à partir de CSS/HTML.")
            combined_css = "\n".join(all_css_content_list)
            parsed_rules = tinycss2.parse_stylesheet(combined_css)

            for rule in parsed_rules:
                if isinstance(rule, tinycss2.ast.QualifiedRule):
                    for declaration in rule.content:
                        if isinstance(declaration, tinycss2.ast.Declaration):
                            if declaration.name == 'font-family':
                                font_families_str = ''.join(t.value for t in declaration.value).strip()
                                families = [f.strip('\'" ') for f in font_families_str.split(',') if f.strip('\'" ')]
                                for fam in families: typography['font_families'][fam] = typography['font_families'].get(fam, 0) + 1
                            elif declaration.name == 'font-size': typography['font_sizes'].append(''.join(t.value for t in declaration.value).strip())
                            elif declaration.name == 'font-weight': typography['font_weights'].append(''.join(t.value for t in declaration.value).strip())
                            elif declaration.name == 'line-height': typography['line_heights'].append(''.join(t.value for t in declaration.value).strip())
                elif isinstance(rule, tinycss2.ast.AtRule) and rule.at_keyword == 'import':
                    import_url_match = re.search(r'url\((["\']?)(.*?)\1\)', ''.join(t.value for t in rule.prelude))
                    if import_url_match:
                        if 'fonts.googleapis.com' in import_url_match.group(2) or 'typekit.net' in import_url_match.group(2):
                            typography['font_loading'].append(import_url_match.group(2))

            elements_with_style = soup.find_all(attrs={'style': True})
            for element in elements_with_style:
                style_attr = element.get('style', '')
                for match in self._FONT_FAMILY_PATTERN.finditer(style_attr):
                    families = [f.strip('\'" ') for f in match.group(1).split(',') if f.strip('\'" ')]
                    for fam in families: typography['font_families'][fam] = typography['font_families'].get(fam, 0) + 1
                for match in self._FONT_SIZE_PATTERN.finditer(style_attr): typography['font_sizes'].append(match.group(1).strip())
                for match in self._FONT_WEIGHT_PATTERN.finditer(style_attr): typography['font_weights'].append(match.group(1).strip())
                for match in self._LINE_HEIGHT_PATTERN.finditer(style_attr): typography['line_heights'].append(match.group(1).strip())

            heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            for tag in heading_tags:
                headings = soup.find_all(tag)
                if headings:
                    typography['text_hierarchy'][tag] = {
                        'count': len(headings),
                        'avg_length': sum(len(h.get_text()) for h in headings) / len(headings) if len(headings) > 0 else 0,
                        'styles': []
                    }
                    for h in headings:
                        style_str = h.get('style', '')
                        current_styles = {}
                        for prop in ['font-size', 'font-weight', 'color']:
                            match = re.search(f'{prop}:\s*([^;]+)', style_str, re.IGNORECASE)
                            if match: current_styles[prop] = match.group(1).strip()
                        typography['text_hierarchy'][tag]['styles'].append(current_styles)

            font_links = soup.find_all('link', {'href': lambda x: x and ('fonts.googleapis.com' in x or 'typekit.net' in x)})
            typography['font_loading'].extend([link.get('href') for link in font_links])
            typography['font_loading'] = list(set(typography['font_loading']))

            typography['font_sizes'] = list(dict.fromkeys(typography['font_sizes']))
            typography['font_weights'] = list(dict.fromkeys(typography['font_weights']))
            typography['line_heights'] = list(dict.fromkeys(typography['line_heights']))

        return typography

    async def _extract_layout_patterns(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Détection des patterns de mise en page (principalement pour HTML)"""
        layout = {
            'grid_systems': [], 'flexbox_usage': [], 'container_patterns': [],
            'sidebar_detection': None, 'header_footer_structure': {}, 'content_width_patterns': []
        }
        grid_indicators = [
            'grid', 'col-', 'row', 'container', 'wrapper', 'columns', 'g-col',
            'flex', 'd-flex', 'justify-content-', 'align-items-',
        ]
        for indicator in grid_indicators:
            elements = soup.find_all(class_=lambda x: x and indicator in ' '.join(x).lower())
            if elements:
                if 'flex' in indicator or 'd-flex' in indicator or 'justify-content-' in indicator or 'align-items-' in indicator:
                    layout['flexbox_usage'].append({'pattern': indicator, 'count': len(elements), 'examples': [elem.get('class') for elem in elements[:3]]})
                elif 'grid' in indicator or 'col-' in indicator or 'row' in indicator or 'columns' in indicator or 'g-col' in indicator:
                    layout['grid_systems'].append({'pattern': indicator, 'count': len(elements), 'examples': [elem.get('class') for elem in elements[:3]]})
                elif 'container' in indicator or 'wrapper' in indicator:
                    layout['container_patterns'].append({'pattern': indicator, 'count': len(elements), 'examples': [elem.get('class') for elem in elements[:3]]})

        header = soup.find(['header', 'div', 'nav'], class_=lambda x: x and ('header' in ' '.join(x).lower() or 'navbar' in ' '.join(x).lower()))
        footer = soup.find(['footer', 'div'], class_=lambda x: x and 'footer' in ' '.join(x).lower())
        layout['header_footer_structure'] = {
            'has_header': header is not None, 'has_footer': footer is not None,
            'header_classes': header.get('class', []) if header else [], 'footer_classes': footer.get('class', []) if footer else []
        }
        sidebar_indicators = ['sidebar', 'aside', 'nav-side', 'menu-side', 'col-md-3', 'col-lg-3']
        for indicator in sidebar_indicators:
            sidebar = soup.find(['aside', 'div', 'nav'], class_=lambda x: x and indicator in ' '.join(x).lower())
            if sidebar:
                layout['sidebar_detection'] = {'type': indicator, 'classes': sidebar.get('class', []), 'position': 'detected'}
                break
        content_width_indicators = [
            '.container', '.wrapper', '.main-content', '.content',
            '[style*="max-width"]', '[style*="width:"]'
        ]
        for selector_str in content_width_indicators:
            elements = soup.select(selector_str)
            for elem in elements:
                style_attr = elem.get('style', '')
                max_width_match = re.search(r'max-width:\s*([^;]+)', style_attr)
                width_match = re.search(r'width:\s*([^;]+)', style_attr)
                if max_width_match: layout['content_width_patterns'].append({'selector': selector_str, 'value': max_width_match.group(1).strip(), 'type': 'max-width'})
                if width_match: layout['content_width_patterns'].append({'selector': selector_str, 'value': width_match.group(1).strip(), 'type': 'width'})
        return layout

    async def _analyze_spacing_system(self, all_css_content_list: List[str], soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyse du système d'espacement (principalement pour HTML)"""
        spacing = {
            'margin_patterns': {}, 'padding_patterns': {}, 'spacing_scale': [], 'consistent_ratios': {}
        }
        combined_css = "\n".join(all_css_content_list)
        margin_values = []
        padding_values = []

        parsed_rules = tinycss2.parse_stylesheet(combined_css)
        for rule in parsed_rules:
            if isinstance(rule, tinycss2.ast.QualifiedRule):
                for declaration in rule.content:
                    if isinstance(declaration, tinycss2.ast.Declaration):
                        if declaration.name.startswith('margin'): margin_values.append(''.join(t.value for t in declaration.value).strip())
                        elif declaration.name.startswith('padding'): padding_values.append(''.join(t.value for t in declaration.value).strip())

        for element in soup.find_all(class_=True):
            classes = element.get('class', [])
            for cls in classes:
                num_val_match = self._UTILITY_SPACING_CLASS_PATTERN.match(cls)
                if num_val_match:
                    prop_type = num_val_match.group(1)
                    value = num_val_match.group(2)
                    if prop_type == 'm': margin_values.append(f"{value}px") # Assigner une unité par défaut
                    elif prop_type == 'p': padding_values.append(f"{value}px") # Assigner une unité par défaut

        elements_with_style = soup.find_all(attrs={'style': True})
        for element in elements_with_style:
            style_attr = element.get('style', '')
            for match in self._MARGIN_PATTERN.finditer(style_attr): margin_values.extend([v.strip() for v in match.group(1).split() if v.strip()])
            for match in self._PADDING_PATTERN.finditer(style_attr): padding_values.extend([v.strip() for v in match.group(1).split() if v.strip()])

        spacing['margin_patterns'] = self._analyze_spacing_values(margin_values)
        spacing['padding_patterns'] = self._analyze_spacing_values(padding_values)
        all_numeric_spacing = self._extract_numeric_spacing(margin_values + padding_values)
        spacing['spacing_scale'] = self._detect_spacing_scale(all_numeric_spacing)
        return spacing

    def _extract_numeric_spacing(self, values: List[str]) -> List[float]:
        """Extrait les valeurs numériques d'espacement (px, em, rem, %) en float."""
        numeric_values = []
        for value_str in values:
            parts = value_str.split()
            for part in parts:
                match = self._NUMERIC_SPACING_PATTERN.match(part.strip())
                if match:
                    try: numeric_values.append(float(match.group(1)))
                    except ValueError: pass
        return numeric_values

    def _analyze_spacing_values(self, values: List[str]) -> Dict[str, Any]:
        """Analyse les valeurs d'espacement pour détecter des patterns"""
        parsed_values = []
        for value in values:
            match = self._NUMERIC_SPACING_PATTERN.match(value.strip())
            if match:
                try: parsed_values.append({'value': float(match.group(1)), 'unit': match.group(2) or 'px', 'original': value})
                except ValueError: pass

        if not parsed_values: return {'common_values': [], 'units': [], 'average': 0, 'range': (0, 0)}

        units_data = {}
        for pv in parsed_values:
            unit = pv['unit']
            if unit not in units_data: units_data[unit] = []
            units_data[unit].append(pv['value'])

        result = {'common_values': [], 'units': list(units_data.keys()), 'average_per_unit': {}, 'range_per_unit': {}}
        all_values_flat = []
        for unit, vals in units_data.items():
            if vals:
                result['average_per_unit'][unit] = sum(vals) / len(vals)
                result['range_per_unit'][unit] = (min(vals), max(vals))
                all_values_flat.extend(vals)
        if all_values_flat: result['common_values'] = self._find_common_values(all_values_flat)
        return result

    def _find_common_values(self, values: List[float]) -> List[float]:
        """Trouve les valeurs les plus communes, arrondies à 1 décimale"""
        from collections import Counter
        rounded_values = [round(v, 1) for v in values]
        counter = Counter(rounded_values)
        return sorted([value for value, count in counter.most_common(5)])

    def _detect_spacing_scale(self, values: List[float]) -> List[float]:
        """Détecte une échelle d'espacement cohérente"""
        if not values: return []
        unique_values = sorted(list(set(values)))
        common_bases = [2, 4, 8, 10, 12, 16]
        detected_scales = {}

        for base in common_bases:
            scale_elements = [v for v in unique_values if v > 0 and (v % base < 0.01 or abs(v % base - base) < 0.01)]
            if len(scale_elements) >= 3: detected_scales[base] = scale_elements
        if detected_scales:
            best_base = max(detected_scales, key=lambda base: (len(detected_scales[base]), -base))
            return sorted(list(set(detected_scales[best_base])))[:10]
        from collections import Counter
        counter = Counter(values)
        most_common = [val for val, count in counter.most_common(5)]
        return sorted(list(set(most_common)))[:10]

    async def _detect_visual_hierarchy(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyse de la hiérarchie visuelle (principalement pour HTML)"""
        hierarchy = {
            'heading_structure': {}, 'emphasis_patterns': {},
            'visual_weight_distribution': {}, 'color_hierarchy': []
        }
        heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        for tag in heading_tags:
            headings = soup.find_all(tag)
            if headings:
                hierarchy['heading_structure'][tag] = {
                    'count': len(headings), 'has_consistent_style': self._check_consistent_heading_style(headings),
                    'nesting_level': int(tag[1])
                }
        emphasis_elements = soup.find_all(['strong', 'b', 'em', 'i', 'mark', 'small', 'big', 'cite', 'blockquote'])
        for elem in emphasis_elements:
            tag = elem.name
            hierarchy['emphasis_patterns'][tag] = hierarchy['emphasis_patterns'].get(tag, 0) + 1
        large_elements = soup.find_all(lambda tag: tag.name in ['h1', 'img', 'video'] or ('class' in tag.attrs and 'button' in tag['class']))
        if large_elements: hierarchy['visual_weight_distribution']['large_elements_count'] = len(large_elements)
        return hierarchy

    def _check_consistent_heading_style(self, headings: List) -> bool:
        """Vérifie la cohérence de style des headings"""
        if len(headings) < 2: return True
        first_heading = headings[0]
        first_classes = set(first_heading.get('class', []))
        first_inline_style = first_heading.get('style', '')

        for heading in headings[1:]:
            current_classes = set(heading.get('class', []))
            current_inline_style = heading.get('style', '')
            if first_classes and current_classes:
                if not first_classes.intersection(current_classes): return False
            elif (first_classes and not current_classes) or (not first_classes and current_classes): return False
            if first_inline_style and current_inline_style:
                if first_inline_style != current_inline_style: return False
            elif (first_inline_style and not current_inline_style) or (not first_inline_style and current_inline_style): return False
        return True

    async def _extract_branding_elements(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extraction des éléments de branding (principalement pour HTML)"""
        branding = {
            'logo_detection': {}, 'brand_colors': [], 'consistent_elements': [], 'style_signatures': []
        }
        logo_selectors = [
            'img[alt*="logo" i]', 'img[src*="logo" i]', '.logo', '#logo',
            '[class*="brand" i]', '[id*="brand" i]',
            'a[aria-label*="logo" i]', 'a[title*="logo" i]'
        ]
        for selector in logo_selectors:
            try:
                elements = soup.select(selector)
                if elements: branding['logo_detection'][selector] = len(elements)
            except Exception as e:
                logger.warning(f"Erreur de sélecteur lors de la détection de branding ({selector}): {e}")
                continue
        repeated_classes = self._find_repeated_classes(soup)
        branding['consistent_elements'] = repeated_classes[:10]
        return branding

    def _find_repeated_classes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Trouve les classes CSS répétées (indicateurs de cohérence)"""
        class_counts = {}
        all_elements = soup.find_all(class_=True)
        for element in all_elements:
            classes = element.get('class', [])
            for class_name in classes: class_counts[class_name] = class_counts.get(class_name, 0) + 1

        repeated = []
        for class_name, count in class_counts.items():
            is_excluded = False
            for pattern in self._EXCLUDED_CLASS_PATTERNS:
                if pattern.match(class_name):
                    is_excluded = True
                    break
            if not is_excluded and count > 2: repeated.append({'class': class_name, 'count': count})
        return sorted(repeated, key=lambda x: x['count'], reverse=True)

    async def _analyze_responsive_design(self, all_css_content_list: List[str], soup: BeautifulSoup) -> List[int]:
        """Détection des breakpoints responsive (principalement pour HTML)"""
        breakpoints = set()
        combined_css = "\n".join(all_css_content_list)
        parsed_rules = tinycss2.parse_stylesheet(combined_css)
        for rule in parsed_rules:
            if isinstance(rule, tinycss2.ast.AtRule) and rule.at_keyword == 'media':
                media_text = ''.join(t.value for t in rule.prelude)
                matches = re.findall(r'(?:max-width|min-width):\s*(\d+)(?:px|em|rem)', media_text, re.IGNORECASE)
                for bp_val in matches:
                    try: breakpoints.add(int(bp_val))
                    except ValueError: pass
        viewport_meta = soup.find('meta', {'name': 'viewport'})
        if viewport_meta and 'width=device-width' in viewport_meta.get('content', '') and 'initial-scale' in viewport_meta.get('content', ''): pass
        if len(breakpoints) < 3:
            common_defaults = [320, 576, 768, 992, 1200, 1400]
            for bp in common_defaults: breakpoints.add(bp)
        return sorted(list(breakpoints))

    def _extract_css_rules(self, all_css_content_list: List[str]) -> Dict[str, Any]:
        """Extraction structurée des règles CSS (principalement pour HTML)"""
        css_rules = {
            'selectors': {}, 'properties': {}, 'media_queries': [], 'animations': [], 'custom_properties': {}
        }
        combined_css = "\n".join(all_css_content_list)
        try:
            parsed_stylesheet = tinycss2.parse_stylesheet(combined_css)
            for rule in parsed_stylesheet:
                if isinstance(rule, tinycss2.ast.QualifiedRule):
                    selector_text = ''.join(t.value for t in rule.prelude).strip()
                    if selector_text:
                        if selector_text not in css_rules['selectors']: css_rules['selectors'][selector_text] = {}
                        for declaration in rule.content:
                            if isinstance(declaration, tinycss2.ast.Declaration):
                                prop_name = declaration.name.strip()
                                prop_value = ''.join(t.value for t in declaration.value).strip()
                                css_rules['selectors'][selector_text][prop_name] = prop_value

                                if prop_name not in css_rules['properties']: css_rules['properties'][prop_name] = []
                                css_rules['properties'][prop_name].append(prop_value)

                                if prop_name.startswith('--'): css_rules['custom_properties'][prop_name] = prop_value

                elif isinstance(rule, tinycss2.ast.AtRule):
                    if rule.at_keyword == 'media':
                        css_rules['media_queries'].append({
                            'condition': ''.join(t.value for t in rule.prelude).strip(),
                            'rules_count': len([r for r in tinycss2.parse_stylesheet(rule.content) if isinstance(r, tinycss2.ast.QualifiedRule)])
                        })
                    elif rule.at_keyword in ['keyframes', 'font-face']: css_rules['animations'].append({'type': rule.at_keyword, 'name': ''.join(t.value for t in rule.prelude).strip()})
                    elif rule.at_keyword == 'import':
                        import_url_match = re.search(r'url\((["\']?)(.*?)\1\)', ''.join(t.value for t in rule.prelude))
                        if import_url_match: css_rules['media_queries'].append({'condition': f"@import {import_url_match.group(2)}", 'rules_count': 0})
        except Exception as e: logger.warning(f"Erreur lors du parsing CSS avec tinycss2: {e}")
        return css_rules

    async def _extract_metadata_profile(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extraction du profil de métadonnées (principalement pour HTML)"""
        metadata = {
            'title': '', 'description': '', 'keywords': [], 'og_tags': {},
            'twitter_tags': {}, 'schema_markup': [], 'lang': '', 'charset': ''
        }
        title = soup.find('title')
        if title: metadata['title'] = title.string.strip() if title.string else ''
        description = soup.find('meta', {'name': 'description'})
        if description: metadata['description'] = description.get('content', '').strip()
        keywords = soup.find('meta', {'name': 'keywords'})
        if keywords and keywords.get('content'): metadata['keywords'] = [k.strip() for k in keywords.get('content').split(',')]
        og_tags = soup.find_all('meta', {'property': lambda x: x and x.startswith('og:')})
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            metadata['og_tags'][property_name] = tag.get('content', '').strip()
        twitter_tags = soup.find_all('meta', {'name': lambda x: x and x.startswith('twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            metadata['twitter_tags'][name] = tag.get('content', '').strip()
        schema_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        for script in schema_scripts:
            if script.string:
                try: metadata['schema_markup'].append(json.loads(script.string))
                except json.JSONDecodeError: logger.warning("Erreur de décodage JSON dans le script Schema Markup.")
        html_tag = soup.find('html')
        if html_tag: metadata['lang'] = html_tag.get('lang', '').strip()
        charset_meta = soup.find('meta', {'charset': True})
        if charset_meta: metadata['charset'] = charset_meta.get('charset', '').strip()
        elif soup.find('meta', {'http-equiv': 'Content-Type'}):
            content_type = soup.find('meta', {'http-equiv': 'Content-Type'}).get('content', '')
            charset_match = re.search(r'charset=([\w-]+)', content_type)
            if charset_match: metadata['charset'] = charset_match.group(1).strip()
        return metadata

    async def _analyze_design_mood_with_ai(self, soup: BeautifulSoup, style_data: Dict[str, Any]) -> str:
        """Analyse du mood design avec IA"""
        if not self.openai_client.api_key or self.openai_client.api_key.strip() in ['', 'your-openai-key', 'your-anthropic-key']:
            logger.warning("Clé API OpenAI non configurée ou valeur par défaut. Impossible d'analyser le mood design avec l'IA. Retourne le fallback 'modern'.")
            return "modern"

        analysis_prompt = f"""
        Analyse ce profil stylistique d'une page web et détermine son "mood" design.

        Palette de couleurs: {style_data.get('colors', [])}
        Typographie: {json.dumps(style_data.get('typography', {}), indent=2)}
        Layout: {json.dumps(style_data.get('layout', {}), indent=2)}
        Nombre d'éléments analysés: {len(soup.find_all())}

        Détermine le mood design parmi ces catégories:
        - minimal: Design épuré, beaucoup d'espace blanc
        - corporate: Professionnel, couleurs sobres
        - creative: Créatif, couleurs vives, layouts non-conventionnels
        - elegant: Raffiné, typographie soignée
        - modern: Tendances actuelles, gradients, animations
        - classic: Intemporel, couleurs neutres
        - bold: Audacieux, contrastes forts
        - friendly: Accessible, couleurs chaudes

        Réponds avec un seul mot suivi d'une courte justification (max 50 mots).
        Format: "mood_name: justification"
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Tu es un expert en design web et UX. Fournis des analyses concises et précises."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            mood_analysis = response.choices[0].message.content
            return mood_analysis.split(':')[0].strip() if ':' in mood_analysis else mood_analysis.strip()
        # CORRECTION 3: L'exception `APIStatusError` est dépréciée, utilisation de `APIError`
        except openai.APIError as e:
            logger.error(f"Erreur API OpenAI lors de l'analyse du mood design: {e}")
            return "modern"
        except Exception as e:
            logger.warning(f"Erreur inattendue lors de l'analyse du mood design IA: {e}")
            return "modern"

    def _calculate_analysis_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """
        Calcule le score de confiance de l'analyse en fonction de la complétude des données,
        en utilisant une pondération pour chaque facteur (pour analyse HTML).
        """
        confidence_factors = {}
        weights = {
            'color_palette': 0.15, 'typography': 0.15, 'layout_patterns': 0.15,
            'spacing_system': 0.10, 'visual_hierarchy': 0.10, 'branding_elements': 0.10,
            'responsive_breakpoints': 0.05, 'css_rules': 0.15, 'metadata_profile': 0.05
        }

        color_count = len(analysis_data.get('color_palette', []))
        if 5 <= color_count <= 10: confidence_factors['color_palette'] = 1.0
        elif color_count > 10: confidence_factors['color_palette'] = 1.0 - (color_count - 10) * 0.05
        else: confidence_factors['color_palette'] = color_count / 5.0
        confidence_factors['color_palette'] = max(0.0, min(1.0, confidence_factors['color_palette']))

        typography = analysis_data.get('typography', {})
        typo_score = 0
        if typography and typography.get('font_families') and len(typography['font_families']) > 0: typo_score += 0.3
        if typography and typography.get('font_sizes') and len(typography['font_sizes']) > 0: typo_score += 0.3
        if typography and typography.get('font_weights') and len(typography['font_weights']) > 0: typo_score += 0.2
        if typography and typography.get('text_hierarchy') and len(typography['text_hierarchy']) > 0: typo_score += 0.2
        confidence_factors['typography'] = typo_score

        layout = analysis_data.get('layout_patterns', {})
        layout_score = 0
        if layout and (layout.get('grid_systems') or layout.get('flexbox_usage')): layout_score += 0.5
        if layout and layout.get('header_footer_structure', {}).get('has_header'): layout_score += 0.25
        if layout and layout.get('header_footer_structure', {}).get('has_footer'): layout_score += 0.25
        confidence_factors['layout_patterns'] = layout_score

        spacing = analysis_data.get('spacing_system', {})
        spacing_score = 0
        if spacing and (spacing.get('margin_patterns', {}).get('common_values') or spacing.get('padding_patterns', {}).get('common_values')): spacing_score += 0.7
        if spacing and spacing.get('spacing_scale') and len(spacing['spacing_scale']) > 0: spacing_score += 0.3
        confidence_factors['spacing_system'] = spacing_score

        hierarchy = analysis_data.get('visual_hierarchy', {})
        hierarchy_score = 0
        if hierarchy and hierarchy.get('heading_structure') and len(hierarchy['heading_structure']) > 0: hierarchy_score += 0.6
        if hierarchy and hierarchy.get('emphasis_patterns') and len(hierarchy['emphasis_patterns']) > 0: hierarchy_score += 0.4
        confidence_factors['visual_hierarchy'] = hierarchy_score

        branding = analysis_data.get('branding_elements', {})
        branding_score = 0
        if branding and branding.get('logo_detection') and len(branding['logo_detection']) > 0: branding_score += 0.7
        if branding and branding.get('consistent_elements') and len(branding['consistent_elements']) > 0: branding_score += 0.3
        confidence_factors['branding_elements'] = branding_score

        responsive_breakpoints = analysis_data.get('responsive_breakpoints', [])
        responsive_score = min(len(responsive_breakpoints) / 4.0, 1.0)
        confidence_factors['responsive_breakpoints'] = responsive_score

        css_rules = analysis_data.get('css_rules', {})
        css_score = 0
        if css_rules and css_rules.get('selectors') and len(css_rules['selectors']) > 0: css_score += 0.5
        if css_rules and css_rules.get('properties') and len(css_rules['properties']) > 0: css_score += 0.5
        confidence_factors['css_rules'] = css_score

        metadata = analysis_data.get('metadata_profile', {})
        metadata_score = 0
        if metadata.get('title'): metadata_score += 0.25
        if metadata.get('description'): metadata_score += 0.25
        if metadata.get('og_tags') and len(metadata['og_tags']) > 0: metadata_score += 0.25
        if metadata.get('schema_markup') and len(metadata['schema_markup']) > 0: metadata_score += 0.25
        confidence_factors['metadata_profile'] = metadata_score

        final_confidence = 0.0
        total_weight = 0.0

        for factor_name, weight in weights.items():
            if factor_name in confidence_factors:
                final_confidence += confidence_factors[factor_name] * weight
                total_weight += weight

        return round(final_confidence / total_weight, 2) if total_weight > 0 else 0.0

    def _calculate_pdf_analysis_confidence(self, pdf_analysis_results: Dict[str, Any]) -> float:
        """
        Calcule le score de confiance de l'analyse pour un document PDF.
        Ce score est basé sur la présence de données extraites du PDF.
        """
        score = 0.0
        # Poids des facteurs pour le PDF
        weights = {
            'color_palette': 0.4,
            'typography': 0.3,
            'text_content': 0.2,
            'page_count': 0.05,
            'image_count': 0.05
        }

        if pdf_analysis_results.get('color_palette'):
            score += weights['color_palette']
        if pdf_analysis_results.get('typography') and pdf_analysis_results['typography'].get('font_families'):
            score += weights['typography']
        if pdf_analysis_results.get('text_content'):
            score += weights['text_content']
        if pdf_analysis_results.get('page_count') is not None and pdf_analysis_results['page_count'] > 0:
            score += weights['page_count']
        if pdf_analysis_results.get('image_count') is not None and pdf_analysis_results['image_count'] > 0:
            score += weights['image_count']
        
        return round(score, 2)

    async def _analyze_pdf_document(self, pdf_data: bytes, extract_style: bool, extract_text: bool) -> Dict[str, Any]:
        """
        Analyse un document PDF pour en extraire le style et/ou le texte.
        """
        results = {
            'color_palette': [],
            'typography': {'font_families': {}, 'font_sizes': [], 'details': 'Extrait du PDF'},
            'text_content': "",
            'page_count': 0,
            'image_count': 0
        }
        
        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            results['page_count'] = doc.page_count
            
            all_pdf_text = []
            # CORRECTION 4 & 5: Utilisation d'une liste de dictionnaires pour les données d'image
            pdf_image_data_list = []
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                
                if extract_text:
                    all_pdf_text.append(page.get_text())

                if extract_style:
                    # Extraction des polices
                    fonts = page.get_fonts()
                    for font in fonts:
                        font_name = font[3].split('+')[-1]
                        results['typography']['font_families'][font_name] = results['typography']['font_families'].get(font_name, 0) + 1

                    # CORRECTION 4 & 5: Logique d'extraction d'image améliorée et plus efficace
                    # Rendu de la page en RGB pour l'analyse, réduit pour la performance
                    # Forcer RGB simplifie le traitement des couleurs plus tard
                    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), colorspace=fitz.csRGB, alpha=False)
                    
                    # Passer les données brutes du pixmap pour éviter l'encodage/décodage inefficace en PNG
                    image_data = {
                        "width": pix.width,
                        "height": pix.height,
                        "samples": pix.samples,
                        "colorspace": "RGB"  # Nous avons forcé l'espace colorimétrique RGB
                    }
                    pdf_image_data_list.append(image_data)
                    
                    results['image_count'] += len(page.get_images())

            doc.close()

            if extract_text:
                results['text_content'] = "\n".join(all_pdf_text)
            
            if extract_style and pdf_image_data_list:
                # CORRECTION 4 & 5: Passer la nouvelle structure de données à la fonction d'extraction de palette
                results['color_palette'] = await self._extract_color_palette([], soup=None, pdf_image_data_list=pdf_image_data_list)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du document PDF: {e}")
            # Réinitialiser les résultats en cas d'erreur
            results['color_palette'] = []
            results['typography'] = {'font_families': {}, 'font_sizes': [], 'details': 'Erreur lors de l\'analyse du PDF'}
            results['text_content'] = ""
            results['page_count'] = 0
            results['image_count'] = 0

        return results


    async def apply_style_transfer(self, target_html: str, style_fingerprint: StyleFingerprint,
                                   transfer_options: Dict[str, Any] = None) -> str:
        """Application intelligente du transfert de style"""
        if transfer_options is None:
            transfer_options = {
                'preserve_layout': True, 'transfer_colors': True,
                'transfer_typography': True, 'transfer_spacing': True,
                'intensity': 0.8
            }

        logger.info(f"Application transfert style (confiance: {style_fingerprint.confidence_score})")

        soup = BeautifulSoup(target_html, 'html.parser')
        transfer_css = await self._generate_transfer_css(style_fingerprint, transfer_options)

        style_tag = soup.new_tag('style')
        style_tag.string = transfer_css

        head = soup.find('head')
        if not head:
            head = soup.new_tag('head')
            soup.insert(0, head)
        head.append(style_tag)

        await self._apply_transfer_classes(soup, style_fingerprint, transfer_options)

        return str(soup)


    async def _generate_transfer_css(self, fingerprint: StyleFingerprint, options: Dict[str, Any]) -> str:
        """Génère le CSS de transfert basé sur l'empreinte stylistique"""
        css_parts = []
        intensity = options.get('intensity', 0.8)

        css_parts.append("""
/* Neural Style Transfer - Generated CSS */
:root {
""")

        if options.get('transfer_colors', True) and fingerprint.color_palette:
            for i, color in enumerate(fingerprint.color_palette[:8]): css_parts.append(f"    --color-{i}: {color};")
            css_parts.append(f"    --primary-color: {fingerprint.color_palette[0]};")
            if len(fingerprint.color_palette) > 1: css_parts.append(f"    --secondary-color: {fingerprint.color_palette[1]};")

        if options.get('transfer_typography', True):
            typography = fingerprint.typography
            font_families = list(typography.get('font_families', {}).keys())
            if font_families:
                css_parts.append(f"    --primary-font: '{font_families[0]}', sans-serif;")
                if len(font_families) > 1: css_parts.append(f"    --secondary-font: '{font_families[1]}', sans-serif;")

        if options.get('transfer_spacing', True):
            spacing = fingerprint.spacing_system
            spacing_scale = spacing.get('spacing_scale', self._DEFAULT_SPACING_SCALE)
            for i, value in enumerate(spacing_scale[:6]): css_parts.append(f"    --spacing-{i}: {value}px;")

        css_parts.append("}")
        css_parts.append("""
/* Base styles */
body {""")

        if options.get('transfer_colors', True) and fingerprint.color_palette:
            bg_color = fingerprint.color_palette[-1] if len(fingerprint.color_palette) > 2 else "#ffffff"
            text_color = fingerprint.color_palette[0]
            css_parts.append(f"    background-color: {bg_color};")
            css_parts.append(f"    color: {text_color};")

        if options.get('transfer_typography', True):
            font_families = list(fingerprint.typography.get('font_families', {}).keys())
            if font_families: css_parts.append(f"    font-family: var(--primary-font, '{font_families[0]}', sans-serif);")

        css_parts.append("}")

        if options.get('transfer_typography', True):
            heading_hierarchy = fingerprint.visual_hierarchy.get('heading_structure', {})
            # Vérifie si la hiérarchie est applicable (i.e. pas une analyse PDF simplifiée)
            # Une hiérarchie de titre PDF n'a pas la même structure qu'une hiérarchie HTML.
            if 'pdf_document' not in heading_hierarchy or not heading_hierarchy['pdf_document']:
                for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if tag in heading_hierarchy:
                        css_parts.append(f"""
{tag} {{
    color: var(--primary-color, {fingerprint.color_palette[0] if fingerprint.color_palette else '#333'});
    margin-bottom: var(--spacing-2, 16px);
}}""")

        if fingerprint.color_palette:
            link_color = fingerprint.color_palette[1] if len(fingerprint.color_palette) > 1 else fingerprint.color_palette[0]
            css_parts.append(f"""
a {{
    color: {link_color};
    text-decoration: none;
    transition: opacity 0.2s ease;
}}

a:hover {{
    opacity: 0.8;
}}""")

        # N'applique les media queries que si elles ont été détectées pour HTML (non-PDF)
        if fingerprint.responsive_breakpoints and 'pdf_document' not in fingerprint.css_rules:
            for breakpoint in fingerprint.responsive_breakpoints:
                css_parts.append(f"""
@media (max-width: {breakpoint}px) {{
    body {{ font-size: 14px; }}
    .container {{ padding: var(--spacing-1, 8px); }}
}}""")

        return '\n'.join(css_parts)


    async def _apply_transfer_classes(self, soup: BeautifulSoup, fingerprint: StyleFingerprint,
                                   options: Dict[str, Any]):
        """Applique les classes de transfert aux éléments HTML"""

        if options.get('transfer_colors', True):
            important_elements = soup.select('h1, h2, .btn, button, .cta')
            for element in important_elements:
                existing_classes = element.get('class', [])
                existing_classes.append('primary-accent')
                element['class'] = existing_classes

        # N'applique la mise en page que si ce n'est pas un document PDF original
        if not options.get('preserve_layout', True) and 'pdf_document' not in fingerprint.layout_patterns:
            layout_patterns = fingerprint.layout_patterns
            if layout_patterns.get('grid_systems'):
                containers = soup.select('div, section, main')
                for container in containers[:3]:
                    existing_classes = container.get('class', [])
                    existing_classes.append('transfer-grid')
                    container['class'] = existing_classes

        # N'applique l'espacement que si ce n'est pas un document PDF original
        if options.get('transfer_spacing', True) and 'pdf_document' not in fingerprint.spacing_system:
            sections = soup.select('section, article, div')
            for section in sections[:5]:
                existing_classes = section.get('class', [])
                existing_classes.append('transfer-spacing')
                section['class'] = existing_classes


    def save_style_fingerprint(self, fingerprint: StyleFingerprint, filepath: str):
        """Sauvegarde l'empreinte stylistique"""
        fingerprint_data = {
            'color_palette': fingerprint.color_palette, 'typography': fingerprint.typography,
            'layout_patterns': fingerprint.layout_patterns, 'spacing_system': fingerprint.spacing_system,
            'visual_hierarchy': fingerprint.visual_hierarchy, 'branding_elements': fingerprint.branding_elements,
            'responsive_breakpoints': fingerprint.responsive_breakpoints, 'css_rules': fingerprint.css_rules,
            'metadata_profile': fingerprint.metadata_profile, 'design_mood': fingerprint.design_mood,
            'confidence_score': fingerprint.confidence_score, 'timestamp': asyncio.get_event_loop().time(),
            'pdf_text_content': fingerprint.pdf_text_content, 'pdf_page_count': fingerprint.pdf_page_count,
            'pdf_image_count': fingerprint.pdf_image_count
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(fingerprint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Empreinte sauvegardée: {filepath}")
        except IOError as e:
            logger.error(f"Erreur d'écriture du fichier '{filepath}': {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la sauvegarde de l'empreinte: {e}")


    def load_style_fingerprint(self, filepath: str) -> StyleFingerprint:
        """Charge une empreinte stylistique"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Empreinte chargée: {filepath}")
            return StyleFingerprint(
                color_palette=data['color_palette'], typography=data['typography'],
                layout_patterns=data['layout_patterns'], spacing_system=data['spacing_system'],
                visual_hierarchy=data['visual_hierarchy'], branding_elements=data['branding_elements'],
                responsive_breakpoints=data['responsive_breakpoints'], css_rules=data['css_rules'],
                metadata_profile=data['metadata_profile'], design_mood=data['design_mood'],
                confidence_score=data['confidence_score'],
                pdf_text_content=data.get('pdf_text_content'),
                pdf_page_count=data.get('pdf_page_count'),
                pdf_image_count=data.get('pdf_image_count')
            )
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: '{filepath}'.")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON dans '{filepath}': {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement de l'empreinte: {e}")
            raise


# Fonctions utilitaires

async def main():
    """Fonction principale de démonstration"""
    # Remplacez ces clés par vos vraies clés API
    ai_config = {
        'openai_api_key': 'your-openai-key', # EXEMPLE: "sk-abcdef1234567890"
        'anthropic_api_key': 'your-anthropic-key' # EXEMPLE: "sk-ant-abcdef1234567890"
    }

    async with WebStyleAnalyzer(ai_config) as analyzer:
        print("--- Démonstration HTML ---")
        reference_url_html = "https://www.wikipedia.org/"
        try:
            style_fingerprint_html = await analyzer.analyze_reference_page(reference_url_html, is_url=True, is_pdf=False)
            print(f"✅ Analyse HTML terminée (confiance: {style_fingerprint_html.confidence_score})")
            print(f"🎭 Mood détecté: {style_fingerprint_html.design_mood}")
            print(f"🎨 Couleurs extraites: {len(style_fingerprint_html.color_palette)}")
            print(f"📝 Familles de police: {len(style_fingerprint_html.typography.get('font_families', {}))}")
            analyzer.save_style_fingerprint(style_fingerprint_html, 'style_fingerprint_html.json')
            print("💾 Empreinte stylistique HTML sauvegardée dans 'style_fingerprint_html.json'")

            target_html = """
            <!DOCTYPE html>
            <html>
            <head><title>Page cible</title></head>
            <body>
                <h1>Titre principal de la page cible</h1>
                <p style="font-size: 18px; line-height: 1.5; color: #444;">Ceci est un paragraphe de texte pour tester le transfert de style.</p>
                <button style="background-color: lightgray; padding: 10px 20px; border-radius: 5px; margin: 16px;">Cliquez-moi</button>
                <section style="margin: 32px 16px; padding: 24px; border: 1px solid #ccc; background-color: rgba(200, 220, 240, 0.7);">
                    <h2>Section Contenu</h2>
                    <p>Un autre paragraphe dans une section avec des styles inline.</p>
                    <a href="#" style="color: blue; text-decoration: underline;">Un lien</a>
                </section>
                <div class="m-8 p-4 bg-red-100">
                    <p>Ceci est un div avec des classes utilitaires d'espacement.</p>
                </div>
                <footer>
                    <p>Pied de page.</p>
                </footer>
            </body>
            </html>
            """
            print("🔄 Application du transfert de style HTML...")
            styled_html = await analyzer.apply_style_transfer(target_html, style_fingerprint_html)
            with open('result_styled_html.html', 'w', encoding='utf-8') as f:
                f.write(styled_html)
            print("✅ Transfert HTML terminé - Résultat sauvegardé dans 'result_styled_html.html'")

        except Exception as e:
            logger.error(f"Une erreur est survenue lors de l'exécution de la démonstration HTML: {e}")

        print("\n--- Démonstration PDF ---")
        # Créer un PDF factice pour la démonstration si aucun n'existe
        pdf_filepath = "sample.pdf"
        if not Path(pdf_filepath).exists():
            print(f"Création d'un fichier PDF factice '{pdf_filepath}' pour la démonstration...")
            doc = fitz.open()
            page = doc.new_page()
            page.insert_textbox(
                (50, 50, 200, 100),
                "Ceci est un exemple de texte PDF.\n\n"
                "Le style du texte dans les PDF est géré différemment du HTML.\n"
                "Ceci est une deuxième ligne avec un style basique.",
                fontsize=12,
                fontname="Helvetica",
                color=(0, 0, 0) # Noir
            )
            page.insert_textbox(
                (50, 150, 200, 200),
                "Un autre paragraphe avec une police différente.",
                fontsize=18,
                fontname="Times-Roman",
                color=(1, 0, 0) # Rouge
            )
            # Insérer une forme pour tester la détection de couleur à partir d'un pixelmap
            page.draw_rect(fitz.Rect(250, 50, 300, 100), color=(0, 0, 1), fill=(0, 0, 1)) # Carré bleu
            doc.save(pdf_filepath)
            doc.close()
            print("Fichier PDF factice créé.")

        try:
            # Exemple d'analyse PDF: extraction du style et du texte
            style_fingerprint_pdf = await analyzer.analyze_reference_page(
                pdf_filepath, is_url=False, is_pdf=True, extract_pdf_style=True, extract_pdf_text=True
            )
            print(f"✅ Analyse PDF terminée (confiance: {style_fingerprint_pdf.confidence_score})")
            print(f"📄 Nombre de pages PDF: {style_fingerprint_pdf.pdf_page_count}")
            print(f"🖼️ Nombre d'images PDF détectées: {style_fingerprint_pdf.pdf_image_count}")
            print(f"🎨 Couleurs extraites du PDF: {style_fingerprint_pdf.color_palette}")
            print(f"📝 Familles de police du PDF: {style_fingerprint_pdf.typography.get('font_families', {})}")
            
            if style_fingerprint_pdf.pdf_text_content:
                print(f"\nExtrait du texte du PDF:\n--- START TEXT ---\n{style_fingerprint_pdf.pdf_text_content[:500]}...\n--- END TEXT ---")
            
            analyzer.save_style_fingerprint(style_fingerprint_pdf, 'style_fingerprint_pdf.json')
            print("💾 Empreinte stylistique PDF sauvegardée dans 'style_fingerprint_pdf.json'")

        except Exception as e:
            logger.error(f"Une erreur est survenue lors de l'exécution de la démonstration PDF: {e}")


if __name__ == "__main__":
    asyncio.run(main())

