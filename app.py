#!/usr/bin/env python3

"""
Version de DEBUG avec traces détaillées
"""

import sys
import traceback

def debug_print(message, level="INFO"):
    """Print de debug avec flush forcé"""
    print(f"[{level}] {message}")
    sys.stdout.flush()

def main_debug():
    """Version de debug de main() avec traces détaillées"""
    debug_print("🚀 === DÉBUT DU DEBUG ===")
    
    try:
        debug_print("1. Test des imports de base...")
        import asyncio
        debug_print("   ✅ asyncio OK")
        
        import aiohttp
        debug_print("   ✅ aiohttp OK")
        
        from bs4 import BeautifulSoup
        debug_print("   ✅ BeautifulSoup OK")
        
        import webcolors
        debug_print("   ✅ webcolors OK")
        
        debug_print("2. Test de création de session aiohttp...")
        
        async def test_session():
            debug_print("   2a. Création session...")
            session = aiohttp.ClientSession()
            debug_print("   2b. Session créée")
            
            debug_print("   2c. Test de requête simple...")
            try:
                async with session.get('https://httpbin.org/status/200', timeout=aiohttp.ClientTimeout(total=5)) as response:
                    debug_print(f"   2d. Réponse reçue: {response.status}")
                    text = await response.text()
                    debug_print(f"   2e. Texte lu: {len(text)} caractères")
            except Exception as e:
                debug_print(f"   2d. ERREUR requête: {e}", "ERROR")
                raise
            finally:
                debug_print("   2f. Fermeture session...")
                await session.close()
                debug_print("   2g. Session fermée")
        
        debug_print("3. Exécution test asyncio...")
        asyncio.run(test_session())
        debug_print("4. Test asyncio terminé avec succès")
        
        # Test de votre classe
        debug_print("5. Test de la classe WebStyleAnalyzer...")
        
        # Import de votre code (simplifié pour le test)
        class WebStyleAnalyzerSimple:
            def __init__(self, ai_config):
                debug_print("   5a. Initialisation WebStyleAnalyzer...")
                self.ai_config = ai_config
                self.session = None
                debug_print("   5b. WebStyleAnalyzer initialisé")
            
            async def __aenter__(self):
                debug_print("   5c. Entrée context manager...")
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; StyleAnalyzer/1.0)'}
                )
                debug_print("   5d. Session créée dans context manager")
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                debug_print("   5e. Sortie context manager...")
                if self.session and not self.session.closed:
                    await self.session.close()
                    debug_print("   5f. Session fermée dans context manager")
            
            async def test_analyze(self, url):
                debug_print(f"   5g. Début analyse de {url}...")
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        debug_print(f"   5h. Réponse HTTP: {response.status}")
                        html_content = await response.text()
                        debug_print(f"   5i. HTML récupéré: {len(html_content)} caractères")
                        
                        soup = BeautifulSoup(html_content, 'html.parser')
                        debug_print(f"   5j. BeautifulSoup parsing OK: {soup.title}")
                        
                        return f"Analyse OK - {len(html_content)} caractères"
                except Exception as e:
                    debug_print(f"   5h. ERREUR dans test_analyze: {e}", "ERROR")
                    raise
        
        async def test_analyzer():
            debug_print("6. Test complet de l'analyzer...")
            ai_config = {'openai_api_key': '', 'anthropic_api_key': ''}
            
            try:
                async with WebStyleAnalyzerSimple(ai_config) as analyzer:
                    debug_print("   6a. Context manager OK")
                    result = await analyzer.test_analyze('https://httpbin.org/html')
                    debug_print(f"   6b. Résultat: {result}")
                    
            except Exception as e:
                debug_print(f"   6a. ERREUR dans analyzer: {e}", "ERROR")
                traceback.print_exc()
                raise
        
        debug_print("7. Lancement test analyzer...")
        asyncio.run(test_analyzer())
        debug_print("8. ✅ Test analyzer terminé avec succès")
        
        debug_print("🎉 === TOUS LES TESTS PASSÉS ===")
        
    except ImportError as e:
        debug_print(f"❌ ERREUR D'IMPORT: {e}", "ERROR")
        debug_print("Installez les dépendances: pip install aiohttp beautifulsoup4 webcolors", "ERROR")
        sys.exit(1)
        
    except Exception as e:
        debug_print(f"❌ ERREUR GÉNÉRALE: {e}", "ERROR")
        debug_print("Traceback complet:", "ERROR")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    debug_print("Starting debug script...")
    main_debug()
    debug_print("Debug script finished.")
