
import yaml
import os
import requests
from typing import Dict, Any, List, Tuple

import yfinance as yf
import pandas as pd

# Définition d'un User-Agent pour simuler un navigateur et éviter l'erreur 403
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}


def config_yaml(filepath: str) -> Dict[str, Any]:
    """
    Lit un fichier YAML à partir du chemin spécifié et retourne son contenu 
    sous forme de dictionnaire Python.

    Args:
        filepath (str): Chemin vers le fichier YAML (ex: 'config.yaml').

    Returns:
        Dict[str, Any]: Le contenu du fichier YAML.
    """
    if not os.path.exists(filepath):
        print(f"Erreur: Le fichier de configuration YAML n'a pas été trouvé à {filepath}")
        return {}
    
    try:
        # UTILISATION DU BLOC 'with open(...)' :
        # Ceci ouvre le fichier en mode lecture ('r') avec l'encodage 'utf-8'.
        # L'avantage du 'with' est qu'il garantit la fermeture automatique du fichier (file.close())
        # dès que le bloc 'with' est quitté, même en cas d'exception.
        with open(filepath, 'r', encoding='utf-8') as file:
            # La librairie PyYAML lit le contenu du fichier (objet 'file') 
            # et le convertit en un dictionnaire/liste Python.
            config = yaml.safe_load(file)
            print(f"Fichier YAML '{filepath}' chargé avec succès.")
            return config
            
    except yaml.YAMLError as e:
        print(f"Erreur lors du parsing du fichier YAML: {e}")
        return {}
    except Exception as e:
        print(f"Une erreur inattendue s'est produite lors de la lecture du fichier: {e}")
        return {}

def get_sp500_tickers() -> List[str]:
    """
    Récupère les tickers actuels du S&P 500 à partir de la page Wikipedia.
    Utilise 'requests' avec un User-Agent pour contourner l'erreur 403.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    try:
        # Utilisation de requests pour obtenir le contenu avec l'en-tête User-Agent
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status() # Lève une erreur si la réponse est mauvaise (4xx ou 5xx)
        
        # pandas.read_html lit le texte HTML brut de la réponse
        tables = pd.read_html(response.text) 
        
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [t.replace('\n', '') for t in tickers]
        print(f"Nombre de tickers S&P 500 récupérés : {len(tickers)}")
        return tickers
    except Exception as e:
        print(f"Erreur lors de la récupération des tickers S&P 500 : {e}")
        return []

def get_nasdaq100_tickers() -> List[str]:
    """
    Récupère les tickers actuels du NASDAQ 100 à partir de la page Wikipedia.
    Utilise 'requests' avec un User-Agent pour contourner l'erreur 403.
    """
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    
    try:
        # Utilisation de requests pour obtenir le contenu avec l'en-tête User-Agent
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status() 
        
        tables = pd.read_html(response.text) 
        
        # Le tableau des composants est généralement à l'index 4 ou 3
        try:
            nasdaq100_df = tables[4]
        except IndexError:
            nasdaq100_df = tables[3] 
            
        tickers = nasdaq100_df['Ticker'].tolist()
        tickers = [t.replace('\n', '') for t in tickers]
        print(f"Nombre de tickers NASDAQ 100 récupérés : {len(tickers)}")
        return tickers
    except Exception as e:
        print(f"Erreur lors de la récupération des tickers NASDAQ 100 : {e}")
        return []

# --- Fonction de Téléchargement pour Liste de Tickers (Multi-Assets) ---
def download_yfinance(tickers: list, debut: str, fin: str) -> pd.DataFrame:
    """
    Télécharge les données historiques pour une liste de symboles boursiers donnés via yfinance.
    Retourne un DataFrame avec une structure MultiIndex (Colonnes: Prix/Volume, Tickers).
    """
    if not tickers:
        return pd.DataFrame()
        
    print(f"Téléchargement de {len(tickers)} actifs du {debut} au {fin}...")
    
    try:
        # yf.download avec une liste de tickers retourne un DataFrame MultiIndex
        df = yf.download(tickers, start=debut, end=fin, progress=True)
        
        if df.empty:
            print(f"Avertissement: Aucune donnée trouvée pour les tickers spécifiés.")
        else:
            print(f"Téléchargement réussi. Dimensions du DataFrame : {df.shape}")
            
        return df
    
    except Exception as e:
        print(f"Une erreur s'est produite lors du téléchargement: {e}")
        return pd.DataFrame()

