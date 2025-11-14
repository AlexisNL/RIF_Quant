import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy.stats import gaussian_kde  # Méthode à noyau pour l'estimation de densité

class RIFRegression:
    def __init__(self, data: pd.DataFrame, y_col: str, statistic: float or str):
        self.data = data.copy()
        self.y_col = y_col
        self.statistic = statistic
        self.y = self.data[self.y_col]
        self.X = None
        self.model = None
        self.rif_values = None
        self.stat_value = None

        if len(self.y) == 0:
            raise ValueError(f"La colonne '{self.y_col}' est vide ou ne contient aucune valeur valide.")

    def _compute_statistic(self) -> float:
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour calculer la statistique.")
        if isinstance(self.statistic, float) and 0 < self.statistic < 1:
            self.stat_value = np.quantile(self.y, self.statistic)
        elif self.statistic == 'mean':
            self.stat_value = np.mean(self.y)
        elif self.statistic == 'std':
            self.stat_value = np.std(self.y)
        else:
            raise ValueError("La statistique doit être un quantile (float entre 0 et 1), 'mean', ou 'std'.")
        return self.stat_value

    def _compute_rif(self) -> np.ndarray:
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour calculer la RIF.")
        stat_value = self._compute_statistic()
        n = len(self.y)
        if isinstance(self.statistic, float) and 0 < self.statistic < 1:
            indicator = (self.y <= stat_value).astype(int)
            density = norm.pdf(norm.ppf(self.statistic))
            self.rif_values = stat_value + (indicator - self.statistic) / (self.statistic * (1 - self.statistic) * density)
        elif self.statistic == 'mean':
            self.rif_values = self.y - stat_value
        elif self.statistic == 'std':
            self.rif_values = (self.y - np.mean(self.y))**2 - stat_value**2
        return self.rif_values

    def fit(self, x_cols: list) -> 'RIFRegression':
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour ajuster le modèle.")
        if not all(col in self.data.columns for col in x_cols):
            missing_cols = [col for col in x_cols if col not in self.data.columns]
            raise ValueError(f"Les colonnes suivantes sont manquantes dans les données : {missing_cols}")

        self.X = self.data[x_cols].copy()
        self.X = add_constant(self.X, has_constant='add')

        if len(self.X) != len(self.y):
            raise ValueError("Le nombre de lignes dans X et y ne correspond pas.")

        self.rif_values = self._compute_rif()
        self.model = OLS(self.rif_values, self.X).fit()
        return self

    def summary(self) -> str:
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Appelez d'abord fit().")
        return self.model.summary()

    def get_coefficients(self) -> pd.Series:
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Appelez d'abord fit().")
        # Exclut la constante et retourne les coefficients des covariables
        return pd.Series(self.model.params[1:], index=self.X.columns[1:])

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Appelez d'abord fit().")
        new_data = add_constant(new_data, has_constant='add')
        return self.model.predict(new_data)


class RollingRIFRegression:
    def __init__(self, data: pd.DataFrame, y_col: str, statistic: float or str, window_size: int = 30):
        self.data = data.copy()
        self.y_col = y_col
        self.statistic = statistic
        self.window_size = window_size
        self.y = self.data[self.y_col]
        self.X = None
        self.models = {}
        self.rif_values = None
        self.stat_values = None
        self.coefficients = None
        if len(self.y) == 0:
            raise ValueError(f"La colonne '{self.y_col}' est vide ou ne contient aucune valeur valide.")

    def _compute_rolling_statistic(self) -> pd.Series:
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour calculer la statistique.")
        if isinstance(self.statistic, float) and 0 < self.statistic < 1:
            self.stat_values = self.y.rolling(window=self.window_size, min_periods=1).quantile(self.statistic)
        elif self.statistic == 'mean':
            self.stat_values = self.y.rolling(window=self.window_size, min_periods=1).mean()
        elif self.statistic == 'std':
            self.stat_values = self.y.rolling(window=self.window_size, min_periods=1).std()
        else:
            raise ValueError("La statistique doit être un quantile (float entre 0 et 1), 'mean', ou 'std'.")
        return self.stat_values

    def _compute_rif(self) -> pd.Series:
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour calculer la RIF.")

        stat_values = self._compute_rolling_statistic()

        if isinstance(self.statistic, float) and 0 < self.statistic < 1:
            indicator = (self.y <= stat_values).astype(int)
            density = norm.pdf(norm.ppf(self.statistic))
            self.rif_values = stat_values + (indicator - self.statistic) / (self.statistic * (1 - self.statistic) * density)
        elif self.statistic == 'mean':
            self.rif_values = self.y - stat_values
        elif self.statistic == 'std':
            self.rif_values = (self.y - self.y.rolling(window=self.window_size, min_periods=1).mean())**2 - stat_values**2

        return self.rif_values

    def fit(self, x_cols: list) -> 'RollingRIFRegression':
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour ajuster le modèle.")
        if not all(col in self.data.columns for col in x_cols):
            missing_cols = [col for col in x_cols if col not in self.data.columns]
            raise ValueError(f"Les colonnes suivantes sont manquantes dans les données : {missing_cols}")

        self.X = self.data[x_cols].copy()
        self.X = add_constant(self.X, has_constant='add')

        self.rif_values = self._compute_rif()
        self.X = self.X.loc[self.rif_values.index]
        self.y = self.y.loc[self.rif_values.index]

        if len(self.X) != len(self.rif_values):
            raise ValueError("Le nombre de lignes dans X et rif_values ne correspond pas.")

        # Ajustement d'un modèle OLS pour chaque fenêtre glissante
        self.coefficients = {}
        for i in range(self.window_size, len(self.rif_values)):
            X_window = self.X.iloc[i-self.window_size+1:i+1]
            y_window = self.rif_values.iloc[i-self.window_size+1:i+1]
            model = OLS(y_window, X_window).fit()
            # Utiliser l'index (date) de l'observation actuelle comme clé
            self.coefficients[self.rif_values.index[i]] = model.params[1:]  # Exclut la constante

        return self

    def get_coefficients(self) -> pd.DataFrame:
        if self.coefficients is None:
            raise ValueError("Le modèle n'a pas été ajusté. Appelez d'abord fit().")
        return pd.DataFrame.from_dict(self.coefficients, orient='index', columns=self.X.columns[1:])

    def summary(self) -> str:
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Appelez d'abord fit().")
        return self.model.summary()

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Appelez d'abord fit().")
        new_data = add_constant(new_data, has_constant='add')
        return self.model.predict(new_data)
    
class RollingRIFQRegression:
    def __init__(self, data: pd.DataFrame, y_col: str, quantile: float, window_size: int = 30):
        """
        Classe pour effectuer une régression RIF (Recentered Influence Function) glissante
        sur un quantile spécifique, avec estimation non paramétrique de la densité.

        Args:
            data (pd.DataFrame): DataFrame contenant les données.
            y_col (str): Nom de la colonne cible (variable dépendante).
            quantile (float): Quantile pour la régression (doit être entre 0 et 1).
            window_size (int): Taille de la fenêtre glissante. Par défaut 30.
        """
        self.data = data.copy()
        self.y_col = y_col
        self.quantile = quantile
        self.window_size = window_size
        self.y = self.data[self.y_col]
        self.X = None
        self.models = {}
        self.rif_values = None
        self.coefficients = None

        if not (0 < self.quantile < 1):
            raise ValueError("Le quantile doit être un float strictement compris entre 0 et 1.")
        if len(self.y) == 0:
            raise ValueError(f"La colonne '{self.y_col}' est vide ou ne contient aucune valeur valide.")

    def _compute_rolling_quantile(self) -> pd.Series:
        """Calcule les valeurs du quantile sur une fenêtre glissante."""
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour calculer le quantile.")
        self.quantile_values = self.y.rolling(window=self.window_size, min_periods=1).quantile(self.quantile)
        return self.quantile_values

    def _estimate_density_at_quantile(self, y_window: pd.Series) -> float:
        """
        Estime la densité au niveau du quantile en utilisant une méthode à noyau (KDE).
        Args:
            y_window (pd.Series): Série des valeurs de y dans la fenêtre glissante.
        Returns:
            float: Estimation de la densité au niveau du quantile.
        """
        # Supprimer les NaN
        y_clean = y_window.dropna().values

        # Si pas assez de données, retourner une valeur par défaut (ex. : densité de la normale standard)
        if len(y_clean) < 2:
            return norm.pdf(norm.ppf(self.quantile))

        # Estimation de densité par noyau (KDE)
        kde = gaussian_kde(y_clean)
        quantile_value = np.quantile(y_clean, self.quantile)
        density = kde(quantile_value)

        # Éviter les densités trop faibles (pour éviter des divisions par zéro ou des valeurs extrêmes)
        if density <= 0:
            return 1e-6  # Valeur minimale arbitraire
        return density

    def _compute_rif(self) -> pd.Series:
        """Calcule les valeurs de la fonction d'influence recentrée (RIF) pour le quantile."""
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour calculer la RIF.")

        quantile_values = self._compute_rolling_quantile()
        indicator = (self.y <= quantile_values).astype(int)

        # Estimer la densité pour chaque fenêtre glissante
        density_values = pd.Series(index=self.y.index, dtype=float)
        for i in range(self.window_size, len(self.y)):
            y_window = self.y.iloc[max(0, i - self.window_size + 1):i+1]
            density_values.iloc[i] = self._estimate_density_at_quantile(y_window)

        # Pour les premières observations (avant que la fenêtre ne soit pleine), utiliser une densité par défaut
        density_values.iloc[:self.window_size] = norm.pdf(norm.ppf(self.quantile))

        self.rif_values = quantile_values + (indicator - self.quantile) / (self.quantile * (1 - self.quantile) * density_values)
        return self.rif_values

    def fit(self, x_cols: list) -> 'RollingQuantileRIFRegression':
        """
        Ajuste le modèle de régression RIF sur les colonnes spécifiées.
        Args:
            x_cols (list): Liste des noms des colonnes à utiliser comme variables explicatives.
        Returns:
            RollingQuantileRIFRegression: L'instance de la classe pour le chaînage de méthodes.
        """
        if len(self.y) == 0:
            raise ValueError("Aucune donnée disponible pour ajuster le modèle.")
        if not all(col in self.data.columns for col in x_cols):
            missing_cols = [col for col in x_cols if col not in self.data.columns]
            raise ValueError(f"Les colonnes suivantes sont manquantes dans les données : {missing_cols}")

        self.X = self.data[x_cols].copy()
        self.X = add_constant(self.X, has_constant='add')
        self.rif_values = self._compute_rif()

        # Aligner les index de X et rif_values
        self.X = self.X.loc[self.rif_values.index]
        self.y = self.y.loc[self.rif_values.index]

        if len(self.X) != len(self.rif_values):
            raise ValueError("Le nombre de lignes dans X et rif_values ne correspond pas.")

        # Ajustement d'un modèle OLS pour chaque fenêtre glissante
        self.coefficients = {}
        for i in range(self.window_size, len(self.rif_values)):
            X_window = self.X.iloc[i-self.window_size+1:i+1]
            y_window = self.rif_values.iloc[i-self.window_size+1:i+1]
            model = OLS(y_window, X_window).fit()
            # Utiliser l'index (date) de l'observation actuelle comme clé
            self.coefficients[self.rif_values.index[i]] = model.params[1:]  # Exclut la constante

        return self

    def get_coefficients(self) -> pd.DataFrame:
        """
        Récupère les coefficients estimés pour chaque fenêtre glissante.
        Returns:
            pd.DataFrame: DataFrame contenant les coefficients estimés.
        """
        if self.coefficients is None:
            raise ValueError("Le modèle n'a pas été ajusté. Appelez d'abord fit().")
        return pd.DataFrame.from_dict(self.coefficients, orient='index', columns=self.X.columns[1:])
