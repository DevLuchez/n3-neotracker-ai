"""
src/classifier.py
Classificador Probabilístico — Naive Bayes
------------------------------------------
Treina um GaussianNB supervisionado com o rótulo `is_hazardous` do dataset
NASA/JPL e atribui a cada NEO uma de três classes:

    0 → Lixo Espacial          (albedo alto, diâmetro pequeno, brilho baixo)
    1 → Asteroide Inofensivo   (não-PHA padrão)
    2 → PHA                    (Potentially Hazardous Asteroid)

A triagem inicial (classe 0 vs 1+2) descarta ~99 % dos objetos comuns
antes de passar para os módulos mais caros computacionalmente.
"""

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

# Features usadas pelo classificador
FEATURES = ["magnitude", "albedo", "velocity", "diameter"]

# Limiares para heurística de "Lixo Espacial"
_JUNK_ALBEDO_MIN    = 0.35   # alto albedo → provavelmente satélite/lixo
_JUNK_DIAMETER_MAX  = 0.01   # < 10 m → provável detrito
_JUNK_MAGNITUDE_MIN = 28.0   # muito fraco → detrito pequeno


class NEOClassifier:
    """
    Classificador Naive Bayes para NEOs.

    Fluxo:
      1. Heurística rápida → Lixo Espacial (classe 0)
      2. GaussianNB → Asteroide Inofensivo (1) ou PHA (2)
    """

    CLASS_LABELS = {
        0: "Lixo Espacial",
        1: "Asteroide Inofensivo",
        2: "PHA",
    }

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("nb",     GaussianNB()),
        ])
        self._trained = False

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        """
        Treina o modelo com o DataFrame padronizado.

        Parâmetros
        ----------
        df : pd.DataFrame
            Dataset com colunas padronizadas (saída de ingestion.load_dataset).

        Retorna
        -------
        dict com métricas de treinamento.
        """
        df_train = df.copy()

        # Garantir features disponíveis
        available = [f for f in FEATURES if f in df_train.columns]
        if len(available) < 2:
            raise ValueError(f"Colunas insuficientes para treino: {available}")

        # Rótulo binário → PHA (2) ou não (1)
        y = np.where(df_train["is_hazardous"], 2, 1)

        X = df_train[available].fillna(df_train[available].median())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline.fit(X_train, y_train)
        self._trained     = True
        self._features    = available

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True,
                                       zero_division=0)
        acc = report.get("accuracy", 0)
        logger.info(f"[Classifier] Treinamento concluído | Acurácia: {acc:.2%}")
        return {"accuracy": acc, "report": report, "features_used": available}

    # ------------------------------------------------------------------
    # Classificação
    # ------------------------------------------------------------------
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifica todos os NEOs e retorna o DataFrame com colunas adicionais:
            - neo_class    : 0, 1 ou 2
            - class_label  : nome da classe
            - pha_prob     : probabilidade de ser PHA (0–1)
        """
        result = df.copy()
        result["neo_class"]   = 1  # padrão: Asteroide Inofensivo
        result["class_label"] = "Asteroide Inofensivo"
        result["pha_prob"]    = 0.0

        # --- Etapa 1: Heurística de Lixo Espacial ---
        junk_mask = self._junk_heuristic(result)
        result.loc[junk_mask, "neo_class"]   = 0
        result.loc[junk_mask, "class_label"] = "Lixo Espacial"

        # --- Etapa 2: Naive Bayes para os restantes ---
        non_junk = result[~junk_mask].copy()
        if len(non_junk) > 0 and self._trained:
            X = non_junk[self._features].fillna(
                non_junk[self._features].median()
            )
            proba = self.pipeline.predict_proba(X)

            # Índice da classe 2 (PHA) nas probabilidades
            classes = list(self.pipeline.named_steps["nb"].classes_)
            if 2 in classes:
                pha_idx = classes.index(2)
                pha_probs = proba[:, pha_idx]
            else:
                pha_probs = np.zeros(len(non_junk))

            predictions  = self.pipeline.predict(X)
            result.loc[non_junk.index, "neo_class"]   = predictions
            result.loc[non_junk.index, "class_label"] = [
                self.CLASS_LABELS.get(p, "Desconhecido") for p in predictions
            ]
            result.loc[non_junk.index, "pha_prob"] = pha_probs

        # Log resumo
        counts = result["class_label"].value_counts()
        logger.info(f"[Classifier] Resultado: {counts.to_dict()}")
        return result

    def get_phas(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """Filtra apenas os NEOs classificados como PHA."""
        return classified_df[classified_df["neo_class"] == 2].copy()

    # ------------------------------------------------------------------
    # Persistência do modelo
    # ------------------------------------------------------------------
    def save(self, filepath: str | Path) -> None:
        """
        Persiste o pipeline treinado e os metadados em disco usando joblib.

        Parâmetros
        ----------
        filepath : str | Path
            Caminho do arquivo de saída (ex: 'models/neo_classifier.joblib').
        """
        if not self._trained:
            raise RuntimeError("O modelo ainda não foi treinado. Chame .train() primeiro.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline":  self.pipeline,
            "features":  self._features,
            "trained":   self._trained,
        }
        joblib.dump(payload, filepath)
        logger.info(f"[Classifier] Modelo salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "NEOClassifier":
        """
        Carrega um modelo previamente salvo com .save().

        Parâmetros
        ----------
        filepath : str | Path
            Caminho do arquivo salvo por .save().

        Retorna
        -------
        NEOClassifier com pipeline e metadados restaurados.
        """
        payload  = joblib.load(filepath)
        instance = cls()
        instance.pipeline  = payload["pipeline"]
        instance._features = payload["features"]
        instance._trained  = payload["trained"]
        logger.info(f"[Classifier] Modelo carregado de: {filepath}")
        return instance

    # ------------------------------------------------------------------
    # Heurística interna
    # ------------------------------------------------------------------
    @staticmethod
    def _junk_heuristic(df: pd.DataFrame) -> pd.Series:
        """
        Identifica prováveis lixos espaciais / detritos por regras simples.
        Retorna máscara booleana.
        """
        mask = pd.Series(False, index=df.index)

        if "albedo" in df.columns:
            mask |= df["albedo"] > _JUNK_ALBEDO_MIN

        if "diameter" in df.columns:
            mask |= df["diameter"] < _JUNK_DIAMETER_MAX

        if "magnitude" in df.columns:
            mask |= df["magnitude"] > _JUNK_MAGNITUDE_MIN

        return mask
