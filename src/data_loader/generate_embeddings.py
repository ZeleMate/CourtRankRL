# src/data_loader/generate_embeddings.py
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys
import os

# Project root hozzáadása a Python path-hoz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs import config
from src.models.embedding import QwenEmbeddingModel

logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

def load_csv_data(file_path: str) -> pd.DataFrame:
    """CSV adatok betöltése."""
    logging.info(f"CSV adatok betöltése: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding=config.CSV_ENCODING)
        logging.info(f"Betöltve: {len(df):,} sor")
        return df
    except Exception as e:
        logging.error(f"Hiba a CSV betöltésekor: {e}")
        raise

def generate_embeddings(texts: list) -> np.ndarray:
    """Embedding-ek generálása Qwen3-8B-vel."""
    logging.info(f"Embedding generálás {len(texts):,} szöveghez")
    
    try:
        # Modell inicializálás
        model = QwenEmbeddingModel()
        
        # Embedding generálás
        embeddings = model.encode(texts)
        
        logging.info(f"Embedding generálás sikeres: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logging.error(f"Hiba az embedding generáláskor: {e}")
        raise

def save_parquet_data(df: pd.DataFrame, output_path: str):
    """Feldolgozott adatok mentése Parquet formátumban."""
    logging.info(f"Parquet mentés: {output_path}")
    
    try:
        # Kimeneti könyvtár létrehozása
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Parquet mentés
        df.to_parquet(
            output_path,
            engine=config.PARQUET_ENGINE,
            index=config.PARQUET_INDEX
        )
        
        logging.info(f"Parquet fájl sikeresen mentve: {output_path}")
        
    except Exception as e:
        logging.error(f"Hiba a Parquet mentéskor: {e}")
        raise

def main():
    """Fő embedding generálási folyamat."""
    try:
        # 1. CSV adatok betöltése
        df = load_csv_data(config.CLEANED_CSV_DATA_PATH)
        
        if df.empty:
            logging.error("Nincs adat feldolgozásra")
            return
        
        # 2. Szövegek kinyerése
        if 'text' not in df.columns:
            logging.error("Nincs 'text' oszlop a CSV-ben")
            return
        
        texts = df['text'].fillna('').astype(str).tolist()
        logging.info(f"Feldolgozandó szövegek: {len(texts):,}")
        
        # 3. Embedding generálás
        embeddings = generate_embeddings(texts)
        
        # 4. Eredmények hozzáadása DataFrame-hez
        df['embedding'] = embeddings.tolist()
        
        # 5. Metadata JSON készítése (ha szükséges)
        if 'metadata_json' not in df.columns:
            metadata_list = []
            for _, row in df.iterrows():
                metadata = {
                    'doc_id': row.get('doc_id', ''),
                    'birosag': row.get('birosag', ''),
                    'jogterulet': row.get('jogterulet', ''),
                    'hatarozat_id_mappa': row.get('hatarozat_id_mappa', '')
                }
                metadata_list.append(json.dumps(metadata, ensure_ascii=False))
            
            df['metadata_json'] = metadata_list
        
        # 6. Csak szükséges oszlopok megtartása
        if config.EMBEDDING_OUTPUT_COLUMNS:
            available_columns = [col for col in config.EMBEDDING_OUTPUT_COLUMNS if col in df.columns]
            df = df[available_columns]
            logging.info(f"Megtartott oszlopok: {available_columns}")
        
        # 7. Parquet fájl mentése
        save_parquet_data(df, config.PROCESSED_PARQUET_DATA_PATH)
        
        logging.info("Embedding generálás befejezve!")
        
    except Exception as e:
        logging.error(f"Hiba a főfolyamatban: {e}")
        raise

if __name__ == "__main__":
    main()