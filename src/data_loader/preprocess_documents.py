import pandas as pd
import json, os, re, logging, csv, gc
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from docx import Document
import unidecode
import spacy
from tqdm import tqdm
import sys
import pyarrow # Hozzáadva a Parquet kezeléshez
import pyarrow.parquet as pq # Hozzáadva a Parquet íráshoz/olvasáshoz

# Calculate the project root directory (two levels up from the script's directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Konfiguráció importálása
from configs import config

# ------------------------------------------------------------------
# Konfiguráció betöltése
# ------------------------------------------------------------------
DATA_DIR = config.DATA_DIR
OUT_DIR = config.OUT_DIR
OUT_FILENAME = config.OUT_FILENAME
OUT_PATH = OUT_DIR / OUT_FILENAME
CACHE_FILENAME = "preprocessed_cache.parquet" # Cache fájl neve
CACHE_FILE_PATH = OUT_DIR / CACHE_FILENAME # Teljes cache fájl útvonal
NLP_MODEL = config.NLP_MODEL
PROCESSING_BATCH_SIZE = config.PROCESSING_BATCH_SIZE
CSV_WRITE_CHUNK_SIZE = config.CSV_WRITE_CHUNK_SIZE
SPACY_N_PROCESS = config.SPACY_N_PROCESS
LEGAL_STOP = config.LEGAL_STOP_WORDS
OUTPUT_COLUMNS = config.OUTPUT_COLUMNS
MEMORY_LIMIT_GB = config.MEMORY_LIMIT_GB

# Ellenőrizzük a kimeneti könyvtárat
os.makedirs(OUT_DIR, exist_ok=True)

# Logging beállítása
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# spaCy modell betöltése
try:
    logging.info(f"spaCy modell betöltése: {NLP_MODEL}...")
    nlp = spacy.load(NLP_MODEL)
    nlp.Defaults.stop_words |= LEGAL_STOP
    logging.info(f"spaCy modell sikeresen betöltve, jogi stop szavak hozzáadva: {LEGAL_STOP}")
except IOError:
    logging.error(f"Hiba: A(z) '{NLP_MODEL}' spaCy modell nincs telepítve.")
    logging.error(f"Kérlek, telepítsd a következő paranccsal: python -m spacy download {NLP_MODEL}")
    raise SystemExit(f"Hiba: spaCy modell '{NLP_MODEL}' nem található.")
except Exception as e:
    logging.error(f"Váratlan hiba a spaCy modell betöltésekor: {e}")
    raise SystemExit("Hiba a spaCy modell betöltésekor.")

# ------------------------------------------------------------------
# Segédfüggvények
# ------------------------------------------------------------------
EXT_HANDLERS = {
    '.docx': lambda p: re.sub(r'\s+', ' ', '\n'.join(para.text for para in Document(p).paragraphs)),
    '.rtf':  lambda p: re.sub(r'\s+', ' ', rtf_to_text(open(p,'r',encoding='utf-8',errors='ignore').read())),
}

def extract_text(path: Path):
    """Kinyeri a szöveget a támogatott fájltípusokból."""
    try:
        handler = EXT_HANDLERS.get(path.suffix.lower())
        if handler:
            return handler(path).strip()
        else:
            return None
    except Exception as e:
        logging.error(f"extract_text_error file={path} err={e}")
        return None

def load_json(path: Path):
    """Betölti a JSON adatokat egy fájlból."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"json_load_error file={path} err={e}")
        return None

# ------------------------------------------------------------------
# Fájl‑bejárás és adatgyűjtés VAGY Cache betöltés
# ------------------------------------------------------------------

if CACHE_FILE_PATH.exists():
    logging.info(f"Gyorsítótár fájl ({CACHE_FILENAME}) található. Betöltés...")
    try:
        df = pq.read_table(CACHE_FILE_PATH).to_pandas()
        logging.info(f"DataFrame sikeresen betöltve a gyorsítótárból ({len(df)} sor). Fájlbejárás és norm_cite kihagyva.")
        # Győződjünk meg róla, hogy a szükséges oszlopok léteznek
        if 'text' not in df.columns:
             logging.error("A gyorsítótárazott DataFrame-ből hiányzik a 'text' oszlop.")
             raise ValueError("Hiányzó 'text' oszlop a cache-ben.")
        cache_loaded = True
    except Exception as e:
        logging.error(f"Hiba a gyorsítótár ({CACHE_FILENAME}) betöltése közben: {e}. Újragenerálás...")
        cache_loaded = False
else:
    logging.info(f"Gyorsítótár fájl ({CACHE_FILENAME}) nem található. Fájlok feldolgozása...")
    cache_loaded = False

if not cache_loaded:
    if not DATA_DIR.is_dir():
        logging.error(f"A megadott adatkönyvtár nem létezik: {DATA_DIR}")
        raise SystemExit(f"Hiba: A(z) '{DATA_DIR}' könyvtár nem található.")

    records = []
    logging.info(f"Fájlok keresése és feldolgozása a(z) '{DATA_DIR}' könyvtárban...")
    for root, _, files in tqdm(list(os.walk(DATA_DIR)), desc="Mappák bejárása"):
        json_files = [f for f in files if f.lower().endswith('.json')]
        for jf in json_files:
            base = jf.split('.')[0]
            root_p = Path(root)
            json_path = root_p / jf

            text = None
            found_text_path = None
            for ext in ('.docx', '.DOCX', '.rtf', '.RTF'):
                txt_path = root_p / f"{base}{ext}"
                if txt_path.exists():
                    text = extract_text(txt_path)
                    found_text_path = txt_path
                    break

            if not text:
                continue

            md = load_json(json_path)
            if md is None:
                logging.warning(f"Nem sikerült betölteni a metaadatokat a(z) '{json_path}' fájlból.")
                continue

            rec = md.copy()
            try:
                relative_path_parts = root_p.relative_to(DATA_DIR).parts
            except ValueError:
                logging.warning(f"Hiba a relatív útvonal meghatározásakor: {root_p}")
                relative_path_parts = []

            rec.update({
                'text': text,
                'doc_id': base,
                'metadata_json': json.dumps(md, ensure_ascii=False),
                'birosag': relative_path_parts[0] if len(relative_path_parts) > 0 else None,
                'jogterulet': relative_path_parts[1] if len(relative_path_parts) > 1 else None,
                'hatarozat_id_mappa': relative_path_parts[2] if len(relative_path_parts) > 2 else None,
            })
            records.append(rec)

    logging.info(f"Összesen {len(records)} párosított dokumentum található.")

    # ------------------------------------------------------------------
    # DataFrame létrehozása és norm_cite alkalmazása (csak ha nincs cache)
    # ------------------------------------------------------------------
    if not records:
        logging.warning("Nem található feldolgozható dokumentumpár. A szkript leáll.")
        raise SystemExit("Nincsenek feldolgozható adatok.")

    logging.info("DataFrame létrehozása...")
    df = pd.DataFrame(records)
    logging.info(f"DataFrame létrehozva {len(df)} sorral.")

    logging.info("A 'records' lista törlése a memória felszabadítása érdekében...")
    del records
    gc.collect()

    LAW_PAT = re.compile(r'\b([A-ZÁÉÍÓÖŐÚÜŰa-záéíóöőúüű]{1,5})\.?\s*(\d{1,4})\.?\s*§\s*(\(\s*\d+\s*\))?')
    def norm_cite(t:str):
        """Normalizálja a jogszabályhelyeket a szövegben."""
        if not isinstance(t, str): return t
        def repl(m):
            law_abbr = m.group(1).upper().replace('.', '')
            section = m.group(2)
            paragraph_group = m.group(3)
            paragraph = ''
            if paragraph_group:
                paragraph = re.sub(r'\D', '', paragraph_group)

            if paragraph:
                return f"{law_abbr}_{section}_{paragraph}"
            else:
                return f"{law_abbr}_{section}"
        try:
            return LAW_PAT.sub(repl, t)
        except Exception as e:
            logging.warning(f"Hiba a jogszabályhely normalizálása közben: {e} Szöveg: {t[:100]}...")
            return t

    logging.info("Jogszabályhelyek normalizálása a szövegben...")
    tqdm.pandas(desc="Jogszabályhelyek normalizálása")
    df["text"] = df["text"].progress_apply(norm_cite)

    # Gyorsítótár mentése
    try:
        logging.info(f"DataFrame mentése a gyorsítótárba: {CACHE_FILE_PATH}")
        table = pyarrow.Table.from_pandas(df)
        pq.write_table(table, CACHE_FILE_PATH)
        logging.info("DataFrame sikeresen mentve a gyorsítótárba.")
        del table # Memória felszabadítása
        gc.collect()
    except Exception as e:
        logging.error(f"Hiba a DataFrame gyorsítótárba mentése közben ({CACHE_FILE_PATH}): {e}")

# ------------------------------------------------------------------
# NLP feldolgozás (mindig lefut, cache-ből vagy friss adatokból)
# ------------------------------------------------------------------

def process_texts(series, batch=PROCESSING_BATCH_SIZE, n_proc=SPACY_N_PROCESS):
    """
    Feldolgozza a szövegeket a spaCy pipeline segítségével:
    lemmatizál (stop szavak, írásjelek, szóközök nélkül) és kinyeri az entitásokat.
    """
    lemmatized_texts = []
    entities_json = []
    if n_proc > 1:
        logging.warning(f"A spaCy párhuzamos feldolgozása ({n_proc} processz) jelentős memóriát igényelhet. Figyelje a rendszer erőforrásait.")
    elif n_proc == -1:
         logging.warning("A spaCy párhuzamos feldolgozása (n_process=-1) az összes elérhető CPU magot használja, ami jelentős memóriát igényelhet. Figyelje a rendszer erőforrásait.")
    elif n_proc < -1:
        logging.error(f"Érvénytelen n_process érték: {n_proc}. Alapértelmezett (1) használata.")
        n_proc = 1

    logging.info(f"Szövegek feldolgozása (lemmatizálás + entitáskinyerés) indítása {len(series)} szövegen (batch méret: {batch}, n_process={n_proc})...")
    processed_docs = 0
    error_count = 0
    try:
        text_list = series.fillna('').astype(str).tolist()
        for doc in tqdm(nlp.pipe(text_list, batch_size=batch, n_process=n_proc), total=len(text_list), desc="Szövegfeldolgozás"):
            lemmas = " ".join(t.lemma_.lower()
                              for t in doc
                              if not t.is_stop and not t.is_punct and not t.is_space and t.lemma_)
            lemmatized_texts.append(lemmas)

            if "ner" not in nlp.disabled:
                try:
                    ents = json.dumps([(e.text, e.label_) for e in doc.ents], ensure_ascii=False)
                    entities_json.append(ents)
                except Exception as e:
                    logging.error(f"Hiba az entitások kinyerésekor egy dokumentumban: {e}")
                    entities_json.append(json.dumps([], ensure_ascii=False))
                    error_count += 1
            else:
                 entities_json.append(json.dumps([], ensure_ascii=False))

            processed_docs += 1

    except Exception as e:
        logging.error(f"Váratlan hiba a spaCy feldolgozás (nlp.pipe) során: {e}")
        remaining = len(series) - processed_docs
        lemmatized_texts.extend([''] * remaining)
        entities_json.extend([json.dumps([], ensure_ascii=False)] * remaining)
        error_count += remaining

    logging.info(f"Szövegek feldolgozása befejezve. Feldolgozott: {processed_docs}, Hibás entitáskinyerés: {error_count}")
    if len(lemmatized_texts) != len(series):
        logging.warning(f"Lemmatizált szövegek listájának hossza ({len(lemmatized_texts)}) nem egyezik az eredeti sorozat hosszával ({len(series)}). Korrigálás...")
        lemmatized_texts.extend([''] * (len(series) - len(lemmatized_texts)))
    if len(entities_json) != len(series):
         logging.warning(f"Entitások listájának hossza ({len(entities_json)}) nem egyezik az eredeti sorozat hosszával ({len(series)}). Korrigálás...")
         entities_json.extend([json.dumps([], ensure_ascii=False)] * (len(series) - len(entities_json)))

    return lemmatized_texts, entities_json

lemmas, entities = process_texts(df["text"])
df["lemmatized_text"] = lemmas
df["entities"] = entities

logging.info("Lemmas és entities listák törlése...")
del lemmas
del entities
gc.collect()

logging.info("Ékezetmentesített lemmák létrehozása...")
tqdm.pandas(desc="Ékezetmentesítés")
df["plain_lemmas"] = df["lemmatized_text"].fillna('').astype(str).progress_apply(unidecode.unidecode)

# ------------------------------------------------------------------
# CSV fájl írása – Memóriafigyelés és gzip tömörítés
# ------------------------------------------------------------------
try:
    total_mem_bytes = df.memory_usage(deep=True).sum()
    total_mem_gb = total_mem_bytes / 1e9
    logging.info(f"A DataFrame teljes memóriaigénye: {total_mem_gb:.2f} GB")
except Exception as e:
    logging.error(f"Hiba a DataFrame memóriaigényének számításakor: {e}")
    total_mem_bytes = 0

logging.info(f"CSV fájl írása ide: {OUT_PATH}")

if total_mem_bytes > MEMORY_LIMIT_GB * 1_000_000_000 and CSV_WRITE_CHUNK_SIZE > 0:
    logging.warning(f"A DataFrame mérete ({total_mem_gb:.2f} GB) meghaladja a beállított limitet ({MEMORY_LIMIT_GB} GB). Chunk-olt CSV írás ({CSV_WRITE_CHUNK_SIZE} soronként)...")
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
        logging.info(f"Meglévő fájl törölve: {OUT_PATH}")

    for i in tqdm(range(0, len(df), CSV_WRITE_CHUNK_SIZE), desc="CSV írás (chunk-olt)"):
        chunk = df.iloc[i:i+CSV_WRITE_CHUNK_SIZE]
        chunk.to_csv(
            OUT_PATH,
            columns=OUTPUT_COLUMNS,
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
            mode="a",
            header=(i==0),
        )
    del chunk
    gc.collect()
else:
    logging.info(f"DataFrame írása egy lépésben ({OUT_PATH})...")
    df.to_csv(
        OUT_PATH,
        columns=OUTPUT_COLUMNS,
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL,
    )

logging.info(f"A feldolgozott adatok sikeresen elmentve ide: {OUT_PATH}")
print("A dokumentum-feldolgozó szkript futása befejeződött.")