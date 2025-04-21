import pandas as pd
import json, os, re, logging, csv
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from docx import Document
import unidecode
import spacy
from tqdm import tqdm               # ← progress bar

# ------------------------------------------------------------------
# Konfiguráció
# TODO: Fontolja meg ezen elérési utak kiemelését a configs/config.py fájlba
#       vagy parancssori argumentumként való átadását.
# ------------------------------------------------------------------
DATA_DIR = Path('/Users/zelenyianszkimate/Downloads/BHGY-k') # Bemeneti adatok könyvtára
OUT_PATH = Path(
    '/Users/zelenyianszkimate/Downloads/Feldolgozott BHGY-k/feldolgozott_hatarozatok.csv.gz' # Kimeneti CSV fájl
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        return EXT_HANDLERS[path.suffix.lower()](path).strip()
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
# Fájl‑bejárás és adatgyűjtés
# ------------------------------------------------------------------
if not DATA_DIR.is_dir():
    logging.error(f"A megadott adatkönyvtár nem létezik: {DATA_DIR}")
    raise SystemExit(f"Hiba: A(z) '{DATA_DIR}' könyvtár nem található.")

records = []
logging.info(f"Fájlok keresése és feldolgozása a(z) '{DATA_DIR}' könyvtárban...")
for root, _, files in tqdm(list(os.walk(DATA_DIR)), desc="Mappák bejárása"):
    json_files = [f for f in files if f.lower().endswith('.json')]
    for jf in json_files:
        base = jf.split('.')[0] # Fájlnév kiterjesztés nélkül
        root_p = Path(root)
        json_path = root_p / jf

        # Megkeresi a páros szövegfájlt (.docx vagy .rtf)
        text = None
        for ext in ('.docx', '.DOCX', '.RTF', '.rtf'):
            txt_path = root_p / f"{base}{ext}"
            if txt_path.exists():
                text = extract_text(txt_path)
                break
        else:
            # logging.debug(f"Nem található szöveges pár a(z) '{json_path}' fájlhoz.")
            continue # Nincs szöveges pár ehhez a JSON-hoz

        if not text:
            logging.warning(f"Nem sikerült szöveget kinyerni a(z) '{txt_path}' fájlból.")
            continue

        # Betölti a JSON metaadatokat
        md = load_json(json_path)
        if md is None:
            logging.warning(f"Nem sikerült betölteni a metaadatokat a(z) '{json_path}' fájlból.")
            continue

        # Összeállítja a rekordot
        rec = md.copy()
        relative_path_parts = Path(root).relative_to(DATA_DIR).parts
        rec.update({
            'text': text,
            'doc_id': base, # Dokumentum azonosító a fájlnévből
            'metadata_json': json.dumps(md, ensure_ascii=False), # Eredeti metaadatok JSON stringként
            'birosag': relative_path_parts[0] if len(relative_path_parts) > 0 else None,
            'jogterulet': relative_path_parts[1] if len(relative_path_parts) > 1 else None,
            'hatarozat_id_mappa': relative_path_parts[2] if len(relative_path_parts) > 2 else None,
        })
        records.append(rec)

logging.info(f"Összesen {len(records)} párosított dokumentum található.")

# ------------------------------------------------------------------
# DataFrame létrehozása és NLP feldolgozás
# ------------------------------------------------------------------
if not records:
    logging.warning("Nem található feldolgozható dokumentumpár. A szkript leáll.")
    raise SystemExit("Nincsenek feldolgozható adatok.")

# ------------------------------------------------------------------
# spaCy pipeline betöltése – huspacy transformer (hu_core_news_lg)
# Győződjön meg róla, hogy a modell telepítve van: python -m spacy download hu_core_news_lg
# ------------------------------------------------------------------
NLP_MODEL = "hu_core_news_lg" # Változtassa meg a kívánt modellre
try:
    logging.info(f"spaCy modell betöltése: {NLP_MODEL}...")
    nlp = spacy.load(NLP_MODEL)
    logging.info("spaCy modell sikeresen betöltve.")
except IOError:
    logging.error(f"Hiba: A(z) '{NLP_MODEL}' spaCy modell nincs telepítve.")
    logging.error(f"Kérlek, telepítsd a következő paranccsal: python -m spacy download {NLP_MODEL}")
    raise SystemExit(f"Hiba: spaCy modell '{NLP_MODEL}' nem található.")

# Jogi specifikus stop szavak hozzáadása
LEGAL_STOP = {"felperes","alperes","eljárás","ítélet","kérelmező",
              "kérelmezett","határozat","indítvány","kelt","fentiek"}
nlp.Defaults.stop_words |= LEGAL_STOP
logging.info(f"Jogi stop szavak hozzáadva: {LEGAL_STOP}")

# Reguláris kifejezés jogszabályhelyek normalizálásához (pl. Ptk. 123. § (1) -> PTK_123_1)
LAW_PAT = re.compile(r'\b([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]{1,4})\.\s*(\d{1,3})\.\s*§\s*(\(\d+\))?')
def norm_cite(t:str):
    """Normalizálja a jogszabályhelyeket a szövegben."""
    def repl(m):
        art, para = m.group(1).upper(), m.group(2)
        bek = (m.group(3) or '').strip('()')
        return f"{art}_{para}_{bek}" if bek else f"{art}_{para}"
    return LAW_PAT.sub(repl, t)

logging.info("DataFrame létrehozása...")
df = pd.DataFrame(records)
logging.info("Jogszabályhelyek normalizálása a szövegben...")
df["text"] = df["text"].apply(norm_cite)

def process_texts(series, batch=128, n_proc=1): # Alapértelmezett processzor szám 1-re állítva
    """
    Feldolgozza a szövegeket a spaCy pipeline segítségével:
    lemmatizál (stop szavak, írásjelek, szóközök nélkül) és kinyeri az entitásokat.
    Használja az nlp.pipe funkciót a hatékony kötegelt feldolgozáshoz és párhuzamosításhoz.
    """
    lemmatized_texts = []
    entities_json = []
    # Figyelmeztetés hozzáadása, ha n_proc > 1
    if n_proc > 1:
        logging.warning(f"A spaCy párhuzamos feldolgozása ({n_proc} processz) jelentős memóriát igényelhet.")
    elif n_proc == -1:
         logging.warning("A spaCy párhuzamos feldolgozása (n_process=-1) az összes elérhető CPU magot használja, ami jelentős memóriát igényelhet.")

    logging.info(f"Szövegek feldolgozása (lemmatizálás + entitáskinyerés) indítása {len(series)} szövegen (batch méret: {batch}, n_process={n_proc})...")
    # n_process értékét a függvény argumentumából vesszük
    for doc in tqdm(nlp.pipe(series.astype(str).tolist(), batch_size=batch, n_process=n_proc), total=len(series), desc="Szövegfeldolgozás"):
        # Lemmatizálás
        lemmas = " ".join(t.lemma_.lower()
                          for t in doc
                          if not t.is_stop and not t.is_punct and not t.is_space)
        lemmatized_texts.append(lemmas)

        # Entitáskinyerés
        try:
            ents = json.dumps([(e.text, e.label_) for e in doc.ents], ensure_ascii=False)
            entities_json.append(ents)
        except Exception as e:
            logging.error(f"Hiba az entitások kinyerésekor egy dokumentumban: {e}")
            entities_json.append(json.dumps([], ensure_ascii=False)) # Hiba esetén üres lista

    logging.info("Szövegek feldolgozása befejezve.")
    return lemmatized_texts, entities_json

# Szövegek feldolgozása (lemmatizálás és entitáskinyerés egy lépésben)
# Itt adjuk át az n_proc értékét, alapértelmezetten 1 lesz.
# Ha több processzort szeretnél használni (és van elég memóriád), növeld ezt az értéket (pl. n_proc=2 vagy 4).
lemmas, entities = process_texts(df["text"], n_proc=1)
df["lemmatized_text"] = lemmas
df["entities"] = entities

logging.info("Ékezetmentesített lemmák létrehozása...")
df["plain_lemmas"] = df["lemmatized_text"].apply(unidecode.unidecode)

# ------------------------------------------------------------------
# CSV fájl írása – Memóriafigyelés és gzip tömörítés
# ------------------------------------------------------------------
# Kiválasztott oszlopok a kimeneti CSV-hez (IDE HELYEZVE)
cols = ["doc_id","birosag","jogterulet","hatarozat_id_mappa",
        "text","lemmatized_text","plain_lemmas","entities", "metadata_json"]

# Memóriaigény számítása (IDE HELYEZVE)
total_mem = df.memory_usage(deep=True).sum()
logging.info(f"A DataFrame teljes memóriaigénye: {total_mem/1e9:.2f} GB")

# Kimeneti könyvtár létrehozása, ha nem létezik
os.makedirs(OUT_PATH.parent, exist_ok=True)
logging.info(f"CSV fájl írása ide: {OUT_PATH}")

# Memóriahasználat alapján döntés a chunk-olt írásról
# TODO: A memória limitet (15GB) tegye konfigurálhatóvá
MEMORY_LIMIT_GB = 15
if total_mem > MEMORY_LIMIT_GB * 1_000_000_000:
    # TODO: A chunk méretet (50k) tegye konfigurálhatóvá
    CHUNK_SIZE = 50_000
    logging.warning(f"A DataFrame mérete meghaladja a {MEMORY_LIMIT_GB} GB-ot. Chunk-olt CSV írás ({CHUNK_SIZE} soronként)...")
    for i in tqdm(range(0, len(df), CHUNK_SIZE), desc="CSV írás (chunk-olt)"):
        df.iloc[i:i+CHUNK_SIZE].to_csv(
            OUT_PATH,
            columns=cols,
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
            compression="gzip",
            mode="a", # Hozzáfűzés mód
            header=(i==0), # Csak az első chunk kap fejlécet
        )
else:
    logging.info("DataFrame írása egy lépésben...")
    df.to_csv(
        OUT_PATH,
        columns=cols,
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL,
        compression="gzip",
    )

logging.info(f"A feldolgozott adatok sikeresen elmentve ide: {OUT_PATH}")
print("A dokumentum-feldolgozó szkript futása befejeződött.")

# tqdm progress_apply integrációhoz
tqdm.pandas()
