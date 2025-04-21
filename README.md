# Szemantikus keresés + Megerősítéses tanulás projekt

Ez a projekt szemantikus keresési képességeket kombinál megerősítéses tanulással (RL) a jogi dokumentumok keresési eredményeinek pontosságának és relevanciájának javítása érdekében. A rendszer először szemantikai hasonlóság alapján lekéri a releváns dokumentumokat (jelölteket), majd egy RL ügynök újrarendezi ezeket a találatokat egy tanult stratégia szerint, amely szakértői értékeléseken alapul. A projekt tartalmaz egy Flask alapú webes felületet is az interaktív kereséshez.

## Telepítés

1.  **Klónozza a tárolót (ha még nem tette meg):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Hozzon létre egy virtuális környezetet:** Ez elszigeteli a projekt függőségeit a globális Python telepítéstől.
    ```bash
    python -m venv venv
    ```

3.  **Aktiválja a virtuális környezetet:**
    *   Windows (Command Prompt/PowerShell): `venv\Scripts\activate`
    *   macOS/Linux (Bash/Zsh): `source venv/bin/activate`
    A parancssor elején meg kell jelennie a `(venv)` szövegnek.

4.  **Telepítse a szükséges Python csomagokat:**
    ```bash
    pip install -r requirements.txt
    ```
    Ez telepíti az összes függőséget, beleértve a `flask`, `torch`, `faiss-cpu`, `gym`, `pandas`, `numpy`, `tqdm`, `openai`, `python-docx`, `striprtf` és egyéb szükséges könyvtárakat.

5.  **Konfigurálja az OpenAI API kulcsot:** A projekt alapértelmezés szerint az OpenAI beágyazási modelljét használja (`text-embedding-3-large`).
    *   Állítsa be az `OPENAI_API_KEY` környezeti változót az OpenAI API kulcsával.
    *   Alternatívaként közvetlenül beírhatja a kulcsot a `configs/config.py` fájlba (ez **nem ajánlott** biztonsági okokból).

6.  **(Opcionális) Neo4j telepítése és konfigurálása:** Ha a gráf adatbázis funkciókat is használni szeretné (`scripts/populate_graph.py`), telepítenie és futtatnia kell egy Neo4j adatbázist.
    *   Töltse le és telepítse a Neo4j Desktop-ot vagy a Neo4j Servert.
    *   Indítsa el az adatbázist.
    *   Frissítse a kapcsolati adatokat (URI, felhasználó, jelszó) a `configs/config.py` fájlban (`GRAPH_DB_URI`, `GRAPH_DB_USER`, `GRAPH_DB_PASSWORD`).

## Projekt felépítése

- `src/`: A projekt fő forráskódja.
  - `app.py`: Flask webalkalmazás az interaktív kereséshez.
  - `main.py`: Parancssori interfész kereséshez és kiértékeléshez.
  - `data_loader/`: Modulok dokumentumok betöltéséhez különböző formátumokból.
    - `legal_docs.py`: Dokumentumok betöltése mappából (.txt, .docx, .rtf, .json, .csv stb.).
  - `models/`: Beágyazási modellek betöltése.
    - `embedding.py`: Modell betöltő logika (jelenleg OpenAI-ra konfigurálva).
  - `search/`: Szemantikus keresés implementációja.
    - `semantic_search.py`: FAISS index használata a jelöltek lekéréséhez.
  - `rl/`: Megerősítéses tanulás komponensek (újrarendezés).
    - `agent.py`: RL ügynök (Policy Network) implementációja és betöltése/mentése.
    - `environment.py`: Gym környezet a rangsorolási feladathoz.
    - `reward.py`: Jutalom számítása szakértői értékelések alapján (pl. NDCG).
  - `graph/`: Gráf adatbázis komponensek (opcionális).
    - `extractor.py`: Entitások és kapcsolatok kinyerése szövegből (implementáció szükséges).
    - `graph_db.py`: Kapcsolat a Neo4j adatbázishoz.
- `scripts/`: Segédszkriptek.
  - `populate_graph.py`: Szkript a gráf adatbázis feltöltéséhez dokumentumokból (opcionális).
  - `train_rl_agent.py`: Szkript az RL ügynök tanításához szakértői értékelésekkel.
  - `preprocess_documents.py`: Szkript jogi dokumentumok (.docx, .rtf) és a hozzájuk tartozó JSON metaadatok előfeldolgozására, szövegkinyerésre, lemmatizálásra, entitásfelismerésre és az eredmények CSV fájlba mentésére.
- `configs/`: Konfigurációs fájlok.
  - `config.py`: Központi konfiguráció (modellnevek, API kulcsok, útvonalak, hiperparaméterek).
- `templates/`: HTML sablonok a Flask alkalmazáshoz.
  - `index.html`: A keresőoldal sablonja.
- `requirements.txt`: Python függőségek listája.
- `README.md`: Ez a fájl.

## Használat

A projekt többféleképpen használható:

1.  **Dokumentumok Előfeldolgozása:**
    Futtassa a `scripts/preprocess_documents.py` szkriptet a nyers dokumentumok feldolgozásához. Győződjön meg róla, hogy a `DATA_DIR` és `OUT_PATH` változók helyesen vannak beállítva a szkriptben, vagy fontolja meg azok konfigurációs fájlba helyezését vagy parancssori argumentumként való átadását. Telepítenie kell a `hu_core_news_trf` spaCy modellt is (`python -m spacy download hu_core_news_trf`).
    ```bash
    python scripts/preprocess_documents.py
    ```
    Ez a szkript bejárja a `DATA_DIR` könyvtárat, párosítja a `.docx`/`.rtf` fájlokat a `.json` metaadat fájlokkal, kinyeri és normalizálja a szöveget, lemmatizálja, kinyeri az entitásokat, majd az eredményeket egy tömörített CSV fájlba (`OUT_PATH`) menti.

2.  **Interaktív keresés (Web UI):**
    Indítsa el a Flask alkalmazást:
    ```bash
    python src/app.py
    ```
    Nyissa meg a böngészőjében a `http://127.0.0.1:5001` (vagy a terminálban megjelenő címet). Írja be a keresési lekérdezést, és a rendszer visszaadja a szemantikailag releváns és RL által újrarendezett találatokat.

3.  **Parancssori keresés:**
    Használja a `src/main.py` szkriptet `search` módban:
    ```bash
    python src/main.py search --query "az ön keresési lekérdezése itt" --top_k 5
    ```
    További opciók (pl. modell, dokumentum elérési út) megtekintéséhez használja a `--help` kapcsolót:
    ```bash
    python src/main.py search --help
    ```

4.  **Kiértékelés:**
    Futtassa a rendszert `evaluate` módban, hogy összehasonlítsa a kezdeti szemantikus keresés és az RL újrarendezés teljesítményét (NDCG alapján) a szakértői értékelésekkel (`EXPERT_EVAL_PATH` a `config.py`-ban):
    ```bash
    python src/main.py evaluate --top_k 5
    ```

5.  **RL Ügynök Tanítása:**
    Használja a `scripts/train_rl_agent.py` szkriptet az RL ügynök tanításához. Szüksége lesz egy CSV fájlra (`EXPERT_EVAL_PATH`), amely tartalmazza a szakértői értékeléseket (pl. `query`, `doc_id`, `relevance` oszlopokkal).
    ```bash
    python scripts/train_rl_agent.py --iterations 1000 --batch_size 32
    ```
    A tanított ügynök modellje a `RL_AGENT_SAVE_PATH` helyre mentődik (`config.py`).

6.  **(Opcionális) Gráf Adatbázis Feltöltése:**
    Ha konfigurálta a Neo4j-t, futtassa a `scripts/populate_graph.py` szkriptet a dokumentumokból kinyert entitások és kapcsolatok adatbázisba töltéséhez. (Megjegyzés: Az `extractor.py`-nak implementálnia kell a tényleges NLP alapú kinyerési logikát.)
    ```bash
    python scripts/populate_graph.py --doc_path /eleresi/ut/a/dokumentumokhoz
    ```

## Konfiguráció

A fő konfigurációs beállítások a `configs/config.py` fájlban találhatók:

- `EMBEDDING_MODEL_NAME`: A használt beágyazási modell neve (pl. OpenAI modell vagy Sentence Transformer).
- `EMBEDDING_DIMENSION`: A beágyazási modell dimenziója.
- `OPENAI_API_KEY`: Az OpenAI API kulcsa (ajánlott környezeti változóként beállítani).
- `INITIAL_TOP_K`: A szemantikus keresés által visszaadott jelöltek száma az RL újrarendezéshez.
- `FINAL_TOP_K`: A felhasználónak megjelenített végső találatok száma.
- `GRAPH_DB_URI`, `GRAPH_DB_USER`, `GRAPH_DB_PASSWORD`: Neo4j kapcsolati adatok (opcionális).
- `RL_ALGORITHM`: A használt RL algoritmus (jelenleg "GRPO" vagy "PolicyGradient" támogatott, de implementációra szorulnak).
- `POLICY_NETWORK_PARAMS`: Az RL policy hálózat paraméterei (input/output dimenzió, rejtett réteg mérete).
- `LEARNING_RATE`, `DISCOUNT_FACTOR`, stb.: RL tanítási hiperparaméterek.
- `RAW_DATA_PATH`: Az elérési út a mappához, amely a feldolgozandó jogi dokumentumokat tartalmazza. A `legal_docs.py` innen tölti be a fájlokat.
- `EXPERT_EVAL_PATH`: Az elérési út a szakértői értékeléseket tartalmazó CSV fájlhoz (a kiértékeléshez és tanításhoz szükséges).
- `RL_AGENT_SAVE_PATH`: Az elérési út, ahová a tanított RL ügynök modellje mentésre kerül.

## Adatok

- **Nyers Dokumentumok:** Helyezze a jogi dokumentumokat (támogatott formátumokban: .txt, .docx, .rtf, .json, .csv, .md, .html) a `configs/config.py`-ban megadott `RAW_DATA_PATH` mappába.
- **Szakértői Értékelések:** Hozzon létre egy CSV fájlt (az elérési útját állítsa be `EXPERT_EVAL_PATH`-ként a `config.py`-ban) a következő oszlopokkal (minimum):
    - `query`: A keresési lekérdezés.
    - `doc_id`: A dokumentum azonosítója (meg kell egyeznie a betöltés során generált vagy használt azonosítóval, pl. fájlnév vagy index).
    - `relevance`: A dokumentum relevanciája a lekérdezésre (numerikus érték, pl. 0-tól 3-ig, ahol a magasabb jobb). Ezt használja a rendszer az NDCG számításához és a jutalom meghatározásához a tanítás során.