# Szemantikus keresés + Megerősítéses tanulás projekt

Ez a projekt szemantikus keresési képességeket kombinál megerősítéses tanulással (RL) a jogi dokumentumok keresési eredményeinek pontosságának és relevanciájának javítása érdekében. A rendszer először szemantikai hasonlóság alapján lekéri a releváns dokumentumokat, majd egy RL ügynök újrarendezi ezeket a találatokat a felhasználói visszajelzések (vagy szakértői értékelések) alapján tanult stratégia szerint.

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
    Ez telepíti az összes függőséget, beleértve a `transformers`, `torch`, `faiss-cpu`, `gym`, `pandas`, `numpy`, `tqdm` és egyéb szükséges könyvtárakat.

5.  **(Opcionális) Neo4j telepítése és konfigurálása:** Ha a gráf adatbázis funkciókat is használni szeretné (pl. `scripts/populate_graph.py`), telepítenie és futtatnia kell egy Neo4j adatbázist.
    *   Töltse le és telepítse a Neo4j Desktop-ot vagy a Neo4j Servert.
    *   Indítsa el az adatbázist.
    *   Frissítse a kapcsolati adatokat (URI, felhasználó, jelszó) a `configs/config.py` fájlban (`GRAPH_DB_URI`, `GRAPH_DB_USER`, `GRAPH_DB_PASSWORD`).

## Projekt felépítése

- `src/`: Forráskódok
  - `models/`: Gépi tanulási modellek
  - `search/`: Szemantikus keresés implementációja
  - `rl/`: Megerősítéses tanulás implementációja
- `data/`: Adatfájlok könyvtára
  - `raw/`: Nyers adatok
  - `processed/`: Feldolgozott adatok
- `notebooks/`: Jupyter notebookok kísérletekhez
- `tests/`: Egységtesztek
- `configs/`: Konfigurációs fájlok
- `requirements.txt`: Python függőségek
- `setup.sh`: Telepítő script

## Használat

Futtassa a fő szkriptet a szemantikus keresés és az RL környezet teszteléséhez:

```bash
python src/main.py
```

Paraméterek:
```bash
python src/main.py --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" --top_k 5 --query "Jogellenes felmondás munkahelyen"
```

## Valós adatok hozzáadása

Cserélje le a példa adatokat a `src/main.py` fájlban valós jogi dokumentumokkal a `documents` lista módosításával, vagy készítsen betöltő függvényt a `data/raw` könyvtárból.