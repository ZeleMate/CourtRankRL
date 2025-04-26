Graph-Augmented Semantic Search for Court Decision Retrieval

Projekt leírása
---------------
Ez a projekt célja egy intelligens keresőrendszer fejlesztése, amely bírósági határozatok szemantikai alapú visszakeresését támogatja.
A rendszer transzformer-alapú szövegembeddingeket alkalmaz a bírósági dokumentumok mély jelentésbeli reprezentációjához, és a dokumentumok közötti kapcsolatok feltérképezésére gráfalapú modellezést használ.
A találatok rangsorolását megerősítéses tanulási (Reinforcement Learning) technikával finomítja.

Mappastruktúra
--------------
/Szakdolgozat
├── /configs
│   └── config.py
├── /data_loader
│   ├── preprocess_documents.py
│   ├── generate_embeddings.py
│   └── graph_builder.py (tervezett)
├── /notebooks
│   ├── eda_birosagi_hatarozatok.ipynb
│   └── keresoproto_terv.ipynb
├── /processed_data
│   ├── raw_data_for_eda.csv
│   └── embedded_data.parquet
├── /logs
│   └── run_log.txt
├── /data
│   └── (Eredeti RTF/DOCX és JSON fájlok)
/README.txt

Telepítési követelmények
------------------------
- Python 3.10 vagy újabb
- OpenAI API kulcs szükséges az embeddingek generálásához

Telepítendő csomagok:
pip install pandas openai tqdm pyarrow networkx matplotlib seaborn python-dotenv faiss-cpu backoff

Konfiguráció
------------
- /configs/config.py fájlban adható meg:
  - Bemeneti adatok elérési útja
  - OpenAI API kulcs
  - Embedding modell (text-embedding-3-small)
  - Batch méret
  - Kimeneti fájlok helye
  - Naplózás beállításai

Használati lépések
------------------
1. Nyers adatok feldolgozása:
   python src/data_loader/preprocess_documents.py

2. Embeddingek generálása:
   python src/data_loader/generate_embeddings.py

3. Gráf építése NetworkX segítségével:
   (Később: graph_builder.py)

4. Exploratory Data Analysis (EDA):
   jupyter notebook notebooks/eda_birosagi_hatarozatok.ipynb

Technikai részletek
-------------------
- Embedding modell: text-embedding-3-small (OpenAI)
- Token limit figyelembevétele (max 8192 token/kérés)
- OpenAI API hívások batch kezelése
- Rate limit és API hibák automatikus kezelése
- Gráf kezelése NetworkX-szel
- Adatok tárolása .parquet és .graphml formátumban

Költségbecslés
--------------
- Python csomagok: 0 USD
- OpenAI API használat: kb. 20–60 USD
- Gráfkezelés (NetworkX): 0 USD
- Összes várható költség: max. 80 USD

Jövőbeli fejlesztések
---------------------
- Gráf-alapú keresési bővítések
- Rangsortanulás implementációja (GRPO)
- Kereső prototípus fejlesztése webes felületen

Kapcsolat
---------
Szakdolgozatkészítő: Dr. Zelenyiánszki Máté
Témavezető: Dr. Kummer Alex, Gombás Veronika
Intézmény: Pannon Egyetem, Mérnöki Kar