Graph-Augmented Semantic Search for Court Decision Retrieval

Projekt leírása
---------------
A projekt célja egy intelligens keresőrendszer fejlesztése, amely bírósági határozatok szemantikai alapú visszakeresését támogatja.
A rendszer transzformer-alapú szövegembeddingeket alkalmaz a bírósági dokumentumok mély jelentésbeli reprezentációjához,
és a dokumentumok közötti kapcsolatok feltérképezésére gráfalapú modellezést használ.
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
├── /reinforcement_learning
│   ├── train_agent.py (tervezett)
│   ├── evaluate_agent.py (tervezett)
│   └── reward_models/
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
pip install pandas openai tqdm pyarrow networkx matplotlib seaborn python-dotenv faiss-cpu backoff torch scikit-learn

(Később opcionálisan HuggingFace TRL is szükséges lehet az RL modulhoz.)

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
   python src/data_loader/graph_builder.py

4. Exploratory Data Analysis (EDA):
   jupyter notebook notebooks/eda_birosagi_hatarozatok.ipynb

5. Kereső prototípus építése:
   jupyter notebook notebooks/keresoproto_terv.ipynb

6. Rangsortanulás (megerősítéses tanulás):
   python src/reinforcement_learning/train_agent.py

7. Agent kiértékelése:
   python src/reinforcement_learning/evaluate_agent.py

Megerősítéses tanulás és rangsortanulási modul
----------------------------------------------
A projekt célja, hogy a keresési találatok rangsorát megerősítéses tanulással optimalizálja.

- A rendszer egy trainelt RL agent segítségével képes lesz a kezdeti embedding-alapú rangsorokat jogi relevancia alapján továbbfinomítani.
- Reward alapú tanulás történik: a jogi szakértők által értékelt találatok alapján.
- Tervezett algoritmus: GRPO (Generalized Reward Policy Optimization).
- Tervezett keretrendszer: saját PyTorch implementáció, vagy HuggingFace TRL.

Struktúra:
- /reinforcement_learning/train_agent.py: agent tanítása.
- /reinforcement_learning/evaluate_agent.py: agent teljesítményének mérése.
- /reinforcement_learning/reward_models/: reward függvények tárolása.

Technikai részletek
-------------------
- Embedding modell: text-embedding-3-small (OpenAI)
- Token limit figyelembevétele (max 8192 token/kérés)
- OpenAI API hívások batch kezelése
- Rate limit és API hibák automatikus kezelése
- Gráf kezelése NetworkX-szel
- Adatok tárolása .parquet és .graphml formátumban
- Reinforcement Learning: PyTorch alapú tanító ciklus
- Reward Model: jogi relevancia scoring függvény

Költségbecslés
--------------
- Python csomagok: 0 USD
- OpenAI API használat: kb. 20–60 USD
- Gráfkezelés (NetworkX): 0 USD
- RL tanítás: helyi gépen (CPU) elegendő
- Összes várható költség: max. 80–100 USD

Jövőbeli fejlesztések
---------------------
- Gráf-alapú keresési bővítések
- Rangsortanulás (agent finomhangolása új reward modellel)
- Kereső prototípus fejlesztése webes felületen (pl. Streamlit)

Kapcsolat
---------
Szakdolgozatkészítő: Dr. Zelenyiánszki Máté
Témavezető: Gombás Veronika, Dr. Kummer Alex
Intézmény: Pannon Egyetem, Mérnöki Kar
