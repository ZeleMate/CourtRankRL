# CourtRankRL – Projekt Összefoglaló és Technikai Dokumentáció

Ez a dokumentum a **CourtRankRL** projekt részletes technikai összefoglalója, amely alapul szolgálhat a szakdolgozat megírásához. A projekt célja egy magyar nyelvű bírósági határozatokra optimalizált, hibrid visszakeresési és megerősítéses tanulással (RL) támogatott rangsoroló rendszer fejlesztése.

## 1. Projekt Célkitűzés és Áttekintés

A CourtRankRL egy olyan informatikai rendszer, amely képes nagy mennyiségű, strukturálatlan jogi dokumentum (bírósági határozatok) hatékony keresésére és relevanciapontos rangsorolására. A rendszer különlegessége, hogy a hagyományos kulcsszóalapú és szemantikus keresést ötvözi egy modern, nagy nyelvi modellen (LLM) alapuló újra-rangsoroló (reranking) lépéssel, amelyet megerősítéses tanulással (Reinforcement Learning, RL) finomhangoltunk.

**Fő innovációk:**
*   **Hibrid Visszakeresés:** Sparse (BM25) és Dense (Embedding) módszerek fúziója.
*   **RL-alapú Reranking:** Egy 4 milliárd paraméteres nyelvi modell (Qwen3-4B) tanítása GRPO (Group Relative Policy Optimization) algoritmussal, kifejezetten a keresési rangsor javítására.
*   **Chunk-alapú Megközelítés:** A dokumentumok intelligens darabolása (chunking) a pontosabb találatok érdekében.

## 2. Rendszerarchitektúra

## 2. Rendszerarchitektúra

A rendszer architektúrája két fő komponensre bontható a feladatok jellege szerint:

1.  **Adatfeldolgozás és Visszakeresés:**
    *   Adatfeldolgozás (ETL pipeline).
    *   Indexépítés.
    *   Hibrid visszakeresés (Inference).

2.  **RL Modell Tanítása (Cloud GPU):**
    *   A nagy számítási igényű megerősítéses tanulás (GRPO) futtatása felhő alapú környezetben, NVIDIA A100 GPU gyorsítással.

## 3. Adatfeldolgozási Folyamat (Data Pipeline)

Az adatfeldolgozás célja a nyers DOCX fájlok átalakítása kereshető, strukturált formátummá.

### 3.1. Dokumentum Feldolgozás
*   **Bemenet:** Magyar bírósági határozatok DOCX formátumban.
*   **Normalizálás:** Unicode karakterek tisztítása, felesleges whitespace eltávolítása.
*   **Metaadat Kinyerés:**
    *   Mintaillesztés alapú kinyerés a fájlútvonalból és a szövegből.
    *   Kinyert adatok: Bíróság neve, Ügyszám (Case ID), Év, Jogterület (Domain).

### 3.2. Chunking (Darabolás)
*   **Logika:** A dokumentumokat logikai egységekre (bekezdések, szakaszok) bontja, nem mechanikusan fix karakterszámra. Ez biztosítja, hogy a szemantikai kontextus ne sérüljön.
*   **Kimenet:** Strukturált adatállomány, amely tartalmazza a szövegrészleteket és a hozzájuk tartozó metaadatokat.

## 4. Visszakeresési Rendszer (Retrieval System)

A rendszer egy kétlépcsős visszakeresési stratégiát alkalmaz.

### 4.1. Sparse Retrieval (Kulcsszavas Keresés)
*   **Módszer:** BM25 (Best Matching 25).
*   **Jellemzők:** Hatékonyan találja meg a pontos kulcsszó-egyezéseket (pl. ügyszámok, jogszabályhelyek).
*   **Tokenizálás:** Magyar nyelvre optimalizált tokenizálás.

### 4.2. Dense Retrieval (Szemantikus Keresés)
*   **Modell:** `google/embeddinggemma-300m`.
*   **Működés:** A lekérdezést és a dokumentumokat vektorrá alakítja, és a vektorok közötti távolság alapján keres. Képes megtalálni a szinonimákat és a fogalmi egyezéseket, még ha a pontos szavak nem is egyeznek.

### 4.3. Hibrid Fúzió
*   **Algoritmus:** Reciprocal Rank Fusion (RRF).
*   **Cél:** A két módszer (BM25 és vektoros keresés) eredményeinek egyesítése úgy, hogy a mindkét listában előkelő helyen szereplő dokumentumok kerüljenek előre. Ez robusztusabb eredményt ad, mint bármelyik módszer önmagában.

## 5. Megerősítéses Tanulás (RL Reranking)

Ez a projekt leginnovatívabb része. A hibrid keresés eredményeit egy LLM rendezi újra, amelyet kifejezetten erre a feladatra tanítottunk.

### 5.1. Modell és Tanítás
*   **Alapmodell:** `Qwen/Qwen3-4B-Instruct`.
*   **Tanítási Módszer:** GRPO (Group Relative Policy Optimization).
*   **Technika:** QLoRA (Quantized Low-Rank Adaptation) – csak az adapter súlyokat tanítjuk, az alapmodell fagyasztva van, ami hatékonyabb tanítást tesz lehetővé.

### 5.2. Tanítási Adat (Slate)
*   **Slate:** Egy lekérdezéshez tartozó, a hibrid kereső által visszaadott legjobb 30 dokumentum-chunk listája.
*   **Címkézés:** A dokumentumok relevanciája (0: nem releváns, 1: releváns, 2: nagyon releváns) előre definiált "qrels" (query relevance) fájlok alapján.

### 5.3. Reward Funkció (Jutalmazás)
A modell nem a következő tokent jósolja (mint a hagyományos LLM-ek), hanem a rangsorolási metrikákat optimalizálja. A reward függvény egy összetett, több szempontot mérlegelő formula:

1.  **nDCG@10 (50%):** Normalized Discounted Cumulative Gain – a rangsor minőségét méri, figyelembe véve a pozíciót és a relevancia fokát.
2.  **MRR@5 (35%):** Mean Reciprocal Rank – milyen gyorsan találjuk meg az ELSŐ releváns találatot.
3.  **Recall@20 (15%):** A releváns találatok hány százaléka került be a top 20-ba.

**Kiegészítések:**
*   **Sigmoid Transzformáció:** A reward értékek stabilizálása érdekében.
*   **Tie-break Bonus:** Extra jutalom, ha az MRR javul, még ha az nDCG stagnál is.
*   **Diverzitás Büntetés:** Shannon-entrópia alapú büntetés, ha a találati lista túl homogén (pl. csak egyfajta bíróság határozatait tartalmazza).

### 5.4. Curriculum Learning
A tanítás során először a "könnyebb" példákat (ahol a baseline is viszonylag jó) mutatjuk a modellnek, majd fokozatosan nehezednek a feladatok. Ez segíti a konvergenciát.

## 6. Technológiai Stack

*   **Nyelv:** Python 3.11
*   **Keretrendszerek:**
    *   `Hugging Face Transformers`, `PEFT`, `TRL` (Modellek és tanítás)
    *   `Unsloth` (Gyorsított LLM finomhangolás)
    *   Vektorkeresés és Sparse keresés optimalizált könyvtárai
    *   Dokumentum feldolgozó eszközök
*   **Infrastruktúra:**
    *   Adatfeldolgozás és Keresés: Lokális környezet
    *   RL Tanítás: Cloud GPU (NVIDIA A100)

## 7. Eredmények és Értékelés

A rendszer teljesítményét a baseline (csak hibrid keresés) és az RL-finomhangolt modell összehasonlításával mérjük.

*   **Baseline:** Erős alapvonalat ad, de hajlamos a kulcsszó-egyezésre fókuszálni a szemantikai tartalom helyett.
*   **RL Reranker:** Képes felismerni a finomabb összefüggéseket és a felhasználói szándékot, javítva az nDCG és MRR értékeket. A modell megtanulja, hogy a "családi jogi ügy" keresésre ne csak azokat a dokumentumokat hozza, ahol ez a kifejezés szerepel, hanem azokat is, amelyek tartalmilag relevánsak (pl. válóper, gyermekelhelyezés), még ha a pontos kifejezés hiányzik is.

Ez a dokumentáció összefoglalja a CourtRankRL rendszer működését, a felhasznált technológiákat és a tudományos hátteret, amelyre a szakdolgozat építhető.
