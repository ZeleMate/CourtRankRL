# Magyar Bírósági Határozatok End-to-End Szemantikus Keresőrendszere Megerősítéses Tanulással

## Szakdolgozat Áttekintés

Ez a projekt egy komplex, end-to-end megoldást mutat be magyarországi bírósági határozatok hatékony szabadszöveges keresésére. A rendszer egy többlépcsős architektúrát implementál, amely kezdetben szemantikus embeddingek alapján végez hasonlóság-alapú keresést, majd megerősítéses tanulással (RL) finomhangolt intelligens ágensek segítségével optimalizálja a végső találati listát a releváns dokumentumok jobb rangsorolása érdekében.

## Kutatási Motiváció

A modern jogi információkeresés egyik legnagyobb kihívása a szabadszöveges lekérdezések hatékony feldolgozása nagy volumenű dokumentumkorpuszokon. A hagyományos kulcsszó-alapú keresési rendszerek gyakran nem képesek megfelelően kezelni a jogi terminológia komplexitását és a kontextuális jelentéseket. Ez a projekt egy innovatív megközelítést alkalmaz, amely ötvözi a modern nyelvmodell-alapú szemantikus keresést a megerősítéses tanulás adaptív optimalizálási képességeivel.

## Rendszer Architektúra

A következő diagram ábrázolja a teljes end-to-end rendszer működését a felhasználói lekérdezéstől a megerősítéses tanulással optimalizált végső eredményig:

```mermaid
graph TD
    A["Felhasználó<br/>Szabadszöveges lekérdezés"] --> B["Query Preprocessing<br/>Szöveg normalizálás<br/>Tokenizálás"]
    
    B --> C["Embedding Generálás<br/>Qwen3-8B modell<br/>Query → 8192D vektor"]
    
    C --> D["FAISS Index Keresés<br/>Approximate NN search<br/>Top-K jelöltek"]
    
    D --> D1["Gráf Alapú Bővítés<br/>Kapcsolódó dokumentumok<br/>Graph traversal"]
    
    D1 --> E["Hibrid Ranking<br/>Szemantikus + Gráf<br/>alapú sorrend"]
    
    E --> F["RL Agent<br/>RankingPolicyNetwork<br/>State: Query+Docs+Graph embedding"]
    
    F --> G["Re-ranking<br/>Pontszámok generálása<br/>Optimalizált sorrend"]
    
    G --> H["Végső Találati Lista<br/>Releváns dokumentumok<br/>rangsorolt listája"]
    
    H --> I["Felhasználó<br/>Eredmények megjelenítése"]
    
    H --> J["Szakértői Értékelés<br/>Relevancia pontszámok<br/>NDCG@5, NDCG@10"]
    
    J --> K["Reward Számítás<br/>NDCG javulás<br/>vs. baseline"]
    
    K --> L["RL Training<br/>GRPO/Policy Gradient<br/>Policy frissítés"]
    
    L --> F
    
    subgraph "Adatfeldolgozási Réteg"
        M["Nyers Dokumentumok<br/>213,398 határozat"]
        N["Preprocessing<br/>Szöveg tisztítás<br/>Metaadat kinyerés"]
        O["Strukturált Adatok<br/>CSV/Parquet formátum"]
        M --> N --> O
    end
    
    subgraph "Embedding Réteg"
        P["Batch Processing<br/>Chunk-alapú feldolgozás"]
        Q["Qwen3-8B Modell<br/>8192 dimenziós embeddings<br/>A100 80GB GPU"]
        R["Vektor Adatbázis<br/>FAISS index tárolás"]
        O --> P --> Q --> R
        R --> D
    end
    
    subgraph "Gráf Réteg"
        V["Dokumentum Kapcsolatok<br/>1.6M csomópont, 4.4M él"]
        W["Hivatkozási Hálózat<br/>Határozatok közötti links"]
        X["Jogszabály Kapcsolatok<br/>Dokumentum-jogszabály hivatkozások"]
        Y["Bírósági Kapcsolatok<br/>Dokumentum-bíróság kapcsolatok"]
        O --> V
        V --> W
        V --> X
        V --> Y
        W --> D1
        X --> D1
        Y --> D1
    end
    
    subgraph "RL Optimalizálás"
        S["Training Environment<br/>RankingEnv (Gymnasium)"]
        T["Policy Network<br/>Neural Network<br/>Input→Hidden→Output"]
        U["Reward Model<br/>NDCG kalkuláció<br/>Expert evaluations"]
        S --> T --> U --> L
    end
    
    style A fill:#e1f5fe
    style I fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#fff3e0
    style J fill:#fce4ec
    style L fill:#f3e5f5
```

### Főbb Rendszerkomponensek

- **Adatfeldolgozási Réteg**: 213,398 bírósági határozat preprocessing és strukturálása
- **Embedding Réteg**: Qwen3-8B (8192D) modell batch feldolgozással A100 80GB GPU-n
- **Gráf Réteg**: NetworkX irányított gráf 1,585,738 csomóponttal és 4,412,596 éllel
- **Hibrid Keresési Motor**: FAISS ANN + gráf traversal algoritmusok kombinációja
- **RL Optimalizálás**: RankingEnv + PolicyNetwork + NDCG-alapú reward modell

### Megerősítéses Tanulás alapú Re-ranking

A projekt legfőbb innovációja egy intelligens re-ranking rendszer, amely DeepSeek által kifejlesztett **Group Relative Policy Optimization (GRPO)** algoritmussal optimalizálja a keresési eredményeket.

#### GRPO Rendszerarchitektúra

**Állapottér (State Space)**:
- Lekérdezés embedding reprezentációja (8192D Qwen3-8B)
- Top-K jelölt dokumentumok embedding vektorai
- Gráf alapú kapcsolódó dokumentumok metrikái (centralitás, PageRank)
- Összefűzött állapotvektor: `[query_emb, doc1_emb, doc2_emb, ..., docK_emb, graph_features]`

**Akciótér (Action Space)**:
- Minden jelölt dokumentumhoz pontszám generálása
- Csoportos rangsorolási döntések (G darab output per query)
- Determinisztikus policy network output: `scores = π_θ(state)`

**Szabály-alapú Értékelési Rendszer** (nincs tanult reward modell):
- **Objektív kritériumok**: Automatikus pontosság ellenőrzés
- **NDCG kalkuláció**: Szakértői ground truth alapján
- **Csoportos normalizálás**: `A_i = (r_i - μ_group) / σ_group`

#### GRPO (Group Relative Policy Optimization)

A GRPO a DeepSeek által bevezetett hatékony RL algoritmus, amely eltávolítja a value network szükségességét és relatív csoportos értékelésen alapul:

**Algoritmus lépései**:
1. **Csoportos mintavételezés**: Minden lekérdezés `q`-hoz generál G darab ranking `{o_1, o_2, ..., o_G}`
2. **Szabály-alapú pontozás**: Rewards `{r_1, r_2, ..., r_G}` objektív kritériumok alapján
3. **Csoportos advantage**: `A_i = (r_i - μ_group) / σ_group` (nincs value function)
4. **Policy update**: PPO-szerű clipped objective GRPO formulával
5. **KL regularizáció**: Reference policy-től való eltérés korlátozása

**GRPO objektív függvény**:
```
L_GRPO = E[min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i)] - β * KL(π_θ || π_ref)
```

Ahol:
- `ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)` - valószínűségi arány
- `A_i` - csoportos relatív advantage
- `β` - KL regularizációs paraméter

**Előnyök a PPO-hoz képest**:
- **50% kevesebb memóriaigény**: Nincs szükség value network-re
- **Gyorsabb konvergencia**: Relatív értékelés hatékonyabb tanulási jelet ad
- **Stabilabb optimalizálás**: Csoportos normalizálás csökkenti a varianciát
- **Egyszerűbb implementáció**: Kevesebb hiperparaméter hangolás szükséges

### 5. Kiértékelési Framework

A rendszer többrétegű kiértékelési módszertant alkalmaz:

- **Automatikus metrikák**: NDCG@5, NDCG@10, MAP, MRR
- **Szakértői értékelés**: Jogi szakértők által végzett relevancia értékelés
- **A/B teszt környezet**: Baseline vs. RL-optimalizált rendszer összehasonlítása



## Technológiai Stack

### Core Components
- **Embedding Model**: Qwen/Qwen3-8B (HuggingFace Transformers)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **RL Framework**: PyTorch + Gymnasium
- **Data Processing**: Pandas, NumPy
- **Infrastructure**: Python 3.9+, CUDA 11.8+

### Cloud Infrastructure
- **GPU Platform**: RunPod, Vast.ai
- **Recommended Hardware**: A100 80GB GPU
- **Storage**: SSD storage minimum 100GB
- **Memory**: 64GB+ RAM ajánlott

## Kutatási Hozzájárulások

### 1. GRPO-alapú Hibrid Keresési Architektúra
Első alkalommal alkalmazza a DeepSeek GRPO algoritmusát jogi dokumentumkeresésben, kombinálja nagy léptékű szemantikus embeddingeket, gráf alapú kapcsolati hálózatot és hatékony megerősítéses tanulást.

### 2. Magyar Jogi Domain Adaptáció
Specializált pipeline magyar bírósági határozatok feldolgozására, amely figyelembe veszi a jogi terminológia sajátosságait és a magyar nyelv specifikus jellemzőit.

### 3. Szabály-alapú Reward Modelling
Innovatív objektív értékelési rendszer, amely szakértői annotáció helyett szabály-alapú kritériumokat használ (pontosság, relev

## Jövőbeli Fejlesztési Irányok

### Rövidtávú Célok
- **Multi-modal embedding**: Dokumentum metaadatok integrálása
- **Hierarchikus keresés**: Jogterület-specifikus specializáció
- **Real-time learning**: Online RL algoritmusok implementálása

### Hosszútávú Vízió
- **Interdiszciplináris keresés**: Kapcsolódó jogterületek összekötése
- **Prediktív elemzés**: Hasonló ügyek kimenetelének előrejelzése

## Szakdolgozat Kontribúciók

Ez a projekt demonstrálja, hogy a modern gépi tanulási technikák kombinációja jelentős javulást eredményezhet a specializált domain-specifikus keresési feladatokban. A megerősítéses tanulás alkalmazása a keresési eredmények re-ranking problémájára újszerű megközelítést jelent a magyar NLP kutatásokban.

A rendszer nem csupán egy technikai implementáció, hanem egy teljes kutatási framework, amely alkalmas további jogi informatikai alkalmazások fejlesztésére és a szemantikus keresés területén végzett alapkutatások támogatására.

---

**Készítette**: Zelenyiánszki Máté
**Intézmény**: Pannon Egyetem 
**Kutatási terület**: Természetes Nyelvfeldolgozás, Információvisszakeresés, Megerősítéses Tanulás  
**Implementáció**: Python, PyTorch, HuggingFace Transformers  
**Licenc**: Kutatási célú felhasználás