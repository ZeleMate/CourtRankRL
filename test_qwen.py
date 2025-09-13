#!/usr/bin/env python3
"""
Egyszerű Qwen3 teszt script
"""

def main():
    print("🧪 QWEN3 TESZT MOD")
    print("=" * 50)

    try:
        print("1. Modell betöltése...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
        print("✅ Modell betöltve")

        print("2. Teszt embedding generálás...")
        test_texts = [
            "Ez egy teszt mondat magyar bírósági határozatokhoz.",
            "A szerződés érvénytelen, mert hiányzik az aláírás."
        ]
        embeddings = model.encode(test_texts, normalize_embeddings=True)
        print(f"✅ Embeddingek generálva: {len(embeddings)} db, dimenzió: {len(embeddings[0])}")

        print("3. FAISS teszt...")
        import faiss
        import numpy as np
        arr = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        print("✅ FAISS index létrehozva és feltöltve")

        print("\n✅ MINDEN TESZT SIKERES!")
        print("A Qwen3-Embedding-0.6B modell használatra kész.")

    except Exception as e:
        print(f"❌ TESZT HIBA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
