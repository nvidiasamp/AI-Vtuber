import json
import requests


RAG_SEARCH_URL = "http://localhost:8081/v1/search"
TRITON_CHAT_URL = "http://localhost:9000/v1/chat/completions"

COLLECTION_NAME = "YOUR_COLLECTION_NAME"
TRITON_MODEL = "ensemble"


def search_retriever(query: str) -> dict:
    """
    NVIDIA RAG Blueprint retrieval-only の /v1/search を呼び出す関数。

    query:
        検索したい質問文。

    collection_names:
        検索対象のコレクション名。

    enable_reranker:
        Trueならrerankerを使う。
        精度は上がりやすいが遅くなる。
        今回のようにすぐ返るならTrueでOK。

    reranker_top_k:
        rerank後に最終的に返す件数。
        1なら最上位1件のみ。
        最初は 1〜3 で十分。

    vdb_top_k:
        vector DBから最初に拾う候補数。
        10なら候補10件からrerank。
    """
    payload = {
        "query": query,
        "collection_names": [COLLECTION_NAME],
        "enable_reranker": True,
        "reranker_top_k": 1,
        "vdb_top_k": 10,
    }

    response = requests.post(RAG_SEARCH_URL, json=payload, timeout=(10, 120))
    response.raise_for_status()
    return response.json()


def build_context(search_result: dict, max_chars: int = 3000) -> str:
    """
    /v1/search の results から LLM に渡す context を作る関数。

    max_chars:
        LLMに渡すcontextの最大文字数。
        長すぎるとTriton側の入力tokenが増えて遅くなる。
        まずは 4000〜8000文字くらいが扱いやすい。
    """
    results = search_result.get("results", [])

    chunks = []
    for i, result in enumerate(results, start=1):
        content = result.get("content", "")
        document_name = result.get("document_name", "unknown")
        score = result.get("score", None)

        chunk = f"[{i}] document={document_name}, score={score}\n{content}"
        chunks.append(chunk)

    context = "\n\n---\n\n".join(chunks)
    return context[:max_chars]


def ask_triton(query: str, context: str) -> dict:
    """
    Triton OpenAI frontend の /v1/chat/completions を呼び出す関数。

    model:
        Triton側のモデル名。
        いま動作確認できている "ensemble" を使う。

    messages:
        OpenAI互換の会話形式。
        system にルール、user に context と質問を渡す。

    temperature:
        0にすると回答が安定しやすい。
        RAGでは 0〜0.2 が無難。

    max_tokens:
        出力token数の上限。
        64だと短すぎることが多い。
        まずは 256 くらいがおすすめ。
    """
    system_prompt = (
        "You are a retrieval-augmented question answering assistant. "
        "Answer using only the provided context. "
        "If the answer is not in the context, say that it is not found in the context."
    )

    user_prompt = f"""Answer the question using the context below.

# Context
{context}

# Question
{query}
"""

    payload = {
        "model": TRITON_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }

    response = requests.post(TRITON_CHAT_URL, json=payload, timeout=(10, 300))
    response.raise_for_status()
    return response.json()


def main() -> None:
    query = "YOUR_QUESTION"

    search_result = search_retriever(query)
    context = build_context(search_result)

    print("=== Retrieved context ===")
    print(context[:1500])
    print()

    triton_result = ask_triton(query, context)

    print("=== Triton raw response ===")
    print(json.dumps(triton_result, ensure_ascii=False, indent=2))
    print()

    answer = triton_result["choices"][0]["message"]["content"]

    print("=== Final answer ===")
    print(answer)


if __name__ == "__main__":
    main()