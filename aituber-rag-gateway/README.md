# AI Tuber Kit -> NVIDIA RAG Search -> Triton Gateway

会話のRAG登録と会話メモリ検索を削除した、検索専用ゲートウェイ。

## 処理順

1. AI Tuber KitからOpenAI形式の`messages`を受信
2. 最新のユーザー発話で、`RAG_COLLECTIONS`に指定したコレクションだけを検索
3. RAG Blueprintの`confidence_threshold`で低関連文書を除外
4. 関連文書が0件なら、元の`messages`をそのままTritonへ送信
5. 関連文書がある場合だけコンテキストを追加
6. LLMへ「直接関係する文書だけを使い、無関係なら無視」と指示
7. TritonのOpenAI互換SSEをAI Tuber Kitへ中継

このゲートウェイからIngestor Serverへの書き込みは一切行わない。

## 以前の版からの変更

削除した機能・設定:

- 会話の`POST /v1/documents`
- 会話メモリコレクション検索
- `MEMORY_COLLECTION`
- `ENABLE_MEMORY_SEARCH`
- `ENABLE_CHAT_INGEST`
- `INGEST_BASE_URL`
- `INGEST_BLOCKING`
- `INGEST_BEFORE_DONE`
- `RAG_VECTORSTORE_NAME`
- `create_memory_collection.sh`

## 起動

```bash
cp .env.example .env
nano .env
```

最低限、次を変更する。

```dotenv
RAG_COLLECTIONS=YOUR_COLLECTION_NAME
GATEWAY_API_KEY=十分に長いランダム文字列
```

Dockerで起動する。

```bash
docker compose -f compose.host.yaml down --remove-orphans
docker compose -f compose.host.yaml build --no-cache
docker compose -f compose.host.yaml up -d --force-recreate
```

## AI Tuber Kit

Custom API URL:

```text
http://localhost:9100/v1/chat/completions
```

Custom API Headers:

```json
{"Authorization":"Bearer change-this-long-random-token"}
```

Custom API Body:

```json
{"model":"ensemble","temperature":0.2,"max_tokens":512}
```

システムメッセージを含める設定は`true`にする。

## 関連性判定

### 1. RAG Server側の閾値

```dotenv
ENABLE_RERANKER=true
RAG_CONFIDENCE_THRESHOLD=0.45
```

`confidence_threshold`以上の文書だけが返る。文書が0件なら、RAGコンテキストを追加せずTritonへ送る。

調整例:

```dotenv
# 関連文書を拾いやすくする
RAG_CONFIDENCE_THRESHOLD=0.35

# 無関係な文書を厳しく落とす
RAG_CONFIDENCE_THRESHOLD=0.55
```

### 2. Triton LLM側の判定

閾値を通過した文書にも、次の指示を付ける。

```text
各文書が最新のユーザー依頼に直接役立つか判断する。
直接関係する文書だけ利用する。
関係がない、または関係が弱い文書は完全に無視する。
検索結果を無理に話題へ結び付けない。
```

この二段構えにより、単純なベクトル検索だけより無関係な文書の混入を抑える。

## ヘルスチェック

```bash
curl -sS http://localhost:9100/health | jq
```

期待例:

```json
{
  "ok": true,
  "mode": "search-only",
  "rag_collections": ["YOUR_COLLECTION_NAME"],
  "confidence_threshold": 0.45,
  "context_policy": "conditional",
  "chat_ingestion": false,
  "memory_search": false
}
```

## ログ

```bash
docker logs -f aituber-rag-gateway
```

例:

```text
thread=... collections=manuals docs_used=0 top_score=n/a threshold=0.450 server_threshold=True context_chars=0
```

`docs_used=0`なら、TritonにはRAGコンテキストを付けていない。

```text
thread=... collections=manuals docs_used=2 top_score=0.8123 threshold=0.450 server_threshold=True context_chars=3180
```

この場合は2文書を候補としてTritonへ渡し、最終的な利用可否をLLMにも判断させる。

## Ingestor Serverについて

既存コレクションを検索するだけなら、ゲートウェイはIngestor Serverを必要としない。
新しい資料を追加・更新するときだけIngestor Serverを起動すればよい。
