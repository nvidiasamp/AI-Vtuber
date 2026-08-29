# AI-VTuber

## Triton Inference Server の起動



**1. 始めにNGC CatalogよりTriton Inference Server の Docker コンテナを取得します (https://catalog.ngc.nvidia.com/)**

``` bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3 bash
```

**Triton Inference Server** のバージョンは**25.11**、**25.12**で確認しました。


**2. TensorRT-LLM をクローンします**

``` bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```


**3. Hugging Face より使用するLLM をダウンロードします**

``` bash
pip install -U "huggingface_hub"
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir llama
```

この例ではLlama-3.2-1B-Instructを使用します。`--local-dir` はオプションでモデルのダウンロード先を指定します。


**4. Hugging Face 形式のモデルをTriton Inference Server で使用できる形に変更します**

``` bash
export MODEL_NAME="llama"     # MODEL_NAME には Hugging Face から取得したモデルのディレクトリ（--local-dir）を指定してください
export UNIFIED_CKPT_PATH=/tmp/ckpt/${MODEL_NAME}/
export ENGINE_PATH=/tmp/engines/${MODEL_NAME}/

python3 /app/examples/models/core/llama/convert_checkpoint.py --model_dir ${MODEL_NAME}/ \
                             --output_dir ${UNIFIED_CKPT_PATH} \
                             --dtype float16

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${ENGINE_PATH} \
             --kv_cache_type paged \
             --max_batch_size 16

cp TensorRT-LLM/triton_backend/all_models/inflight_batcher_llm/ ${MODEL_NAME}_ifb -r

python3 TensorRT-LLM/triton_backend/tools/fill_template.py -i ${MODEL_NAME}_ifb/preprocessing/config.pbtxt tokenizer_dir:${MODEL_NAME}/,triton_max_batch_size:64,preprocessing_instance_count:1
python3 TensorRT-LLM/triton_backend/tools/fill_template.py -i ${MODEL_NAME}_ifb/postprocessing/config.pbtxt tokenizer_dir:${MODEL_NAME}/,triton_max_batch_size:64,postprocessing_instance_count:1
python3 TensorRT-LLM/triton_backend/tools/fill_template.py -i ${MODEL_NAME}_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16
python3 TensorRT-LLM/triton_backend/tools/fill_template.py -i ${MODEL_NAME}_ifb/ensemble/config.pbtxt triton_max_batch_size:64,logits_datatype:TYPE_FP32
python3 TensorRT-LLM/triton_backend/tools/fill_template.py -i ${MODEL_NAME}_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16
```

**${MODEL_NAME}_ifb/tensorrt_llm/config.pbtx** の **decoupled** を **True** に変更します。

**5. OpenAI 形式でモデルを起動します**

``` bash
python3 /opt/tritonserver/python/openai/openai_frontend/main.py --model-repository ${MODEL_NAME}_ifb/ --tokenizer ${MODEL_NAME}/
```

以下のようにterminalで表示されていれば正常です。

``` bash
+----------------------------------+----------------------------------------------------------------------------------------+
| Option                           | Value                                                                                  |
+----------------------------------+----------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                 |
| server_version                   | 2.64.0                                                                                 |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_ |
|                                  | policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data  |
|                                  | parameters statistics trace logging                                                    |
| model_repository_path[0]         | llama_ifb/                                                                             |
| model_control_mode               | MODE_NONE                                                                              |
| strict_model_config              | 0                                                                                      |
| model_config_name                |                                                                                        |
| rate_limit                       | OFF                                                                                    |
| pinned_memory_pool_byte_size     | 268435456                                                                              |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                               |
| min_supported_compute_capability | 6.0                                                                                    |
| strict_readiness                 | 1                                                                                      |
| exit_timeout                     | 30                                                                                     |
| cache_enabled                    | 0                                                                                      |
+----------------------------------+----------------------------------------------------------------------------------------+

Found model: name='ensemble', backend='ensemble'
Found model: name='postprocessing', backend='python'
Found model: name='preprocessing', backend='python'
Found model: name='tensorrt_llm', backend='tensorrtllm'
Found model: name='tensorrt_llm_bls', backend='python'
[WARNING] Adding CORS for the following origins: ['http://localhost']
INFO:     Started server process [30658]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```


**6. Triton Inference Server の停止方法**
`pkill tritonserver` で停止できます。


## NVIDIA RAG Blueprintの起動


**1. NVIDIA NGC より API Key を取得します**

***Legacy Key*** ではなく、**nvapi-** より始まる ***Personal Key*** を使用します。**[こちら](https://org.ngc.nvidia.com/account/api-keys)** から API Key を取得してください。
また、RAG Blueprint の起動に関しては**Triton Inference Server のコンテナ外** で実行してください。

``` bash
export NGC_API_KEY="nvapi-..."
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```


**2. NVIDIA RAG Blueprint をクローンします**

``` bash
git clone https://github.com/NVIDIA-AI-Blueprints/rag.git
cd rag
```


**3. RAG Blueprint を Retrieval-Only Mode で起動します**

Triton Inference Server のLLMで出力するために、Retrieval-Only Mode で起動します。
今回起動するのは以下の機能です。
+ **Embedding NIM** - ベクトル変換
+ **Reranking NIM** - 取得結果の並び替え
+ **Vector Database** - ベクトルの保存と検索
+ **RAG Server** - APIリクエストの処理

今回起動しないのは以下の機能です。
+ **LLM NIM** - 結果の出力

``` bash
export APP_LLM_SERVERURL="https://integrate.api.nvidia.com/v1"

source deploy/compose/.env

cat > docker-compose.host-gateway.yaml <<'YAML'
services:
  rag-server:
    extra_hosts:
      - "host.docker.internal:host-gateway"
YAML

USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d nemotron-ranking-ms nemotron-vlm-embedding-ms
docker compose -f deploy/compose/vectordb.yaml up -d
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d ingestor-server nv-ingest-ms-runtime redis
docker compose \
    -f deploy/compose/docker-compose-rag-server.yaml \
    -f docker-compose.host-gateway.yaml \
    up -d --force-recreate rag-server rag-frontend
```

上記のコードでコンテナを起動し、下記のコードで正常性を確認します。

``` bash
watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
```
出力結果の **STATUS** が **healthy** になっていることを確認します。

``` bash
# 例

NAMES                          STATUS
nemotron-ranking-ms       Up 5 minutes (healthy)
nemotron-vlm-embedding-ms Up 5 minutes (healthy)
                    :
                    :
```

2回目以降の起動では以下のファイルを使用することで起動できます。

``` bash
bash docker_start.sh
```

**4. RAG にデータを追加します**

ブラウザを開き、`http://localhost:8090` または `http://<workstation-ip-address>:8090` にアクセスします。
![rag](https://docs.nvidia.com/rag/2.3.0/_images/ui-empty.png)
**New Collection** をクリックし、ファイルをアップロードします。
![rag2](https://docs.nvidia.com/rag/2.3.0/_images/ui-create-new.png)


**5. RAG Blueprint の停止方法**

``` bash
bash docker_stop.sh
```

## RAG Blueprint と Triton Inference Server の接続確認

**Triron Inference Server** と **RAG Blueprint** が起動している状態かつ2つのコンテナ外で以下のコマンドより出力します。

``` bash
QUESTION="YOUR_QUESTION"

CONTEXT=$(curl -sS -X POST http://localhost:8081/v1/search \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"$QUESTION\",
    \"collection_names\": [\"YOUR_COLLECTION_NAME\"],
    \"enable_reranker\": true,
    \"reranker_top_k\": 1,
    \"vdb_top_k\": 5
  }" | jq -r '.results[0].content' | head -c 3000)  # 推論時にメモリが不足する場合があるので、数値を小さくすると解消できます。

echo "context chars: ${#CONTEXT}"

jq -n \
  --arg model "ensemble" \
  --arg context "$CONTEXT" \
  --arg question "$QUESTION" \
  '{
    model: $model,
    messages: [
      {role: "system", content: "Answer using only the context. Keep it short."},
      {role: "user", content: ("# Context\n" + $context + "\n\n# Question\n" + $question)}
    ],
    temperature: 0,
    max_tokens: 128
  }' \
| curl -sS http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- \
| jq -r '.choices[0].message.content'
```

## RAG Blueprint の結果を Triton Inference Server に反映させ、AI Tuber Kit 上で表示

**1. Gateway の作成**

``` bash
cd aituber-rag-gateway

cp .env.example .env
```

**TRITON_URL**, **TRITON_MODEL**, **RAG_SEARCH_URL**, **RAG_COLLECTIONS** を作成した環境に応じて変更します。
**TRITON_URL**, **TRITON_MODEL**, **RAG_SEARCH_URL** はポートの変更を行なっていなければデフォルトのままで実行できます。
**RAG_COLLECTIONS** には作成したコレクション名を入力してください。複数選択できます。

``` bash
docker compose -f compose.host.yaml up -d --build
```


**2. AI Tuber Kit**

``` bash
git clone https://github.com/tegnike/aituber-kit.git
cd aituber-kit

# パッケージのインストール
npm install

# 必要に応じて.envファイルを作成
cp .env.example .env
```

次に、AI Tuber Kitの設定を変更します。
**設定** -> **AI設定** にある **AIサービスを選択** を　**Custom API** に変更
**設定** -> **AI設定** にある **カスタムAPIエンドポイント** を **http://localhost:9100/v1/chat/completions** に変更

これらの変更は .env の**NEXT_PUBLIC_SELECT_AI_SERVICE**, **NEXT_PUBLIC_CUSTOM_API_URL** からも変更できます。

最後に AI Tuber Kit を起動します。

``` bash
npm run dev
```