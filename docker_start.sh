USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d nemotron-ranking-ms nemotron-vlm-embedding-ms # nemotron-ocr page-elements graphic-elements table-structure
docker compose -f deploy/compose/vectordb.yaml up -d
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d ingestor-server nv-ingest-ms-runtime redis
docker compose \
    -f deploy/compose/docker-compose-rag-server.yaml \
    -f docker-compose.host-gateway.yaml \
    up -d --force-recreate rag-server rag-frontend