docker compose -f deploy/compose/nims.yaml down
docker compose -f deploy/compose/vectordb.yaml down
docker compose -f deploy/compose/docker-compose-rag-server.yaml down
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down