NS := aiforge
DEPLOYMENTS := ollama searxng qdrant postgresql langfuse gateway testrunner
MANIFESTS := namespace.yaml searxng.yaml qdrant.yaml ollama.yaml \
             postgresql.yaml langfuse.yaml gateway.yaml testrunner.yaml
SUITE ?= all

.PHONY: build deploy warmup test ktest testlogs undeploy purge help

.DEFAULT_GOAL := help

build:
	@SHA=$$(images/gateway/build.sh) && \
	    sed -i "s|image: gateway:.*|image: gateway:$$SHA|" k8s/gateway.yaml && \
	    echo "Stamped gateway:$$SHA"
	@SHA=$$(images/testrunner/build.sh) && \
	    sed -i "s|image: testrunner:.*|image: testrunner:$$SHA|" k8s/testrunner.yaml && \
	    echo "Stamped testrunner:$$SHA"

deploy: build
	@. ./config.env && \
	    sed -i "s|AGENT_MODEL: .*|AGENT_MODEL: \"$$AGENT_MODEL\"|" k8s/ollama.yaml
	@kubectl get ns $(NS) >/dev/null 2>&1 || kubectl create ns $(NS)
	@if ! kubectl get secret searxng-secret -n $(NS) >/dev/null 2>&1; then \
	    SECRET=$$(openssl rand -hex 32) && \
	    kubectl create secret generic searxng-secret \
	        --namespace $(NS) \
	        --from-literal=SEARXNG_SECRET_KEY="$$SECRET" && \
	    echo "Created SearXNG secret."; \
	fi
	@for f in $(MANIFESTS); do kubectl apply -f k8s/$$f; done
	kubectl -n $(NS) rollout restart deploy/gateway
	kubectl -n $(NS) rollout restart deploy/testrunner
	@for d in $(DEPLOYMENTS); do kubectl -n $(NS) rollout status deploy/$$d; done
	@EXPECTED=$$(grep 'image: gateway:' k8s/gateway.yaml | sed 's/.*image: //'); \
	RUNNING=$$(kubectl -n $(NS) get pods -l app=gateway \
	    -o jsonpath='{.items[0].spec.containers[0].image}' 2>/dev/null); \
	if [ "$$RUNNING" != "$$EXPECTED" ]; then \
	    echo "Error: running $$RUNNING, expected $$EXPECTED"; exit 1; \
	fi; \
	echo "Verified gateway $$EXPECTED"
	@$(MAKE) warmup
	@echo ""
	@echo "Done. Services:"
	@echo "  SearXNG  http://localhost:31080"
	@echo "  Qdrant   http://localhost:31333"
	@echo "  Gateway  http://localhost:31400"
	@echo "  Langfuse http://localhost:31300"

warmup:
	@echo "Warming up model on GPU (this may take a minute)..."
	@kubectl delete job warmup -n $(NS) --ignore-not-found
	@kubectl apply -f k8s/warmup.yaml
	@kubectl wait --for=condition=complete --timeout=300s job/warmup -n $(NS)
	@kubectl logs job/warmup -n $(NS) --tail=1

test:
	kubectl exec -it -n $(NS) deploy/testrunner -- env SUITE=$(SUITE) /app/run.sh

ktest:
	@kubectl exec -n $(NS) deploy/testrunner -- \
	    sh -c '> /tmp/test.log; env SUITE=$(SUITE) /app/run.sh >> /tmp/test.log 2>&1 &'
	@echo "Tests started (SUITE=$(SUITE)). View with: make testlogs"

testlogs:
	@mkdir -p ~/.lnav/formats/installed
	@cp lnav/testrunner.json ~/.lnav/formats/installed/testrunner.json
	kubectl logs -f -n $(NS) deploy/testrunner | lnav

undeploy:
	@if ! kubectl get ns $(NS) >/dev/null 2>&1; then \
	    echo "Namespace $(NS) does not exist. Nothing to remove."; exit 0; \
	fi
	@echo "Removing resources in $(NS)..."
	@kubectl delete deploy --all -n $(NS) --ignore-not-found
	@kubectl delete job --all -n $(NS) --ignore-not-found
	@kubectl delete svc --all -n $(NS) --ignore-not-found
	@kubectl delete configmap --all -n $(NS) --ignore-not-found
	@kubectl delete secret --all -n $(NS) --ignore-not-found
	@kubectl wait --for=delete pod --all -n $(NS) --timeout=120s 2>/dev/null || true
	@echo ""
	@echo "Removed. Persistent volume claims preserved:"
	@kubectl get pvc -n $(NS) 2>/dev/null || echo "  (none)"

purge: undeploy
	@kubectl delete pvc --all -n $(NS) --ignore-not-found
	@kubectl delete ns $(NS) --ignore-not-found
	@echo "Fully purged. All data deleted."

help:
	@echo "make build          Build gateway and testrunner images"
	@echo "make deploy         Build, apply manifests, wait, verify, warmup"
	@echo "make test           Run all test suites in the cluster"
	@echo "make test SUITE=x   Run one suite (unit, integration, toolcalling, bench)"
	@echo "make ktest          Run tests in background on the pod"
	@echo "make ktest SUITE=x  Run one suite in background on the pod"
	@echo "make testlogs       View test output in lnav"
	@echo "make warmup         Run warmup Job to load model into GPU"
	@echo "make undeploy       Remove deployments, keep persistent data"
	@echo "make purge          Undeploy and delete all persistent data"
