#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Databricks Deployment Setup — CLI Version
#
# Deploys the LangGraph agent using the Databricks CLI instead of the
# Python SDK. Uses a minimal Python call only for MLflow model logging
# (which has no CLI equivalent), then Databricks CLI for registration
# and serving endpoint creation.
#
# Prerequisites:
#   - Databricks CLI (v1.x) installed and authenticated. To target a specific
#     workspace, create a profile FIRST:
#       databricks auth login --host https://<workspace>.databricks.com --profile my-profile
#       databricks auth profiles          # verify it was created
#   - .env file at the repo root with DATABRICKS_MODEL (and DATABRICKS_HOST/
#     DATABRICKS_TOKEN when NOT using --profile)
#   - Python venv activated with project dependencies (uv pip install -r requirements.txt)
#
# Usage (from repo root):
#   # --api-key is REQUIRED: a PAT for the target workspace's serving endpoints.
#   bash wk5_langgraph/11.databricks_deployment/deploy_setup.sh --api-key dapi...
#
#   # Custom model name / endpoint:
#   bash wk5_langgraph/11.databricks_deployment/deploy_setup.sh --api-key dapi... \
#       --model-name main.default.my_agent \
#       --endpoint-name my-agent-endpoint
#
#   # Skip endpoint creation (just log + register):
#   bash wk5_langgraph/11.databricks_deployment/deploy_setup.sh --api-key dapi... --skip-endpoint
#
#   # Deploy to a specific workspace via a Databricks CLI profile:
#   bash wk5_langgraph/11.databricks_deployment/deploy_setup.sh --profile my-profile --api-key dapi...
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────────
MODEL_NAME="main.default.cs4603_langgraph_agent"
ENDPOINT_NAME="cs4603-langgraph-agent"
SKIP_ENDPOINT=false
PROFILE=""
API_KEY=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ─── Parse arguments ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)   MODEL_NAME="$2";   shift 2 ;;
        --endpoint-name) ENDPOINT_NAME="$2"; shift 2 ;;
        --skip-endpoint) SKIP_ENDPOINT=true; shift ;;
        --profile)      PROFILE="$2";      shift 2 ;;
        --api-key)      API_KEY="$2";      shift 2 ;;
        -h|--help)
            echo "Usage: $0 --api-key KEY [--profile NAME] [--model-name NAME] [--endpoint-name NAME] [--skip-endpoint]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Load .env ────────────────────────────────────────────────────────────────
ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
    echo "  Loaded .env from $ENV_FILE"
else
    echo "  ⚠ No .env file found at $ENV_FILE — relying on existing env vars"
fi

: "${DATABRICKS_MODEL:=databricks-qwen35-122b-a10b}"

# --api-key is REQUIRED: PAT for the TARGET workspace's model serving endpoints.
# A profile using OAuth login has no static token for the agent's LLM client.
if [[ -z "$API_KEY" ]]; then
    echo "  ✗ --api-key is required (PAT for the target workspace's serving endpoints)"
    exit 1
fi

# Auth resolution: --profile drives the Databricks CLI + MLflow; otherwise .env.
PROFILE_ARGS=()
if [[ -n "$PROFILE" ]]; then
    # Env vars outrank profiles in the SDK/CLI, so drop the .env host/token first
    # (they were exported by 'set -a; source .env'), then resolve the real host.
    unset DATABRICKS_HOST DATABRICKS_TOKEN DATABRICKS_CONFIG_PROFILE
    DATABRICKS_HOST=$(python -c "from databricks.sdk.core import Config; print(Config(profile='${PROFILE}').host)")
    PROFILE_ARGS=(--profile "$PROFILE")
    MLFLOW_TRACKING_URI="databricks://${PROFILE}"
    echo "  Auth: Databricks CLI profile '${PROFILE}' → ${DATABRICKS_HOST}"
else
    : "${DATABRICKS_HOST:?DATABRICKS_HOST must be set}"
    : "${DATABRICKS_TOKEN:?DATABRICKS_TOKEN must be set}"
    MLFLOW_TRACKING_URI="databricks"
    echo "  Auth: DATABRICKS_HOST/TOKEN from .env"
fi

echo "============================================================"
echo "  LangGraph Agent — Databricks CLI Deployment"
echo "============================================================"
echo "  Host:     $DATABRICKS_HOST"
echo "  Model EP: $DATABRICKS_MODEL"
echo "  UC Model: $MODEL_NAME"
echo "  Endpoint: $ENDPOINT_NAME"
echo ""

# ─── Step 1: Resolve Databricks username ──────────────────────────────────────
echo "────────────────────────────────────────────────────────────"
echo "  Step 1: Resolve workspace username"
echo "────────────────────────────────────────────────────────────"

DB_USER=$(databricks current-user me "${PROFILE_ARGS[@]}" --output json | python -c "import sys,json; print(json.load(sys.stdin)['userName'])")
echo "  ✓ User: $DB_USER"

EXPERIMENT_PATH="/Users/${DB_USER}/wk5-deployment"
echo "  Experiment: $EXPERIMENT_PATH"

# ─── Step 2: Log model to MLflow (requires Python — no CLI equivalent) ───────
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Step 2: Log agent model to MLflow"
echo "────────────────────────────────────────────────────────────"

MODEL_CODE_PATH="${SCRIPT_DIR}/agent.py"
echo "  Model code: $MODEL_CODE_PATH"

# Minimal Python to log the model and output the run ID + model URI
LOG_OUTPUT=$(python -c "
import os, mlflow

# Model serving auth uses the required --api-key (agent.py reads these at import)
os.environ['DATABRICKS_HOST'] = '${DATABRICKS_HOST}'
os.environ['DATABRICKS_TOKEN'] = '${API_KEY}'
os.environ['DATABRICKS_MODEL'] = '${DATABRICKS_MODEL}'

mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${EXPERIMENT_PATH}')

with mlflow.start_run(run_name='langgraph-agent-cli') as run:
    model_info = mlflow.langchain.log_model(
        lc_model='${MODEL_CODE_PATH}',
        name='langgraph_agent',
        input_example={'messages': [{'role': 'user', 'content': 'What is RAG?'}]},
    )
    print(f'{run.info.run_id}')
    print(f'{model_info.model_uri}')
")

RUN_ID=$(echo "$LOG_OUTPUT" | tail -2 | head -1)
MODEL_URI=$(echo "$LOG_OUTPUT" | tail -1)

echo "  ✓ Run ID:    $RUN_ID"
echo "  ✓ Model URI: $MODEL_URI"

# ─── Step 3: Register model in Unity Catalog via CLI ─────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Step 3: Register model in Unity Catalog"
echo "  Name: $MODEL_NAME"
echo "────────────────────────────────────────────────────────────"

# Ensure the registered model exists in UC (create if missing)
if ! databricks registered-models get "$MODEL_NAME" "${PROFILE_ARGS[@]}" --output json 2>/dev/null; then
    echo "  Creating registered model '$MODEL_NAME'..."
    databricks registered-models create --name "$MODEL_NAME" "${PROFILE_ARGS[@]}"
fi

# Create a new model version from the logged run
VERSION_OUTPUT=$(databricks model-versions create \
    --name "$MODEL_NAME" \
    --source "$MODEL_URI" \
    --run-id "$RUN_ID" \
    "${PROFILE_ARGS[@]}" \
    --output json)

MODEL_VERSION=$(echo "$VERSION_OUTPUT" | python -c "import sys,json; print(json.load(sys.stdin)['model_version']['version'])")
echo "  ✓ Registered version: $MODEL_VERSION"

# ─── Step 4: Create / update serving endpoint via CLI ─────────────────────────
if [[ "$SKIP_ENDPOINT" == true ]]; then
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Step 4: Skipped (--skip-endpoint)"
    echo "────────────────────────────────────────────────────────────"
else
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Step 4: Create/update Model Serving endpoint"
    echo "  Endpoint: $ENDPOINT_NAME"
    echo "────────────────────────────────────────────────────────────"

    ENTITY_JSON=$(cat <<EOF
{
    "entity_name": "${MODEL_NAME}",
    "entity_version": "${MODEL_VERSION}",
    "workload_size": "Small",
    "scale_to_zero_enabled": true
}
EOF
)

    # Check if endpoint exists
    if databricks serving-endpoints get "$ENDPOINT_NAME" "${PROFILE_ARGS[@]}" --output json 2>/dev/null; then
        echo "  Endpoint exists — updating to version $MODEL_VERSION..."
        databricks serving-endpoints update-config "$ENDPOINT_NAME" "${PROFILE_ARGS[@]}" \
            --json "{\"served_entities\": [${ENTITY_JSON}]}"
        echo "  ✓ Endpoint updated"
    else
        echo "  Creating endpoint '$ENDPOINT_NAME'..."
        databricks serving-endpoints create "${PROFILE_ARGS[@]}" --json "{
            \"name\": \"${ENDPOINT_NAME}\",
            \"config\": {
                \"served_entities\": [${ENTITY_JSON}]
            }
        }"
        echo "  ✓ Endpoint created (may take a few minutes to become READY)"
    fi

    echo "  Endpoint URL: ${DATABRICKS_HOST}/serving-endpoints/${ENDPOINT_NAME}/invocations"
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
cat <<EOF

  Model:     $MODEL_NAME (version $MODEL_VERSION)
  Endpoint:  $ENDPOINT_NAME
  Run ID:    $RUN_ID

  To check endpoint status:
    databricks serving-endpoints get $ENDPOINT_NAME ${PROFILE:+--profile $PROFILE}

  To test the endpoint (once READY):
    curl -X POST "${DATABRICKS_HOST}/serving-endpoints/${ENDPOINT_NAME}/invocations" \\
      -H "Authorization: Bearer ${API_KEY}" \\
      -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "Convert 100F to Celsius"}]}'

  Or with Python:
    import openai
    client = openai.OpenAI(
        api_key="<your-token>",
        base_url="${DATABRICKS_HOST}/serving-endpoints",
    )
    resp = client.chat.completions.create(
        model="${ENDPOINT_NAME}",
        messages=[{"role": "user", "content": "What is RAG in LLMs?"}],
    )
    print(resp.choices[0].message.content)
EOF
