"""
Databricks Deployment Setup Script

Run this script from a terminal with the Databricks CLI configured.
It creates all the prerequisites needed for the deployment notebook
(deployment.ipynb):

  1. MLflow experiment on Databricks
  2. Logs the LangGraph agent as an MLflow model
  3. Registers the model in Unity Catalog
  4. Creates a Model Serving endpoint

Prerequisites:
  - Databricks CLI configured (`databricks auth login`) OR
  - Running inside a Databricks notebook with workspace auth
  - .env file with DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL

  To deploy with a specific workspace, authenticate and create a profile
  FIRST, then pass it with --profile:
      databricks auth login --host https://<workspace>.databricks.com --profile my-profile
      databricks auth profiles          # verify it was created

Usage (from repo root):
    # --api-key is REQUIRED: a PAT for the target workspace's serving endpoints.
    python wk5_langgraph/11.databricks_deployment/deploy_setup.py --api-key dapi...

    # Or with custom model name:
    python wk5_langgraph/11.databricks_deployment/deploy_setup.py --api-key dapi... --model-name my_agent

    # Skip endpoint creation (just register model):
    python wk5_langgraph/11.databricks_deployment/deploy_setup.py --api-key dapi... --skip-endpoint

    # Authenticate with a specific Databricks CLI profile instead of .env:
    python wk5_langgraph/11.databricks_deployment/deploy_setup.py --profile my-profile --api-key dapi...
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ─── Parse arguments ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Set up Databricks prerequisites for LangGraph deployment")
parser.add_argument("--model-name", default="main.default.cs4603_langgraph_agent", help="Unity Catalog model path (default: main.default.cs4603_langgraph_agent)")
parser.add_argument("--endpoint-name", default="cs4603-langgraph-agent", help="Serving endpoint name (default: cs4603-langgraph-agent)")
parser.add_argument("--skip-endpoint", action="store_true", help="Skip creating the serving endpoint")
parser.add_argument("--profile", default=None, help="Databricks CLI profile (~/.databrickscfg) to deploy with. Overrides DATABRICKS_HOST/TOKEN from .env for this run; .env is still used to run the notebooks.")
parser.add_argument("--api-key", required=True, help="Databricks personal access token (PAT) for the TARGET workspace's model serving endpoints. Required because a profile using OAuth login has no static token for the agent's LLM client to use.")
args = parser.parse_args()

MODEL_REGISTRY_NAME = args.model_name

# ─── Bootstrap ───────────────────────────────────────────────────────────────

print("=" * 60)
print("  LangGraph Agent — Databricks Deployment Setup")
print("=" * 60)

from langchain_common import bootstrap_notebook

DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, (llm, llm_noreason), embeddings = bootstrap_notebook()

# When --profile is given, authenticate via the named Databricks CLI profile
# (~/.databrickscfg) instead of the DATABRICKS_HOST/TOKEN loaded from .env.
# The .env values remain the default so students can still run the notebooks.
USE_PROFILE = args.profile is not None
if USE_PROFILE:
    print(f"\n  Auth: Databricks CLI profile '{args.profile}' (overrides .env)")
    # bootstrap_notebook() called load_dotenv(), which sets DATABRICKS_HOST/TOKEN
    # in the environment. The Databricks SDK ranks env vars ABOVE profiles, so
    # they would silently override --profile. Remove them so the profile wins.
    for _var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CONFIG_PROFILE"):
        os.environ.pop(_var, None)
else:
    print(f"\n  Auth: DATABRICKS_HOST/TOKEN from .env")

# Check for Databricks SDK availability (used for serving endpoint creation)
try:
    from databricks.sdk import WorkspaceClient
    if USE_PROFILE:
        w = WorkspaceClient(profile=args.profile)
        # Use the profile's host for display, URLs, and any REST fallbacks
        DATABRICKS_HOST = w.config.host
    else:
        w = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
    HAS_SDK = True
except ImportError:
    if USE_PROFILE:
        print("  ✗ --profile requires the databricks-sdk package. Install it and retry.")
        sys.exit(1)
    HAS_SDK = False

print(f"  Databricks host: {DATABRICKS_HOST}")
print(f"  Model endpoint:  {DATABRICKS_MODEL}")

# Rebuild the agent's LLM clients using the explicitly provided --api-key and the
# resolved host. bootstrap_notebook() built these from .env; when deploying to a
# different workspace (e.g. via --profile with OAuth login) those baked-in
# credentials are wrong, so re-create the clients against the target workspace.
from langchain_common import DatabricksConfig, create_databricks_client

DATABRICKS_TOKEN = args.api_key
_model_cfg = DatabricksConfig(token=DATABRICKS_TOKEN, host=DATABRICKS_HOST, endpoint=DATABRICKS_MODEL)
llm, llm_noreason, embeddings = create_databricks_client(_model_cfg)
print(f"  Model auth:      --api-key (PAT) against {DATABRICKS_HOST}")

# agent.py (imported below, and logged via models-from-code) builds its OWN
# ChatOpenAI from these env vars at import time. When --profile popped them,
# re-set them to the resolved target workspace + --api-key so the agent's model
# client can authenticate here and at serving time.
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
os.environ["DATABRICKS_MODEL"] = DATABRICKS_MODEL

# ─── Step 1: Quick sanity check of the agent ─────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Step 1: Sanity-check the LangGraph agent")
print(f"{'─'*60}")

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

# Import agent tools from agent.py so we don't duplicate definitions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "agent_module",
    os.path.join(os.path.dirname(__file__), "agent.py"),
)
_agent_mod = importlib.util.module_from_spec(_spec)

# Prevent agent.py from calling mlflow.models.set_model during import
import mlflow as _mlflow_tmp
_orig_set_model = _mlflow_tmp.models.set_model
_mlflow_tmp.models.set_model = lambda *a, **kw: None
_spec.loader.exec_module(_agent_mod)
_mlflow_tmp.models.set_model = _orig_set_model

tools = _agent_mod.tools
SYSTEM_PROMPT = _agent_mod.SYSTEM_PROMPT

llm_with_tools = llm_noreason.bind_tools(tools)


def assistant(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()

# Sanity check
result = graph.invoke({"messages": [HumanMessage(content="What is 5 power 3?")]})
answer = result["messages"][-1].content
print(f"  ✓ Agent compiled and tested — 5^3 answer: {answer}")

# ─── Step 2: Log to MLflow ───────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Step 2: Log agent to MLflow")
print(f"{'─'*60}")

import mlflow
import os

if USE_PROFILE:
    # Route MLflow tracking through the CLI profile (reads ~/.databrickscfg)
    mlflow.set_tracking_uri(f"databricks://{args.profile}")
else:
    # Point MLflow at the Databricks workspace from .env (not local sqlite)
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
    mlflow.set_tracking_uri("databricks")

print(f"  MLflow tracking: {mlflow.get_tracking_uri()}")
print(f"  Target host:     {DATABRICKS_HOST}")

# Resolve the current user's home folder for the experiment
try:
    if HAS_SDK:
        db_username = w.current_user.me().user_name
    else:
        import requests
        resp = requests.get(
            f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/preview/scim/v2/Me",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        )
        db_username = resp.json().get("userName", "unknown")
except Exception:
    db_username = "unknown"

experiment_path = f"/Users/{db_username}/wk5-deployment"
print(f"  Experiment:      {experiment_path}")

mlflow.set_experiment(experiment_path)

# Write model code path for models-from-code logging
model_code_path = os.path.join(os.path.dirname(__file__), "agent.py")
print(f"  Model code:      {model_code_path}")

with mlflow.start_run(run_name="langgraph-agent-setup") as run:
    model_info = mlflow.langchain.log_model(
        lc_model=model_code_path,
        name="langgraph_agent",
        input_example={"messages": [{"role": "user", "content": "Add 2 and 3."}]},
    )
    run_id = run.info.run_id

print(f"  ✓ Model logged: {model_info.model_uri}")
print(f"  ✓ Run ID: {run_id}")

# ─── Step 3: Register in Unity Catalog ────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Step 3: Register model in Unity Catalog")
print(f"  Name: {MODEL_REGISTRY_NAME}")
print(f"{'─'*60}")

try:
    mlflow.set_registry_uri(f"databricks-uc://{args.profile}" if USE_PROFILE else "databricks-uc")

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_REGISTRY_NAME,
    )
    model_version = registered.version
    print(f"  ✓ Registered version {model_version}")

except Exception as e:
    print(f"  ✗ Registration failed: {e}")
    sys.exit(1)

# ─── Step 5: Create serving endpoint ─────────────────────────────────────────

if args.skip_endpoint:
    print(f"\n{'─'*60}")
    print(f"  Step 4: Skipped (--skip-endpoint)")
    print(f"{'─'*60}")
else:
    print(f"\n{'─'*60}")
    print(f"  Step 4: Create Model Serving endpoint")
    print(f"  Endpoint: {args.endpoint_name}")
    print(f"{'─'*60}")

    if not HAS_SDK:
        print("  ⚠ databricks-sdk not available — cannot create endpoint automatically.")
        print(f"    Create it manually in the Databricks UI:")
        print(f"    - Go to Serving → New → select '{MODEL_REGISTRY_NAME}' version {model_version}")
        print(f"    - Name it '{args.endpoint_name}'")
        print(f"    - Enable 'Scale to zero'")
    else:
        try:
            from databricks.sdk.service.serving import (
                EndpointCoreConfigInput,
                ServedEntityInput,
            )

            # Check if endpoint already exists
            existing = None
            try:
                existing = w.serving_endpoints.get(args.endpoint_name)
            except Exception:
                pass

            if existing:
                print(f"  Endpoint '{args.endpoint_name}' already exists (state: {existing.state.ready})")
                print(f"  Updating to model version {model_version}...")
                w.serving_endpoints.update_config(
                    name=args.endpoint_name,
                    served_entities=[
                        ServedEntityInput(
                            entity_name=MODEL_REGISTRY_NAME,
                            entity_version=str(model_version),
                            workload_size="Small",
                            scale_to_zero_enabled=True,
                        )
                    ],
                )
                print(f"  ✓ Endpoint updated")
            else:
                print(f"  Creating endpoint '{args.endpoint_name}'...")
                w.serving_endpoints.create(
                    name=args.endpoint_name,
                    config=EndpointCoreConfigInput(
                        served_entities=[
                            ServedEntityInput(
                                entity_name=MODEL_REGISTRY_NAME,
                                entity_version=str(model_version),
                                workload_size="Small",
                                scale_to_zero_enabled=True,
                            )
                        ]
                    ),
                )
                print(f"  ✓ Endpoint '{args.endpoint_name}' created")
                print(f"    It may take a few minutes to become READY.")

            print(f"\n  Endpoint URL: {DATABRICKS_HOST}/serving-endpoints/{args.endpoint_name}/invocations")

        except Exception as e:
            print(f"  ⚠ Could not create endpoint: {e}")
            print(f"    Create it manually in the Databricks UI.")

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  Setup Complete!")
print(f"{'='*60}")
print(f"""
  Model:     {MODEL_REGISTRY_NAME} (version {model_version})
  Endpoint:  {args.endpoint_name}
  Run ID:    {run_id}

  To test the endpoint (once READY):

    import openai
    client = openai.OpenAI(
        api_key="<your-token>",
        base_url="{DATABRICKS_HOST}/serving-endpoints",
    )
    resp = client.chat.completions.create(
        model="{args.endpoint_name}",
        messages=[{{"role": "user", "content": "Multiply 3 by 2."}}],
    )
    print(resp.choices[0].message.content)
""")
