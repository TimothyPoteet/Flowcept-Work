import json

from flowcept.agents.agent_client import run_tool

input_dir = "./Flowcept-Work/input_files/single_workflow"
df_path = f"{input_dir}/single_workflow.csv"
schema_path = f"{input_dir}/current_tasks_schema.json"
#
queries = [
    {
        "id": 1,
        "query_text": "how many tasks?",
        "expected_response": "result = len(df)"
     }
]


for query in queries:
    query_text = query.get("query_text")
    agent_output = run_tool(tool_name="query_on_saved_df", kwargs={"query": query_text, "dynamic_schema_path": schema_path, "df_path": df_path})[0]
    print(agent_output)
    result = json.loads(agent_output)["result"]
    if result["result_code"] == query["expected_response"]:
        print("Perfect response")


