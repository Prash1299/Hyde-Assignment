import json
from itertools import islice
from dataset.load_data import iter_calls

calls = iter_calls()

for i, row in enumerate(islice(calls, 3)):
    print("\n" + "=" * 70)
    print("CALL:", i)
    print("Trace ID:", row["trace_id"])
    print("Call index:", row["call_index"])
    print("Model:", row["model"])

    print("\nINPUT MESSAGES:")
    print(json.dumps(row["input_messages"], indent=2)[:4000])

    print("\nOUTPUT MESSAGES:")
    print(json.dumps(row["output_messages"], indent=2)[:4000])

    print("\nTOOL DEFINITIONS:")
    print(json.dumps(row["tool_definitions"], indent=2)[:2000])