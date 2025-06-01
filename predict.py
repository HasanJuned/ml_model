import sys
import json

try:
    input_data = json.load(sys.stdin)
    print(json.dumps({"status": "received", "input_keys": list(input_data.keys())}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
