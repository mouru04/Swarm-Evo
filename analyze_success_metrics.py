import json

file_path = '/home/iomgaa/Code/Swarm-Evo/workspace/logs/metrics.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    success_count = 0
    for i, entry in enumerate(data):
        observation_str = entry.get('observation', '{}')
        try:
            if isinstance(observation_str, str):
                obs = json.loads(observation_str)
            else:
                obs = observation_str
            
            if isinstance(obs, dict):
                 if obs.get('ok') is True:
                     success_count += 1
                     print(f"\\n--- Entry {i} (Success) ---")
                     obs_data = obs.get('data', {})
                     if obs_data:
                         if isinstance(obs_data, dict):
                             if obs_data.get('type') == 'WriteFileResult':
                                 print(f"Type: WriteFile")
                                 print(f"Path: {obs_data.get('path')}")
                             else:
                                 print(f"Type: {obs_data.get('type', 'Unknown')}")
                                 preview = str(obs_data)[:2000].replace('\\n', ' ')
                                 print(f"Output: {preview}...")
                         else:
                             print(f"Data: {str(obs_data)[:200]}")
        except Exception as e:
            pass

    print(f"\\nTotal successful entries: {success_count}")

except Exception as e:
    print(f"Error: {e}")
