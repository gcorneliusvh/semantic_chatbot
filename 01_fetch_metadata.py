import looker_sdk
import json

# Authenticate using your looker.ini file
sdk = looker_sdk.init40("looker.ini")

MODEL_NAME = "data_block_acs_bigquery"
EXPLORE_NAME = "acs_census_data"

print(f"Fetching metadata for {MODEL_NAME}::{EXPLORE_NAME}...")

try:
    # 1. This is now working!
    explore = sdk.lookml_model_explore(
        lookml_model_name=MODEL_NAME, 
        explore_name=EXPLORE_NAME
    )
    
    fields = []

    # 2. Parse dimensions (THIS IS THE FIX)
    # The fields are nested inside the 'fields' attribute.
    # We also add a check for 'explore.fields' to be safe.
    if explore.fields and explore.fields.dimensions:
        for dim in explore.fields.dimensions:
            fields.append({
                "name": dim.name,
                "label": dim.label_short or dim.label,
                "description": dim.description,
                "type": "dimension"
            })

    # 3. Parse measures (THIS IS THE FIX)
    if explore.fields and explore.fields.measures:
        for mea in explore.fields.measures:
            fields.append({
                "name": mea.name,
                "label": mea.label_short or mea.label,
                "description": mea.description,
                "type": "measure"
            })

    # 4. Save to a file
    output_filename = "acs_census_metadata.json"
    with open(output_filename, "w") as f:
        json.dump(fields, f, indent=2)

    print(f"Successfully fetched {len(fields)} fields.")
    print(f"Metadata saved to {output_filename}")

except Exception as e:
    print(f"Error fetching explore: {e}")