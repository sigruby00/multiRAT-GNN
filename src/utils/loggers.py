import json


def log_to_file(data, file_path):
    try:
        with open(file_path, "a") as file:
            if isinstance(data, (dict, list)):
                # Convert dictionary or list to JSON string
                file.write(json.dumps(data, indent=4) + "\n")
            else:
                # Directly write the string representation of the data
                file.write(str(data) + "\n")
        print(f"Data successfully logged to {file_path}")
    except Exception as e:
        print(f"An error occurred while logging the data: {e}")
