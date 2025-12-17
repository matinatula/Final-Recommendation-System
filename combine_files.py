import os

# Replace this with the path to your folder containing Python files
folder_path = "/Users/matinatuladhar/Desktop/final_attempt/"

# Name of the new file where all code will be combined
output_file = "combined_code.py"

# Initialize a string to hold all code
all_code = ""

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):  # Only process Python files
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            # Add a header with the filename for clarity
            all_code += f"# ---- {filename} ----\n"
            all_code += code + "\n\n"

# Save all the combined code into the output file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_code)

print(
    f"All Python files in '{folder_path}' have been combined into '{output_file}'.")
