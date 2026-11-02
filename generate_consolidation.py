import os

target_extensions = {'.swift', '.metal', '.h'}
exclude_dirs = {'Tests', '.build', '.swiftpm', '.git', '.github', '.claude'}
root_dir = 'Sources/VectorAccelerate'
output_file = 'consolidated_library.md'

with open(output_file, 'w', encoding='utf-8') as outfile:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude specified directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in target_extensions:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
                
                language = 'swift' if ext == '.swift' else 'cpp'
                
                outfile.write(f"### `{filepath}`\n\n")
                outfile.write(f"```{language}\n")
                outfile.write(content)
                if not content.endswith('\n'):
                    outfile.write('\n')
                outfile.write("```\n\n")

print(f"Consolidated library created at {output_file}")
