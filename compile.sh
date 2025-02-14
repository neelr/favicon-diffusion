#!/bin/bash

# Set shader directory
SHADER_DIR="shaders"

# Create or truncate the output file in the shader directory
echo "/** THIS FILE IS AUTO-GENERATED SHADER CODE. DO NOT MODIFY. */" > "$SHADER_DIR/shaders.js"
echo "" >> "$SHADER_DIR/shaders.js"

# Process each .wgsl file in the shader directory
for file in "$SHADER_DIR"/*.wgsl; do
    if [ -f "$file" ]; then
        # Get just the filename without path and extension
        filename=$(basename "$file")
        # Convert filename to uppercase and replace dots with underscores
        const_name=$(echo "${filename%.*}" | tr '[:lower:]' '[:upper:]')_SHADER_CODE
        
        # Add the constant declaration
        echo "const $const_name = \`" >> "$SHADER_DIR/shaders.js"
        
        # Add the file contents
        cat "$file" >> "$SHADER_DIR/shaders.js"
        
        # Close the template literal
        echo "\`;" >> "$SHADER_DIR/shaders.js"
        echo "" >> "$SHADER_DIR/shaders.js"
    fi
done

# Copy utility.js content if it exists
if [ -f "$SHADER_DIR/utility.js" ]; then
    echo "// Utility functions" >> "$SHADER_DIR/shaders.js"
    echo "" >> "$SHADER_DIR/shaders.js"
    cat "$SHADER_DIR/utility.js" >> "$SHADER_DIR/shaders.js"
fi

echo "Shader compilation complete! Output written to $SHADER_DIR/shaders.js"