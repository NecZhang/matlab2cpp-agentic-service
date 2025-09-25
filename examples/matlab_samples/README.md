# MATLAB Samples

This directory contains sample MATLAB projects for testing the MATLAB2C++ conversion service.

## Directory Structure

```
matlab_samples/
├── .gitkeep              # Maintains directory structure
├── README.md             # This file
├── [user_samples]/       # User-added MATLAB files (ignored by git)
└── [project_folders]/    # Multi-file MATLAB projects (ignored by git)
```

## Usage

### Adding Your Own Samples

1. **Single File Projects**: Place `.m` files directly in this directory
2. **Multi-File Projects**: Create subdirectories with multiple `.m` files

### Example Structure

```
matlab_samples/
├── my_filter.m           # Single file example
├── my_project/           # Multi-file project
│   ├── main.m
│   ├── helper1.m
│   └── helper2.m
└── another_example.m     # Another single file
```

### Testing Conversion

Use the CLI to convert samples:

```bash
# Single file conversion
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/my_filter.m my_filter_cpp

# Multi-file project conversion  
uv run python -m matlab2cpp_agentic_service.cli convert examples/matlab_samples/my_project my_project_cpp
```

## Note

- All MATLAB files in this directory are ignored by git to avoid committing large files
- The directory structure is preserved with `.gitkeep`
- Users can add their own samples without affecting the repository


