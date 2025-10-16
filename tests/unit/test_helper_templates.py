from pathlib import Path


HELPER_DIR = Path(__file__).resolve().parent.parent / "src" / "matlab2cpp_agentic_service" / "infrastructure" / "templates" / "helpers"


def test_helper_templates_exist():
    expected_files = {
        "matlab_image_helpers.h",
        "matlab_image_helpers.cpp",
        "msfm_helpers.h",
        "msfm_helpers.cpp",
        "matlab_array_utils.h",
        "matlab_array_utils.cpp",
        "pointmin_helpers.h",
        "pointmin_helpers.cpp",
        "tensor_helpers.h",
        "tensor_helpers.cpp",
        "rk4_helpers.h",
        "rk4_helpers.cpp",
    }

    present = {p.name for p in HELPER_DIR.iterdir() if p.is_file()}
    missing = expected_files - present
    assert not missing, f"Missing helper template files: {sorted(missing)}"


def test_helper_templates_non_empty():
    for template_file in HELPER_DIR.iterdir():
        if template_file.is_file():
            content = template_file.read_text(encoding="utf-8").strip()
            assert content, f"Template file {template_file.name} is empty"
