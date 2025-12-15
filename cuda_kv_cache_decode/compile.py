import os

def compile():
    setup_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cuda"))
    cmd = f"cd {setup_path} && uv pip install -e . --no-build-isolation"
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    compile()