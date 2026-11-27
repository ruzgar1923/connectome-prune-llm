import os
import sys
from huggingface_hub import HfApi

def main():
    space_id = "RuzgarVatansever/connectome-prune-llama-playground"
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space")
    token = sys.argv[1] if len(sys.argv) > 1 else input("Token: ").strip()

    print("=" * 70)
    print("UPLOADING HUGGING FACE SPACE...")
    print(f"Space ID: {space_id}")
    print("=" * 70)

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    except Exception as e:
        print("Note on repo creation:", e)

    api.upload_folder(
        folder_path=folder_path,
        repo_id=space_id,
        repo_type="space",
        commit_message="Initial release: ConnectomePrune-LLM Live Gradio Playground"
    )
    print("\n[SUCCESS] Hugging Face Space is live at: https://huggingface.co/spaces/" + space_id)

if __name__ == "__main__":
    main()
