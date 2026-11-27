import os
import sys
from huggingface_hub import HfApi

def main():
    repo_id = "RuzgarVatansever/Llama-3.2-1B-ConnectomePruned"
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pruned_llama_3_2_1b_instruct")
    
    token = sys.argv[1] if len(sys.argv) > 1 else input("Lutfen Hugging Face Write Token'inizi yapistirin: ").strip()
    
    print("=" * 70)
    print("HUGGING FACE MODEL YUKLEME BASLATILIYOR...")
    print(f"Hedef Repo:   {repo_id}")
    print(f"Kaynak Klasor: {folder_path}")
    print("=" * 70)
    
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial release: Llama-3.2-1B-ConnectomePruned by Ruzgar Vatansever"
    )
    
    print("\n" + "=" * 70)
    print("TEBRIKLER! Model basariyla Hugging Face Hub'a yuklendi!")
    print(f"Sayfa: https://huggingface.co/{repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
