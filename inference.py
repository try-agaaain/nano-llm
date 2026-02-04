import torch
from pathlib import Path
from model import NanoLLM
from tokenizer import load_or_train_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading tokenizer...")
    tokenizer = load_or_train_tokenizer(tokenizer_path="./tokenizer", force_retrain=False)
    
    print("Loading model...")
    model_state = torch.load("best_model.pt", map_location=device)
    
    # Extract model dimensions from state dict
    d_model = model_state['embedding.weight'].shape[1]
    vocab_size = model_state['embedding.weight'].shape[0]
    
    # Get max_seq_len from pos_encoding shape
    max_seq_len = model_state['pos_encoding.pe'].shape[1]
    
    model = NanoLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=4,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"Model loaded. Device: {device}")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            prompt_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    prompt=prompt_ids,
                    max_length=100,
                    temperature=0.8,
                    top_k=50,
                )
            
            generated_only_ids = generated_ids[0, prompt_ids.size(1):].tolist()
            response = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            
            print(f"Model: {response.strip()}\n")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
