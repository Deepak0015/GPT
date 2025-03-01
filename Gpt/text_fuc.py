import torch
import torch.nn as nn 




def text_to_token_ids(text,  tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded = torch.tensor(encoded).unsqueeze(0)
    return encoded


def token_ids_to_text(tokens , tokenizer):
    flat  = tokens.squeeze(0)
    decode = tokenizer.decode(flat.tolist())
    return decode

    
def generate_and_sample(model  , idx , context_size ,max_new_tokens ):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            print(logits.shape)
        logits  = logits[:, -1  , :]
        print(logits.shape)
        probs  = torch.softmax(logits  , dim=-1)
        print(probs)
        idx_next = torch.argmax(probs, dim=-1 , keepdim= True)
        print(idx_next)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx 


def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):
    for _ in range(max_new_tokens): #A
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None: #B
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
        logits < min_val,
        torch.tensor(float('-inf')).to(logits.device),
        logits
        )
        if temperature > 0.0: #C
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else: #D
             idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx



def real_time_generation(model, initial_input, context_size, temperature, top_k=None, device="cpu"):
    # Tokenize the initial input and prepare the model context
    idx = torch.tensor(initial_input).unsqueeze(0).to(device)  # Assuming initial_input is tokenized
    
    print("Starting real-time generation...")
    
    # Start generating tokens in real-time
    for new_token in generate(model, idx, max_new_tokens=50, context_size=context_size, temperature=temperature, top_k=top_k, device=device):
        print(f"Generated token: {new_token.item()}")  # Or decode it back to a word
        
        # You can check for user input here and update idx with the new input
        # For instance, wait for the user to input a prompt to append to the context
        user_input = input("Enter new input (or press enter to continue generation): ")
        
        if user_input:
            # Tokenize the new user input and append it to the context
            user_input_tokens = torch.tensor(tokenize(user_input)).unsqueeze(0).to(device)
            idx = torch.cat((idx, user_input_tokens), dim=1)  # Append the new tokens to the context
        else:
            # Continue generating if no new user input
            continue

# Function to tokenize input (adjust depending on your tokenizer)
def tokenize(text):
    # Assuming you have a tokenizer function available
    return [ord(c) for c in text]  # Dummy example: ord() converts char to token id

