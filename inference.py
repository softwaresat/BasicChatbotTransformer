import torch
import pickle
from ChatbotTransformer import Transformer  # Your model class
from collections import deque, OrderedDict
import os


class ChatbotInference:
    def __init__(self, model_path, device='cpu', context_turns=3):
        self.device = device
        self.context_turns = context_turns
        self.history = deque(maxlen=context_turns * 2)

        # Load saved model data
        checkpoint = torch.load(model_path, map_location=device)

        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)

        # --- FIX: Handle both new and old model files ---
        try:
            # 1. Try to load parameters from the new model format
            print("Loading model with saved parameters...")
            model_params = checkpoint['model_params']
            model_params['device'] = device
            model_params['dropout'] = 0.0  # Set dropout to 0 for inference
            self.model = Transformer(**model_params).to(device)

        except KeyError:
            # 2. If it fails, fall back to manually defining the architecture
            print("WARNING: 'model_params' not found in checkpoint. Falling back to manual setup.")
            print("Ensure these parameters match the model you trained.")

            vocab_size = len(checkpoint['tokenizer_word_index']) + 1
            src_pad_idx = checkpoint['src_pad_idx']
            trg_pad_idx = checkpoint['trg_pad_idx']
            max_length = checkpoint.get('max_length', 256)

            self.model = Transformer(
                src_vocab_size=vocab_size,
                trg_vocab_size=vocab_size,
                src_pad_idx=src_pad_idx,
                trg_pad_idx=trg_pad_idx,
                embed_size=512,
                num_layers=4,
                forward_expansion=4,
                heads=8,
                dropout=0.0,
                device=device,
                max_length=max_length
            ).to(device)

        # --- FIX for torch.compile() mismatch ---
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                name = k[10:]  # remove `_orig_mod.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        # --- END FIX ---

        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        # Get special token IDs
        wi = self.tokenizer.word_index
        try:
            self.vocab_size = self.model.params['src_vocab_size']
            self.max_length = self.model.params['max_length']
        except AttributeError:
            self.vocab_size = len(checkpoint['tokenizer_word_index']) + 1
            self.max_length = checkpoint.get('max_length', 256)

        self.start_token = wi.get("<start>", 1)
        self.end_token = wi.get("<end>", 1)

    def _build_source_with_context(self, question):
        # Simplify: just use the current question without complex history
        # This reduces confusion and makes responses more direct
        return str(question).strip().lower()

    def _clean_response(self, response):
        """Clean up the generated response"""
        # Remove special tokens
        response = response.replace('<start>', '').replace('<end>', '').replace('<unk>', '')

        # Remove extra separators
        response = response.replace('[sep]', '').replace('[SEP]', '')

        # Clean up whitespace and punctuation
        response = ' '.join(response.split())

        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()

        # Add period if needed
        if response and not response.endswith(('.', '!', '?')):
            response += '.'

        return response.strip()

    def generate_response(self, question, max_gen_length=25, top_k=5, temperature=0.1):
        self.model.eval()
        with torch.no_grad():
            # Use simple question without complex context
            src_text = self._build_source_with_context(question)

            # Debug: Print what we're sending to the model
            print(f"[DEBUG] Input to model: '{src_text}'")

            # Tokenize and prepare the source tensor
            question_seq = self.tokenizer.texts_to_sequences([src_text])[0]
            if not question_seq:  # Handle empty sequences
                question_seq = [1]  # Use a default token

            # Ensure we have at least some tokens
            if len(question_seq) < 2:
                # Add some padding or default tokens
                question_seq = [1] + question_seq

            question_seq = question_seq[:self.max_length]
            src = torch.tensor([question_seq], dtype=torch.long).to(self.device)

            # Initialize the target sequence with the start token
            trg_indices = [self.start_token]

            for step in range(max_gen_length):
                trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(self.device)

                output = self.model(src, trg_tensor)
                logits = output[0, -1, :]

                # Use much more conservative sampling
                if top_k > 0:
                    # Very low temperature for more deterministic responses
                    logits = logits / max(temperature, 0.05)

                    # Even smaller top-k for more focused responses
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))

                    # Apply softmax to get probabilities
                    top_k_probs = torch.softmax(top_k_logits, dim=-1)

                    # Sample from the top-k distribution
                    sampled_index_in_top_k = torch.multinomial(top_k_probs, num_samples=1).item()
                    next_token = top_k_indices[sampled_index_in_top_k].item()
                else:
                    # Greedy sampling - most likely token
                    next_token = torch.argmax(logits).item()

                # Stop if we hit the end token or padding
                if next_token == self.end_token or next_token == 0:
                    break

                trg_indices.append(next_token)

            # Convert token indices to words with better filtering
            trg_tokens = []
            for i in trg_indices[1:]:  # Skip start token
                if i == 0 or i == self.end_token:  # Skip padding and end tokens
                    break
                word = self.tokenizer.index_word.get(i, '')
                if word and word not in ['<unk>', '<pad>', '<start>', '<end>']:
                    trg_tokens.append(word)

            response_text = " ".join(trg_tokens)
            response_text = self._clean_response(response_text)

            # Fallback for empty responses
            if not response_text or len(response_text.strip()) < 2:
                response_text = "I'm not sure how to respond to that."

            # Store only the current question and response (no complex history)
            self.history.clear()  # Clear history to avoid confusion
            self.history.append(question)
            self.history.append(response_text)

            return response_text


def main():
    print("Loading chatbot model...")

    model_path = None
    if os.path.exists('chatbot_model_best.pth'):
        model_path = 'chatbot_model_best.pth'
        print(f"\nAuto-selected: {model_path} (best validation loss)")
    elif os.path.exists('chatbot_model_latest.pth'):
        model_path = 'chatbot_model_latest.pth'
        print(f"\nAuto-selected: {model_path} (most recent training)")
    elif os.path.exists('chatbot_model.pth'):
        model_path = 'chatbot_model.pth'
        print(f"\nAuto-selected: {model_path} (final model)")

    if not model_path:
        print("\nERROR: No trained models found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chatbot = ChatbotInference(model_path, device=device)

    print("\nChatbot loaded successfully!")
    print("Type 'quit' to exit, or 'debug' to toggle debug mode.")
    print("-" * 20)

    debug_mode = False
    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("Bot: Goodbye!")
            break
        if question.lower() == 'debug':
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            continue
        if not question:
            continue
        try:
            # Use very conservative parameters for more consistent responses
            if debug_mode:
                response = chatbot.generate_response(question, top_k=3, temperature=0.1)
            else:
                # Hide debug output in normal mode
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                response = chatbot.generate_response(question, top_k=3, temperature=0.1)
                sys.stdout = old_stdout
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error generating response: {e}\n")


if __name__ == "__main__":
    main()
