from pandas import read_csv
import torch
import torch.nn as nn
import re
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.text import Tokenizer

# Use the new torch.amp API for mixed precision
from torch.amp import GradScaler, autocast

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            mask_value = torch.finfo(energy.dtype).min
            energy = energy.masked_fill(mask == 0, mask_value)

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # --- THE FIX ---
        # In cross-attention, the query is normalized, but the key and value
        # (from the encoder) are passed through directly.
        norm_query = self.norm1(query)
        attention = self.attention(value, key, norm_query, mask)
        # --- END FIX ---

        # Add residual connection to the original query
        x = query + self.dropout(attention)

        # Pre-LN for the feed-forward layer
        norm_x = self.norm2(x)
        forward = self.feed_forward(norm_x)
        out = x + self.dropout(forward)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )
        for layer in self.layers:
            # For self-attention, value, key, and query are the same
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # First sub-layer: Self-attention on the decoder's input
        norm_x = self.norm(x)
        attention = self.attention(norm_x, norm_x, norm_x, trg_mask)
        query = x + self.dropout(attention)

        # Second sub-layer: Cross-attention with the encoder's output
        # Here, 'value' and 'key' are from the encoder, 'query' is from the decoder
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=4, # Standard is 6
        forward_expansion=4,
        heads=8,
        dropout=0.2,
        device="cpu",
        max_length=256,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


def validate_model(model, val_loader, criterion, device, final_vocab_size):
    model.eval()
    total_val_loss = 0
    total_samples = 0
    with torch.no_grad():
        for src_batch, trg_batch in val_loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)
            output = model(src_batch, trg_batch[:, :-1])
            output = output.reshape(-1, final_vocab_size)
            target = trg_batch[:, 1:].reshape(-1)
            loss = criterion(output, target)
            total_val_loss += loss.item() * src_batch.size(0)
            total_samples += src_batch.size(0)
    return total_val_loss / total_samples

def process_dialog_data(data):
    questions = []
    answers = []
    for dialog_str in data['dialog']:
        try:
            matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", dialog_str)
            raw_sentences = [group1 if group1 else group2 for group1, group2 in matches]
            sentences = [s.strip() for s in raw_sentences if s.strip()]
            if len(sentences) < 2:
                continue
            for i in range(len(sentences) - 1):
                history = sentences[:i + 1]
                question = " [SEP] ".join(history)
                questions.append(question)
                answer = f"<start> {sentences[i + 1]} <end>"
                answers.append(answer)
        except Exception:
            continue
    return questions, answers

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. DATA PREPARATION ---
    data = read_csv('train.csv')
    validation = read_csv('validation.csv')

    questions, answers = process_dialog_data(data)
    val_questions, val_answers = process_dialog_data(validation)

    def clean_text(text):
        return str(text).lower().strip()
    questions = [clean_text(q) for q in questions]
    answers = [clean_text(a) for a in answers]
    val_questions = [clean_text(q) for q in val_questions]
    val_answers = [clean_text(a) for a in val_answers]

    special_tokens = ['<pad>', '<start>', '<end>', '[sep]']
    all_texts = special_tokens + questions + answers
    tokenizer = Tokenizer(oov_token='<unk>', filters='')
    tokenizer.fit_on_texts(all_texts)
    pad_idx = tokenizer.word_index['<pad>']

    features = tokenizer.texts_to_sequences(questions)
    targets = tokenizer.texts_to_sequences(answers)
    val_features = tokenizer.texts_to_sequences(val_questions)
    val_targets = tokenizer.texts_to_sequences(val_answers)

    MAX_LENGTH = 256
    features = [seq[:MAX_LENGTH] for seq in features]
    targets = [seq[:MAX_LENGTH] for seq in targets]
    val_features = [seq[:MAX_LENGTH] for seq in val_features]
    val_targets = [seq[:MAX_LENGTH] for seq in val_targets]

    features = [torch.tensor(seq, dtype=torch.long) for seq in features]
    targets = [torch.tensor(seq, dtype=torch.long) for seq in targets]
    val_features = [torch.tensor(seq, dtype=torch.long) for seq in val_features]
    val_targets = [torch.tensor(seq, dtype=torch.long) for seq in val_targets]

    x = pad_sequence(features, batch_first=True, padding_value=pad_idx)
    trg = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    val_x = pad_sequence(val_features, batch_first=True, padding_value=pad_idx)
    val_trg = pad_sequence(val_targets, batch_first=True, padding_value=pad_idx)

    # --- 2. DATALOADER SETUP ---
    BATCH_SIZE = 32 # Adjust based on your VRAM
    dataset = TensorDataset(x, trg)
    val_dataset = TensorDataset(val_x, val_trg)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    # --- 3. MODEL INITIALIZATION ---
    src_pad_idx = pad_idx
    trg_pad_idx = pad_idx
    final_vocab_size = len(tokenizer.word_index) + 1
    embed_size = 512

    model = Transformer(
        final_vocab_size,
        final_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=embed_size,
        dropout=0.2,
        device=device,
        max_length=MAX_LENGTH
    ).to(device)

    if torch.__version__ >= "2.0":
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    torch.backends.cuda.matmul.allow_tf32 = True

    # --- 4. TRAINING LOOP ---
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

    num_epochs = 10
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    warmup_steps = 4000
    step_num = 0

    def get_lr(step, embed_size, warmup_steps):
        if step == 0: return 0
        arg1 = step ** -0.5
        arg2 = step * (warmup_steps ** -1.5)
        return (embed_size ** -0.5) * min(arg1, arg2)

    print(f"Starting training with {len(train_loader)} batches per epoch...")
    scaler = GradScaler(device.type)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (src_batch, trg_batch) in enumerate(train_loader):
            step_num += 1
            lr = get_lr(step_num, embed_size, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                output = model(src_batch, trg_batch[:, :-1])
                output = output.reshape(-1, final_vocab_size)
                target = trg_batch[:, 1:].reshape(-1)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}')

        avg_loss = total_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, criterion, device, final_vocab_size)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Current LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save logic...
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_word_index': tokenizer.word_index,
                'vocab_size': final_vocab_size,
                'max_length': MAX_LENGTH,
                'src_pad_idx': src_pad_idx,
                'trg_pad_idx': trg_pad_idx,
            }, 'chatbot_model_best.pth')
            print(f'  *** New best model saved with val_loss: {avg_val_loss:.4f} ***')
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epochs')

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # Final save
    import pickle
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_word_index': tokenizer.word_index,
        'vocab_size': final_vocab_size,
        'max_length': MAX_LENGTH,
        'src_pad_idx': src_pad_idx,
        'trg_pad_idx': trg_pad_idx
    }, 'chatbot_model.pth')
    print("-" * 50)
    print("Model saved as 'chatbot_model.pth' and tokenizer as 'tokenizer.pkl'")
