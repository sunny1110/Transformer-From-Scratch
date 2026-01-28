from ntpath import exists
from dill.detect import parents
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


from src.dataset import BilingualDataset, causal_mask
from src.model import build_transformer
from src.config import get_config, get_weights_file_path

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from torch.utils.data import DataLoader, random_split


torch.cuda.empty_cache()

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    # print(encoder_output.shape)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # print(decoder_input)
    # print(decoder_input.shape)

    while True:

        if decoder_input.size(1) == max_len:
            break        

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        
        
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(decoder_input).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, num_examples):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break

def get_all_sentences(ds: torch.utils.data.Dataset, lang: str) -> str:
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config: dict, ds: torch.utils.data.Dataset, lang: str) -> Tokenizer:
    """
        Loads the tokenizer or builds it.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config: dict):

    ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size 

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)


    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"], config["N"], config["h"], config["dropout"], config["d_ff"])
    return model

def get_model_parameter_count(model) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    parameter_count = get_model_parameter_count(model)

    print(f"Number of Parameters in model: {parameter_count:,}")

    writer = SummaryWriter(f'{config["experiment_name"]}_{parameter_count}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    if config["overfit_run"]:
        single_batch = next(iter(train_dataloader))
        

    for epoch in range(initial_epoch, config['num_epochs']):

        # Get a single batch for quick training check
        if config["overfit_run"]:
            batch_iterator = tqdm([single_batch], desc=f"Overfit epoch {epoch:02d}")
        else:
            batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")        


        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device) # (B, Seq_Len)
            decoder_input = batch["decoder_input"].to(device) # (B, Seq_Len)
            encoder_mask = batch["encoder_mask"].to(device) # (B, 1, 1, Seq_Len)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, Seq_Len, Seq_Len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_ouput = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_ouput)

            label = batch["label"].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix_str(f"train loss: {loss.item():6.3f}")
            
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, 2)

        

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

def main():
    config = get_config()
    train_model(config)

if __name__ == "__main__":
    main()