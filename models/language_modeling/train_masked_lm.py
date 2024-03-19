import os
import time

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import wandb
from dataset import IPATokenDataset
from intrinsic_eval import IntrinsicEvaluator
from vocab import *
from language_model import MaskedLM
from util import *
from tqdm import tqdm
import random
import pickle
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpuid = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
print('gpu id is', gpuid)


def mask_phoneme(tokens, features):
    """
    tokens: (B, S)
    features: (B, S, 24)
    """
    # https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
    # generate mask (15% tokens), then apply to inputs
    rand = torch.rand(tokens.shape).to(device)
    mask = (rand < MASK_PERCENT) * (tokens != PAD_IDX) * (tokens != CLS_IDX) * (tokens != SEP_IDX)
    # each sequence within a batch
    for b in range(tokens.shape[0]):
        # get position of masked positions
        selection = torch.flatten(mask[b].nonzero()).tolist()
        tokens[b, selection] = MASK_IDX

    # prepend features for CLS and SEP along given dimensions
    cls_features = torch.ones(features.shape[0], 1, features.shape[2]).to(device) * (MASK_IDX + CLS_IDX)
    sep_features = torch.ones(features.shape[0], 1, features.shape[2]).to(device) * (MASK_IDX + SEP_IDX)
    features = torch.cat((cls_features, features, sep_features), dim=1)
    for b in range(features.shape[0]):
        selection = torch.flatten(mask[b].nonzero()).tolist()
        # the features for a masked token will just be [MASK_IDX] * 24
        features[b, selection, :] = MASK_IDX

    return tokens, features


def mask_vector(tokens, feature_matrix):
    # input: features -> mask a feature
    # target: features of masked phoneme -> mask a feature
    # TODO: finish
    pass


def masked_phoneme_objective(logits, target):
    # TODO: is it bad that this objective function is instantiated every time?
    # predict masked phonemes
    loss = CrossEntropyLoss(ignore_index=PAD_IDX)
    # replace non-masked tokens with PAD_IDX (will be ignored)
    # source: Huggingface documentation
    target = torch.where(target == MASK_IDX, target, PAD_IDX)
    return loss(logits, target)


def masked_vector_objective(logits, target):
    # predict the features of masked phonemes
    # for each of 24 features, predict 1/-1/0 (multiclass) - sum across features
    loss = BCEWithLogitsLoss(ignore_index=PAD_IDX, reduction='sum')
    # TODO: finish; check if objective function is even right
    return None


def train_step(model, train_loader, optimizer, objective, limit_iter_per_epoch=None):
    model.train()

    total_loss = 0
    num_masked_batches = 0
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        tokens = batch['tokens'].to(device)                 # (B, S)
        feature_matrix = batch['feature_array'].to(device)  # (B, S, 24)
        batch_size = tokens.size(0)
        num_segments = feature_matrix.size(1)  # EXcludes <bos> and <eos>

        # mask the tokens and respective features
        if args.predict_vector:
            tokens, feature_matrix = mask_vector(tokens, feature_matrix)
        else:
            tokens, feature_matrix = mask_phoneme(tokens, feature_matrix)

        logits = model(feature_matrix)
        logits = logits.transpose(1, 2)  # (B, V, S) as expected by CrossEntropyLoss
        # no need to left shift the input because we are predicting masked tokens, not the next token
        target = tokens

        # predicting masked phoneme or its features
        loss = objective(logits, target)

        loss.backward()
        optimizer.step()

        if not torch.isnan(loss):
            total_loss += loss.item()
            num_masked_batches += 1
        if limit_iter_per_epoch is not None and i >= limit_iter_per_epoch:
            # this is nothing more than a way to log more frequently than every (full) epoch.
            break

    # N = len(train_loader)
    return {
        'train/loss': total_loss / num_masked_batches
    }


@torch.no_grad()
def validate_step(model, val_loader, objective, evaluator):
    model.eval()
    pooled_phon_embs = []  # for intrinsic evaluation

    total_loss = 0
    num_masked_batches = 0
    for batch in val_loader:
        tokens = batch['tokens'].to(device)
        feature_matrix = batch['feature_array'].to(device)
        batch_size = tokens.size(0)
        num_segments = feature_matrix.size(1)  # EXcludes <bos> and <eos>
        if args.predict_vector:
            tokens, feature_matrix = mask_vector(tokens, feature_matrix)
        else:
            tokens, feature_matrix = mask_phoneme(tokens, feature_matrix)

        # intrinsic evaluation - first obtain a pooled embedding (the useful part)
        # take the LSTM output corresponding to the final token
        embeddings = model.pool(feature_matrix)

        assert embeddings.size()[-1] == args.embedding_dim

        pooled_phon_embs.append(embeddings)

        # language modeling objective - see train_step() for more details
        logits = model(feature_matrix)
        logits = logits.transpose(1, 2)
        target = tokens
        loss = objective(logits, target)
        if not torch.isnan(loss):
            total_loss += loss.item()
            num_masked_batches += 1

    # during training, we calculate the MSE between feature edit distance and the cosine similarity of 2 vectors
    # during evaluation, Pearson's/Spearman's correlation coefficient is used instead
    # evaluator assumes that phon_feats and phon_embs are in the same order
    evaluator.set_phon_embs(torch.cat(pooled_phon_embs, dim=0).detach().cpu().numpy())
    intrinsic_eval = evaluator.run()

    # N = len(val_loader)
    return {
        'val/loss': total_loss / num_masked_batches,
        'val/intrinsic_pearson_correlation': intrinsic_eval['pearson'],
        'val/intrinsic_spearman_correlation': intrinsic_eval['spearman'],
    }


def train(args, vocab, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, eps=1e-9)

    if args.predict_vector:
        objective = masked_vector_objective
    else:
        objective = masked_phoneme_objective

    # TODO: sort the sequences by length? to avoid really bad padding scenarios

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True, 'collate_fn': collate_fn}
    num_tokens = 0
    for lang in args.lang_codes:
        with open(f'data/ipa_tokens/{lang}.txt', "r") as f:
            num_tokens += len(f.readlines())
    # dev = random subset of 1000
    dev_indices = random.sample(range(num_tokens), 1000)
    dev_idx_set = set(dev_indices)
    # train = out of the remaining, pick args.number_thousands * 1000
    train_indices = [i for i in range(num_tokens) if i not in dev_idx_set]
    train_indices = train_indices[:args.number_thousands * 1000]
    assert len(dev_idx_set & set(train_indices)) == 0

    train_dset = IPATokenDataset([f'data/ipa_tokens/{lang}.txt' for lang in args.lang_codes], vocab,
                                indices=train_indices)
    train_loader = DataLoader(train_dset, shuffle=True, **loader_kwargs)
    val_dset = IPATokenDataset([f'data/ipa_tokens/{lang}.txt' for lang in args.lang_codes], vocab,
                                indices=dev_indices)
    val_loader = DataLoader(val_dset, shuffle=False, **loader_kwargs)

    print(len(train_loader), len(train_indices), len(train_indices) // args.batch_size)
    print(len(val_loader), len(dev_indices), len(dev_indices) // args.batch_size)
    assert len(train_loader) == math.ceil(len(train_indices) / args.batch_size)
    assert len(val_loader) == math.ceil(len(dev_indices) / args.batch_size)
    print(f"Loaded {len(train_indices) // 1000}k words for training")

    best_intrinsic = 0
    evaluator = IntrinsicEvaluator()
    # list of IPA transcriptions for each word in the val dataset
    evaluator.set_phon_feats([d['ipa'] for d in val_dset])

    # find out the correlation before any training
    val_loss_dict = validate_step(model, val_loader, objective, evaluator)
    spearman = val_loss_dict["val/intrinsic_spearman_correlation"]
    pearson = val_loss_dict["val/intrinsic_pearson_correlation"]
    wandb.log({"val/initial_spearman_correlation": spearman})
    wandb.log({"val/initial_pearson_correlation": pearson})
    print("Initial spearman correlation is,", spearman)
    print("Initial pearson correlation is,", pearson)

    for ep in range(args.epochs):
        t = time.time()

        train_loss_dict = train_step(model, train_loader, optimizer, objective,
                                     limit_iter_per_epoch=args.limit_iter_per_epoch)
        train_time = time.time()
        val_loss_dict = validate_step(model, val_loader, objective, evaluator)
        best_intrinsic = max(best_intrinsic, val_loss_dict["val/intrinsic_spearman_correlation"])
        val_loss_dict["val/BEST_intrinsic_spearman_correlation"] = best_intrinsic
        wandb.log({"train/lr": optimizer.param_groups[0]['lr'], **train_loss_dict, **val_loss_dict})

        spearman = val_loss_dict["val/intrinsic_spearman_correlation"]
        if abs(spearman) > abs(best_intrinsic):
            best_intrinsic = spearman
            save_model(model, optimizer, args, ipa_vocab, ep, model_save_path)

        print(f'< epoch {ep} >  (elapsed: {time.time() - t:.2f}s, decode time: {time.time() - train_time:.2f}s)')
        print(f'  * [train]  loss: {train_loss_dict["train/loss"]:.6f}')
        print(f'  * [ val ]  loss: {val_loss_dict["val/loss"]:.6f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_codes', nargs='+')
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_layers', help='number of Transformer layers', type=int, default=2)
    parser.add_argument('--num_heads', help='number of Transformer attention heads', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=300, help='eventual embedding dimension')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Transformer internal feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--classifier_dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--limit_iter_per_epoch', type=int, default=200)
    parser.add_argument('--wandb_name', type=str, default="")
    parser.add_argument('--wandb_entity', type=str, default="kalvin")
    parser.add_argument('--sweeping', type=str2bool, default=False)
    parser.add_argument('--predict_vector', type=str2bool, default=False)
    parser.add_argument('--mask_percent', type=float, default=0.3)
    parser.add_argument("--output", default="computed/embd_masked_lm.pkl")
    parser.add_argument("--number_thousands", type=int, default=200, help="amount of training data to use")
    return parser.parse_args()


def extract_embeddings(model, batch_size, model_save_path, embedding_path):
    saved_info = torch.load(model_save_path)
    # fetch the best model
    model.load_state_dict(saved_info['model'])
    args = saved_info['args']

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True, 'collate_fn': collate_fn}
    # no indices passed - use the whole data
    full_data = IPATokenDataset([f'data/ipa_tokens/{lang}.txt' for lang in args.lang_codes], saved_info['ipa_vocab'])
    # the data needs to be the same order as multi.tsv, so no shuffling
    loader = DataLoader(full_data, shuffle=False, **loader_kwargs)
    pooled_phon_embs = []

    for batch in loader:
        tokens = batch['tokens'].to(device)
        feature_matrix = batch['feature_array'].to(device)
        if args.predict_vector:
            tokens, feature_matrix = mask_vector(tokens, feature_matrix)
        else:
            tokens, feature_matrix = mask_phoneme(tokens, feature_matrix)
        embeddings = model.pool(feature_matrix)
        assert embeddings.size()[-1] == args.embedding_dim
        pooled_phon_embs.append(embeddings)

    # convert to list of embeddings
    pooled_phon_embs = torch.cat(pooled_phon_embs, dim=0).detach().cpu().numpy()
    pooled_phon_embs = list(pooled_phon_embs)

    assert len(full_data) == len(pooled_phon_embs)

    Path(embedding_path).parent.mkdir(parents=True, exist_ok=True)
    with open(embedding_path, "wb") as f:
        pickle.dump(pooled_phon_embs, f)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    args = parse_args()
    wandb.init(project="phonological-pooling", name=args.wandb_name, entity=args.wandb_entity,
               mode='disabled' if (not args.wandb_name and not args.sweeping) else 'online')
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py'))
    wandb.config.update(args)
    model_save_path = f'./computed/models/masked_lm_panphon_{"-".join(args.lang_codes)}_gpu{gpuid}_best_intrinsic.pt'

    # TODO: find a better way to get the dims from panphon
    PANPHON_FEATURE_DIM = 24
    MASK_PERCENT = args.mask_percent
    CLS_IDX = BOS_IDX
    SEP_IDX = EOS_IDX

    ipa_vocab = Vocab(tokens_file=args.vocab_file)
    model = MaskedLM(
        num_layers=args.num_layers,
        input_dim=PANPHON_FEATURE_DIM,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        classifier_dropout=args.classifier_dropout,
        vocab_size=len(ipa_vocab),
        predict_vector=args.predict_vector,
    ).to(device)
    train(args, ipa_vocab, model)

    extract_embeddings(model, args.batch_size, model_save_path, args.output)
