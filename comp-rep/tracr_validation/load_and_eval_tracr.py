import argparse
import logging

import torch
from comp_rep.models.nanoGPT import GPT
from tracr_data_utils import ErazrTokenizer, load_datasets, unstringify
from tracr_model_utils import get_config_weights_and_vocab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    config, tok_embed, pos_embed, blocks_embeds, vocab, bos, pad, unembedding_mtx = (
        get_config_weights_and_vocab(args.model_path)
    )

    model = GPT.from_tracr(config, tok_embed, pos_embed, unembedding_mtx, blocks_embeds)

    q_w_idx = 0
    k_w_idx = 1
    v_w_idx = 2
    o_w_idx = 3
    up_w_idx = 4
    down_w_idx = 5
    q_b_idx = 6
    k_b_idx = 7
    v_b_idx = 8
    o_b_idx = 9
    up_b_idx = 10
    down_b_idx = 11
    # Loop through the layers and assert that the loaded weights are equal to the model weights from the blocks_embeds
    for layer_idx in range(config.n_layer):
        print(f"Layer {layer_idx}")
        q_w = model.transformer.h[layer_idx].attn.q_proj.weight.T
        assert torch.allclose(q_w, blocks_embeds[layer_idx][q_w_idx])
        k_w = model.transformer.h[layer_idx].attn.k_proj.weight.T
        assert torch.allclose(k_w, blocks_embeds[layer_idx][k_w_idx])
        v_w = model.transformer.h[layer_idx].attn.v_proj.weight.T
        assert torch.allclose(v_w, blocks_embeds[layer_idx][v_w_idx])
        o_w = model.transformer.h[layer_idx].attn.c_proj.weight.T
        assert torch.allclose(o_w, blocks_embeds[layer_idx][o_w_idx])
        up_w = model.transformer.h[layer_idx].mlp.c_fc.weight.T
        assert torch.allclose(up_w, blocks_embeds[layer_idx][up_w_idx])
        down_w = model.transformer.h[layer_idx].mlp.c_proj.weight.T
        assert torch.allclose(down_w, blocks_embeds[layer_idx][down_w_idx])

        q_b = model.transformer.h[layer_idx].attn.q_proj.bias
        assert torch.allclose(q_b, blocks_embeds[layer_idx][q_b_idx])
        k_b = model.transformer.h[layer_idx].attn.k_proj.bias
        assert torch.allclose(k_b, blocks_embeds[layer_idx][k_b_idx])
        v_b = model.transformer.h[layer_idx].attn.v_proj.bias
        assert torch.allclose(v_b, blocks_embeds[layer_idx][v_b_idx])
        o_b = model.transformer.h[layer_idx].attn.c_proj.bias
        assert torch.allclose(o_b, blocks_embeds[layer_idx][o_b_idx])
        up_b = model.transformer.h[layer_idx].mlp.c_fc.bias
        assert torch.allclose(up_b, blocks_embeds[layer_idx][up_b_idx])
        down_b = model.transformer.h[layer_idx].mlp.c_proj.bias
        assert torch.allclose(down_b, blocks_embeds[layer_idx][down_b_idx])

    # Assert that the embedding and LM hedding weights are equal to the loaded weights
    assert torch.allclose(model.transformer.wte.weight, tok_embed)
    assert torch.allclose(model.transformer.wpe.weight, pos_embed)
    # LM head
    assert torch.allclose(model.lm_head.weight.T, unembedding_mtx)

    raw_datasets = load_datasets(args.data_path, 100, 100)
    tokenizer = ErazrTokenizer(vocab, bos, pad)
    eval_dataset = raw_datasets["validation"]

    model.eval()
    eval_batch_size: int = 1
    correct = 0
    print("----- PREDICTION LOOP -----")
    for i in range(0, len(eval_dataset), eval_batch_size):
        input_ids = []
        corr_input_ids = []
        labels = []

        for j in range(i, min(i + eval_batch_size, len(eval_dataset))):
            input_ids_ = tokenizer(
                [unstringify(eval_dataset[j]["seq"])], return_tensors="pt"
            )[0]
            corr_input_ids_ = tokenizer(
                [unstringify(eval_dataset[j]["corr_seq"])], return_tensors="pt"
            )[0]
            labels_ = tokenizer(
                [unstringify(eval_dataset[j]["target"])], return_tensors="pt"
            )[0]

            input_ids.append(input_ids_)
            corr_input_ids.append(corr_input_ids_)
            labels.append(labels_)

        input_ids = torch.stack(input_ids)
        corr_input_ids = torch.stack(corr_input_ids)
        labels = torch.stack(labels)

        with torch.no_grad():
            logits, loss = model(input_ids)
            preds = logits.argmax(-1)
            # calculate accuracy
            c = torch.all(preds[:, 1:] == labels[:, 1:]).float()  # Skip BOS
            correct += c

    print(f"Accuracy: {correct / len(eval_dataset)}")


if __name__ == "__main__":
    main()
