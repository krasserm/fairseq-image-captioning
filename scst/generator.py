import torch
import numpy as np


class SimpleSequenceGenerator:
    """Simple beam search implementation that supports back-propagation.

       Used for self-critical sequence training (SCST) only. Will be later
       extended to supports diverse beam search.
    """
    def __init__(self,
                 beam=5,
                 penalty=1.0,
                 max_pos=1024,
                 eos_index=2):

        self.beam = beam
        self.penalty = penalty
        self.max_pos = max_pos
        self.eos_index = eos_index

    def generate(self, model, sample):
        incremental_state = {}

        sample_ids = sample['id']
        sample_size = len(sample_ids)
        device = sample_ids.device

        sample_features = sample['net_input']['src_tokens'].to(device=device)
        sample_length = sample['net_input']['src_lengths']
        sample_location = sample['net_input']['src_locations'].to(device=device)

        tokens = torch.ones(sample_size, 1, dtype=torch.long).to(device=device) * self.eos_index
        enc_out = model.encoder(sample_features, sample_length, sample_location)
        dec_out, extra = model.decoder(tokens, encoder_out=enc_out, incremental_state=incremental_state)

        lprobs = model.get_normalized_probs((dec_out, extra), log_probs=True)
        lprobs, tokens = torch.topk(lprobs, k=self.beam, dim=2)
        lprobs = lprobs.transpose(2, 1)  # (n, b, l)

        tokens = tokens.transpose(2, 1)  # (n, b, l+1)
        tokens = torch.cat([torch.ones(sample_size, self.beam, 1, dtype=torch.long).to(device=device) * self.eos_index, tokens], dim=2)

        mask = torch.zeros(sample_size, self.beam, 3, dtype=torch.bool, device=device)  # (n, b, l+2)
        mask[:, :, :] = True

        new_order = torch.tensor(np.repeat(range(sample_size), self.beam)).to(device=device)
        enc_out = model.encoder.reorder_encoder_out(enc_out, new_order)
        incremental_state = model.decoder.reorder_incremental_state(incremental_state, new_order)

        for _ in range(self.max_pos):
            tokens_batch = tokens.flatten(0, 1)  # (n x b, l)
            dec_out, extra = model.decoder(tokens_batch, encoder_out=enc_out, incremental_state=incremental_state)

            lprobs_batch = model.get_normalized_probs((dec_out, extra), log_probs=True)
            lprobs_batch = lprobs_batch[:, -1, :]  # (n x b, v)
            lprobs_batch = lprobs_batch.reshape(tokens.shape[0], tokens.shape[1], -1)  # (n, b, v)
            lprobs_k, tokens_k = torch.topk(lprobs_batch, k=self.beam, dim=2)  # (n, b, b)

            tokens_repeated = torch.repeat_interleave(tokens, self.beam, dim=1)
            tokens_k_flattened = tokens_k.flatten().view(sample_size, -1, 1)
            tokens_cat = torch.cat([tokens_repeated, tokens_k_flattened], dim=2)

            mask_repeated = torch.repeat_interleave(mask, self.beam, dim=1)
            mask_k_flattened = (tokens_k_flattened != self.eos_index) & mask_repeated[:, :, -1:]
            mask_cat = torch.cat([mask_repeated, mask_k_flattened], dim=2)

            lprobs_repeated = torch.repeat_interleave(lprobs, self.beam, dim=1)
            lprobs_k_flattened = lprobs_k.flatten().view(sample_size, -1, 1)
            lprobs_cat = torch.cat([lprobs_repeated, lprobs_k_flattened], dim=2)
            lprobs_cat_masked = lprobs_cat * mask_cat[:, :, 1:-1]

            num_tokens = torch.sum(mask_cat[:, :, 1:-1], dim=2)
            scores = torch.sum(lprobs_cat_masked, dim=2) / num_tokens ** self.penalty
            scores_mask = torch.zeros(sample_size, self.beam * self.beam, dtype=torch.bool)

            # Make sure that completed captions can't be
            # selected more than once (avoids duplicates)
            for i in range(sample_size):
                for j in range(self.beam):
                    first = j * self.beam
                    start = first + 1
                    end = first + self.beam
                    scores_mask[i, start:end] = torch.sum(mask_cat[i, first:end, -1]) == 0

            for i in range(sample_size):
                scores[i][scores_mask[i]] = -1e8

            top_values, top_indices = torch.topk(scores, k=self.beam)
            incremental_state = model.decoder.reorder_incremental_state(incremental_state=incremental_state,
                                                                        new_order=top_indices.flatten() // self.beam)

            tokens_list = []
            lprobs_list = []
            mask_list = []

            for i in range(sample_size):
                tokens_selected = tokens_cat[i][top_indices[i]]
                tokens_list.append(tokens_selected)

                lprobs_selected = lprobs_cat[i][top_indices[i]]
                lprobs_list.append(lprobs_selected)

                mask_selected = mask_cat[i][top_indices[i]]
                mask_list.append(mask_selected)

            tokens = torch.stack(tokens_list, dim=0)
            lprobs = torch.stack(lprobs_list, dim=0)
            mask = torch.stack(mask_list, dim=0)

            if torch.sum(mask[:, :, -1]) == 0:
                break

        result_mask = mask[:, :, 1:-1]
        result_tokens = tokens[:, :, 1:] * result_mask
        result_lprobs = lprobs * result_mask

        result_num_tokens = torch.sum(result_mask, dim=2)
        result_scores = torch.sum(result_lprobs, dim=2) / result_num_tokens ** self.penalty

        return result_scores, result_lprobs, result_tokens, result_mask
