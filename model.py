import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels

        e_to_e_size = config.num_hidden_layers * config.num_attention_heads

        self.e1_to_e2_fc_layer = FCLayer(e_to_e_size, e_to_e_size, args.dropout_rate)
        self.e2_to_e1_fc_layer = FCLayer(e_to_e_size, e_to_e_size, args.dropout_rate)
        self.label_classifier = FCLayer(
            e_to_e_size*2,
            config.num_labels,
            args.dropout_rate,
            use_activation=False,
        )

    # @staticmethod
    # def entity_average(hidden_output, e_mask):
    #     """
    #     Average the entity hidden state vectors (H_i ~ H_j)
    #     :param hidden_output: [batch_size, j-i+1, dim]
    #     :param e_mask: [batch_size, max_seq_len]
    #             e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
    #     :return: [batch_size, dim]
    #     """
    #     e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
    #     length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

    #     # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
    #     sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
    #     avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    #     return avg_vector

    @staticmethod
    def attending_entity_average(attentions, e_mask):
        """
        Average the attention vectors (H_i ~ H_j)
        :param attentions: [layers_count, batch_size, heads_count, max_seq_length, max_seq_length]
        :param e_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [layers_count, batch_size, heads_count, max_seq_length]
        """
        e_mask_unsqueeze = e_mask
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(1)  # [batch_size, 1, max_seq_length, 1]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(0)  # [1, batch_size, 1, max_seq_length, 1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(0)  # [1, batch_size, 1, 1]

        mul_vector = e_mask_unsqueeze.float() * attentions # [layers_count, batch_size, heads_count, max_seq_length, max_seq_length]
        sum_vector = torch.sum(mul_vector, dim=3) # [layers_count, batch_size, heads_count, max_seq_length]
        avg_vector = sum_vector.float() / length_tensor.float() # [layers_count, batch_size, heads_count, max_seq_length]
        return avg_vector

    @staticmethod
    def attended_entity_average(attentions, e_mask):
        """
        Average the attention vectors (H_i ~ H_j)
        :param attentions: [layers_count, batch_size, heads_count, max_seq_length]
        :param e_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [layers_count, batch_size, heads_count]
        """
        e_mask_unsqueeze = e_mask
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(1)  # [batch_size, 1, max_seq_length]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(0)  # [1, batch_size, 1, max_seq_length]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(0)  # [1, batch_size, 1, 1]

        mul_vector = e_mask_unsqueeze.float() * attentions # [layers_count, batch_size, heads_count, max_seq_length]
        sum_vector = torch.sum(mul_vector, dim=3) # [layers_count, batch_size, heads_count]
        avg_vector = sum_vector.float() / length_tensor.float() # [layers_count, batch_size, heads_count]
        return avg_vector

    @staticmethod
    def entity_to_entity_attentions(attentions, e1_mask, e2_mask):
        """
        Select the attention heads from e1 to e2 averaging over all tokens for each entity.
        :param attentions: [layers_count, batch_size, heads_count, max_seq_length, max_seq_length]
        :param e1_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param e2_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [layers_count, batch_size, heads_count]
        """
        e1_avg_vector = RBERT.attending_entity_average(attentions, e1_mask)
        e2_avg_vector = RBERT.attended_entity_average(e1_avg_vector, e2_mask)
        return e2_avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            return_dict=True, output_attentions=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)


        attentions = outputs['attentions'] # tuple of layers_count tensors
                                           # each tensor is of shape batch_size, heads_count, max_seq_length, max_seq_length
        attentions = torch.stack(attentions) # layers_count, batch_size, heads_count, max_seq_length, max_seq_length

        # Average
        e1_to_e2_attentions = self.entity_to_entity_attentions(attentions, e1_mask, e2_mask) # layers_count, batch_size, heads_count
        e2_to_e1_attentions = self.entity_to_entity_attentions(attentions, e2_mask, e1_mask) # layers_count, batch_size, heads_count

        # swap dimensions (we could also use transpose)
        e1_to_e2_attentions = e1_to_e2_attentions.permute(1, 0, 2) # batch_size, layers_count, heads_count
        e2_to_e1_attentions = e2_to_e1_attentions.permute(1, 0, 2) # batch_size, layers_count, heads_count

        # flatten and first fc layers
        e1_to_e2_attentions = e1_to_e2_attentions.reshape(e1_to_e2_attentions.shape[0], -1) # batch_size, layers_count*heads_count
        e2_to_e1_attentions = e2_to_e1_attentions.reshape(e2_to_e1_attentions.shape[0], -1) # batch_size, layers_count*heads_count
        e1_to_e2_attentions = self.e1_to_e2_fc_layer(e1_to_e2_attentions) # batch_size, layers_count*heads_count
        e2_to_e1_attentions = self.e2_to_e1_fc_layer(e2_to_e1_attentions) # batch_size, layers_count*heads_count

        # concat and fc classifier
        e_to_e_attentions = torch.cat((e1_to_e2_attentions, e2_to_e1_attentions), 1) # batch_size, layers_count*heads_count*2
        logits = self.label_classifier(e_to_e_attentions)

        outputs = (logits,)

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits
