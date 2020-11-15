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

        assert 0 <= args.first_layer_to_use and args.first_layer_to_use < config.num_hidden_layers, "first_layer_to_use must be between 0 and num_hidden_layers-1"
        assert 0 <= args.last_layer_to_use and args.last_layer_to_use < config.num_hidden_layers, "last_layer_to_use must be between 0 and num_hidden_layers-1"
        assert args.first_layer_to_use <= args.last_layer_to_use, "first_layer_to_use must be lower than or equal to last_layer_to_use"

        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels
        # self.use_residual_layer = args.use_residual_layer
        self.layers_to_use = list(range(args.first_layer_to_use, args.last_layer_to_use+1))

        fc_layer_input_size = len(self.layers_to_use) * config.num_attention_heads * config.num_attention_heads * 2
        fc_layer_output_size = 500 # could be something else
        label_classifier_input_size = fc_layer_output_size
        # if self.use_residual_layer:
        #     label_classifier_input_size += fc_layer_input_size

        self.fc_layer = FCLayer(fc_layer_input_size, fc_layer_output_size, args.dropout_rate)
        self.label_classifier = FCLayer(
            label_classifier_input_size,
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
        :param attentions: [batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count]
        :param e_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, max_seq_length, layers_to_use_count, heads_count]
        """
        e_mask_unsqueeze = e_mask
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1, 1]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1, 1, 1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, 1]

        mul_vector = e_mask_unsqueeze.float() * attentions # [batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count]
        sum_vector = torch.sum(mul_vector, dim=1) # [batch_size, max_seq_length, layers_to_use_count, heads_count]
        avg_vector = sum_vector.float() / length_tensor.float() # [batch_size, max_seq_length, layers_to_use_count, heads_count]
        return avg_vector

    @staticmethod
    def attended_entity_average(attentions, e_mask):
        """
        Average the attention vectors (H_i ~ H_j)
        :param attentions: [batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count]
        :param e_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, max_seq_length, layers_to_use_count, heads_count]
        """
        e_mask_unsqueeze = e_mask
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1, 1]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(1)  # [batch_size, 1, max_seq_length, 1, 1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, 1]

        mul_vector = e_mask_unsqueeze.float() * attentions # [batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count]
        sum_vector = torch.sum(mul_vector, dim=2) # [batch_size, max_seq_length, layers_to_use_count, heads_count]
        avg_vector = sum_vector.float() / length_tensor.float() # [batch_size, max_seq_length, layers_to_use_count, heads_count]
        return avg_vector

    @staticmethod
    def entity_to_entity_attentions(attentions, attending_mask, attended_mask):
        """
        Select the attention heads from attending to attended averaging over all tokens for each entity.
        :param attentions: [batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count]
        :param attending_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param attended_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, layers_to_use_count, max_seq_length, heads_count, heads_count]
        """
        attending_avg = RBERT.attending_entity_average(attentions, attending_mask) # batch_size, max_seq_length, layers_to_use_count, heads_count
        attended_avg = RBERT.attended_entity_average(attentions, attended_mask) # batch_size, max_seq_length, layers_to_use_count, heads_count
        attending_avg_unsqueezed = attending_avg.unsqueeze(-1) # batch_size, max_seq_length, layers_to_use_count, heads_count, 1
        attended_avg_unsqueezed = attended_avg.unsqueeze(-2) # batch_size, max_seq_length, layers_to_use_count, 1, heads_count
        e_to_e_attentions = torch.matmul(attending_avg_unsqueezed, attended_avg_unsqueezed) # batch_size, max_seq_length, layers_to_use_count, heads_count, heads_count
        return e_to_e_attentions

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            return_dict=True, output_attentions=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)


        attentions = outputs['attentions'] # tuple of layers_count tensors
                                           # each tensor is of shape batch_size, heads_count, max_seq_length, max_seq_length
        attentions = torch.stack(attentions) # layers_count, batch_size, heads_count, max_seq_length, max_seq_length

        # select layers whose attention heads will be used for classification
        attentions = attentions[self.layers_to_use] # layers_to_use_count, batch_size, heads_count, max_seq_length, max_seq_length

        # swap dimensions
        attentions = attentions.permute(1, 3, 4, 0, 2) # batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count

        # attentions between entities
        e1_to_e2_attentions = self.entity_to_entity_attentions(attentions, e1_mask, e2_mask) # batch_size, max_seq_length, layers_to_use_count, heads_count, heads_count
        e2_to_e1_attentions = self.entity_to_entity_attentions(attentions, e2_mask, e1_mask) # batch_size, max_seq_length, layers_to_use_count, heads_count, heads_count

        # first fc layer
        e1_to_e2_attentions = e1_to_e2_attentions.reshape(e1_to_e2_attentions.shape[0], e1_to_e2_attentions.shape[1], -1) # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count
        e2_to_e1_attentions = e2_to_e1_attentions.reshape(e2_to_e1_attentions.shape[0], e2_to_e1_attentions.shape[1], -1) # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count
        e_to_e_attentions = torch.cat((e1_to_e2_attentions, e2_to_e1_attentions), 2) # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count*2
        e_to_e_fc_output = self.fc_layer(e_to_e_attentions) # batch_size, max_seq_length, fc_layer_output_size

        # fc classifier
        label_classifier_input = torch.amax(e_to_e_fc_output, dim=1) # batch_size, fc_layer_output_size
        # if self.use_residual_layer:
        #     to_be_cat = (e1_to_e2_attentions, e2_to_e1_attentions, e_to_e_attentions)
        #     label_classifier_input = torch.cat(to_be_cat, 2) # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count*2 (*2 if use_residual_layer set to True)
        logits = self.label_classifier(label_classifier_input) # batch_size, num_labels

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
