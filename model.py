import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from label_smoothing_loss import LabelSmoothingCrossEntropy


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

        self.args = args
        assert 0 <= self.args.first_layer_to_use and self.args.first_layer_to_use < config.num_hidden_layers, "first_layer_to_use must be between 0 and num_hidden_layers-1"
        assert 0 <= self.args.last_layer_to_use and self.args.last_layer_to_use < config.num_hidden_layers, "last_layer_to_use must be between 0 and num_hidden_layers-1"
        assert self.args.first_layer_to_use <= self.args.last_layer_to_use, "first_layer_to_use must be lower than or equal to last_layer_to_use"
        assert not(self.args.skip_1_d1 and self.args.fc1_d1_layer_output_size==0), "residual layer skipping fc1_d1 cannot be set while fc1_d1 is disabled (fc1_d1_layer_output_size==0)"
        assert not(self.args.skip_1_d2 and self.args.fc1_d2_layer_output_size==0), "residual layer skipping fc1_d2 cannot be set while fc1_d2 is disabled (fc1_d2_layer_output_size==0)"
        assert not(self.args.skip_2_d1 and self.args.fc2_layer_output_size==0), "residual layer skipping fc1_d1 and fc2 cannot be set while fc2 is disabled (fc2_layer_output_size==0)"
        assert not(self.args.skip_2_d2 and self.args.fc2_layer_output_size==0), "residual layer skipping fc1_d2 and fc2 cannot be set while fc2 is disabled (fc2_layer_output_size==0)"

        # add args to config so that it is logged in wandb
        # FIXME: there should be a better way to do that (override BertConfig?)
        config.rbert_args = self.args.__dict__
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels
        self.layers_to_use = list(range(self.args.first_layer_to_use, self.args.last_layer_to_use+1))

        # fc layer applied to depth-one attentions
        fc1_d1_layer_input_size = len(self.layers_to_use) * config.num_attention_heads * 2
        if self.args.fc1_d1_layer_output_size > 0:
            fc1_d1_layer_output_size = self.args.fc1_d1_layer_output_size
            self.fc1_d1_layer = FCLayer(fc1_d1_layer_input_size, fc1_d1_layer_output_size, self.args.dropout_rate)
        else:
            fc1_d1_layer_output_size = fc1_d1_layer_input_size

        # fc layer applied to depth-two attentions
        fc1_d2_layer_input_size = len(self.layers_to_use) * config.num_attention_heads * config.num_attention_heads * 2
        if self.args.fc1_d2_layer_output_size > 0:
            fc1_d2_layer_output_size = self.args.fc1_d2_layer_output_size
            self.fc1_d2_layer = FCLayer(fc1_d2_layer_input_size, fc1_d2_layer_output_size, self.args.dropout_rate)
        else:
            fc1_d2_layer_output_size = fc1_d2_layer_input_size

        # fc layer applied to first fc layers output (or to attentions if the latter are disabled)
        fc2_layer_input_size = fc1_d1_layer_output_size + fc1_d2_layer_output_size
        if self.args.skip_1_d1:
            fc2_layer_input_size += fc1_d1_layer_input_size
        if self.args.skip_1_d2:
            fc2_layer_input_size += fc1_d2_layer_input_size
        if self.args.fc2_layer_output_size > 0:
            fc2_layer_output_size = self.args.fc2_layer_output_size
            self.fc2_layer = FCLayer(fc2_layer_input_size, fc2_layer_output_size, self.args.dropout_rate)
        else:
            fc2_layer_output_size = fc2_layer_input_size

        # final layer: label classifier
        label_classifier_input_size = fc2_layer_output_size
        if self.args.skip_2_d1:
            label_classifier_input_size += fc1_d1_layer_input_size
        if self.args.skip_2_d2:
            label_classifier_input_size += fc1_d2_layer_input_size
        self.label_classifier = FCLayer(
            label_classifier_input_size,
            config.num_labels,
            self.args.dropout_rate,
            use_activation=False,
        )


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
    def attended_entity_average_from_averaged_attending_entity(attentions, e_mask):
        """
        Average the attention vectors (H_i ~ H_j)
        :param attentions: [batch_size, max_seq_length, layers_to_use_count, heads_count]
        :param e_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, layers_to_use_count, heads_count]
        """
        e_mask_unsqueeze = e_mask
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1]
        e_mask_unsqueeze = e_mask_unsqueeze.unsqueeze(2)  # [batch_size, max_seq_length, 1, 1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1]

        mul_vector = e_mask_unsqueeze.float() * attentions # [batch_size, max_seq_length, layers_to_use_count, heads_count]
        sum_vector = torch.sum(mul_vector, dim=1) # [batch_size, layers_to_use_count, heads_count]
        avg_vector = sum_vector.float() / length_tensor.float() # [batch_size, layers_to_use_count, heads_count]
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
    def entity_to_entity_attentions_depth1(attentions, attending_mask, attended_mask):
        """
        Select the attention heads from e1 to e2 averaging over all tokens for each entity.
        :param attentions: [batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count]
        :param attending_mask: [batch_size, max_seq_length]
                e.g. attending_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param attended_mask: [batch_size, max_seq_length]
                e.g. attended_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, layers_to_use_count, heads_count]
        """
        e1_avg_vector = RBERT.attending_entity_average(attentions, attending_mask) # [batch_size, max_seq_length, layers_to_use_count, heads_count]
        e2_avg_vector = RBERT.attended_entity_average_from_averaged_attending_entity(e1_avg_vector, attended_mask) #  [batch_size, layers_to_use_count, heads_count]
        return e2_avg_vector

    @staticmethod
    def entity_to_entity_attentions_depth2(attentions, attending_mask, attended_mask):
        """
        Select the attention heads from attending to attended averaging over all tokens for each entity.
        :param attentions: [batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count]
        :param attending_mask: [batch_size, max_seq_length]
                e.g. attending_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param attended_mask: [batch_size, max_seq_length]
                e.g. attended_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, max_seq_length, layers_to_use_count, heads_count, heads_count]
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

        # permute dimensions
        attentions = attentions.permute(1, 3, 4, 0, 2) # batch_size, max_seq_length, max_seq_length, layers_to_use_count, heads_count

        # depth-1 attentions between entities
        e1_to_e2_attentions_d1 = self.entity_to_entity_attentions_depth1(attentions, e1_mask, e2_mask) # batch_size, layers_to_use_count, heads_count
        e2_to_e1_attentions_d1 = self.entity_to_entity_attentions_depth1(attentions, e2_mask, e1_mask) # batch_size, layers_to_use_count, heads_count

        # depth-2 attentions between entities
        e1_to_e2_attentions_d2 = self.entity_to_entity_attentions_depth2(attentions, e1_mask, e2_mask) # batch_size, max_seq_length, layers_to_use_count, heads_count, heads_count
        e2_to_e1_attentions_d2 = self.entity_to_entity_attentions_depth2(attentions, e2_mask, e1_mask) # batch_size, max_seq_length, layers_to_use_count, heads_count, heads_count

        # fc layer applied to depth-1 attentions
        e1_to_e2_attentions_d1 = e1_to_e2_attentions_d1.reshape(e1_to_e2_attentions_d1.shape[0], -1) # batch_size, layers_to_use_count*heads_count
        e2_to_e1_attentions_d1 = e2_to_e1_attentions_d1.reshape(e2_to_e1_attentions_d1.shape[0], -1) # batch_size, layers_to_use_count*heads_count
        attentions_d1 = torch.cat((e1_to_e2_attentions_d1, e2_to_e1_attentions_d1), 1) # batch_size, layers_to_use_count*heads_count*2
        if self.args.fc1_d1_layer_output_size > 0:
            fc1_output_d1 = self.fc1_d1_layer(attentions_d1) # batch_size, fc1_d1_layer_output_size
        else:
            fc1_output_d1 = attentions_d1 # batch_size, layers_to_use_count*heads_count*2 (batch_size, fc1_d1_layer_output_size)

        # fc layer applied to depth-2 attentions with max pooling along tokens
        e1_to_e2_attentions_d2 = e1_to_e2_attentions_d2.reshape(e1_to_e2_attentions_d2.shape[0], e1_to_e2_attentions_d2.shape[1], -1) # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count
        e2_to_e1_attentions_d2 = e2_to_e1_attentions_d2.reshape(e2_to_e1_attentions_d2.shape[0], e2_to_e1_attentions_d2.shape[1], -1) # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count
        attentions_d2 = torch.cat((e1_to_e2_attentions_d2, e2_to_e1_attentions_d2), 2) # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count*2
        if self.args.fc1_d2_layer_output_size > 0:
            fc1_output_d2 = self.fc1_d2_layer(attentions_d2) # batch_size, max_seq_length, fc1_d2_layer_output_size
        else:
            fc1_output_d2 = attentions_d2 # batch_size, max_seq_length, layers_to_use_count*heads_count*heads_count*2 (batch_size, max_seq_length, fc1_d2_layer_output_size)
        fc1_output_d2 = torch.amax(fc1_output_d2, dim=1) # batch_size, fc1_d2_layer_output_size

        # get pooled depth-2 attentions if it needs to be used in a skip connection
        if self.args.skip_1_d2 or self.args.skip_2_d2:
            pooled_attentions_d2 = torch.amax(attentions_d2, dim=1)

        # fc layer applied to first fc layers output (or to attentions if the latter are disabled)
        fc2_input = torch.cat((fc1_output_d1, fc1_output_d2), 1) # batch_size, fc1_d1_layer_output_size+fc1_d2_layer_output_size
        if self.args.skip_1_d1:
            fc2_input = torch.cat((fc2_input, attentions_d1), 1)
        if self.args.skip_1_d2:
            fc2_input = torch.cat((fc2_input, pooled_attentions_d2), 1)
        if self.args.fc2_layer_output_size > 0:
            fc2_output = self.fc2_layer(fc2_input) # batch_size, fc2_layer_output_size
        else:
            fc2_output = fc2_input # batch_size, fc2_layer_output_size
            

        # final layer: label classifier
        label_classifier_input = fc2_output # batch_size, fc2_layer_output_size
        if self.args.skip_2_d1:
            label_classifier_input = torch.cat((label_classifier_input, attentions_d1), 1)
        if self.args.skip_2_d2:
            label_classifier_input = torch.cat((label_classifier_input, pooled_attentions_d2), 1)
        logits = self.label_classifier(label_classifier_input) # batch_size, num_labels

        outputs = (logits,)

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                # loss_fct = nn.MSELoss()
                # loss = loss_fct(logits.view(-1), labels.view(-1))
                raise Exception("Model does not support monoclass classification")
            else:
                if self.config.rbert_args['label_smothing_epsilon'] == 0.0:
                    loss_fct = nn.CrossEntropyLoss()
                else:
                    loss_fct = LabelSmoothingCrossEntropy(self.config.rbert_args['label_smothing_epsilon'])
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits
