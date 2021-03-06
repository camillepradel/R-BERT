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
        assert 0 <= self.args.first_layer_d1 and self.args.first_layer_d1 <= config.num_hidden_layers-1, "first_layer_d1 must be between 0 and num_hidden_layers-1"
        assert 0 <= self.args.last_layer_d1 and self.args.last_layer_d1 <= config.num_hidden_layers-1, "last_layer_d1 must be between 0 and num_hidden_layers-1"
        assert self.args.first_layer_d1 <= self.args.last_layer_d1, "first_layer_d1 must be lower than or equal to last_layer_d1"
        assert 0 <= self.args.first_layer_d2 and self.args.first_layer_d2 <= config.num_hidden_layers-2, "first_layer_d2 must be between 0 and num_hidden_layers-2"
        assert 0 <= self.args.second_to_last_layer_d2 and self.args.second_to_last_layer_d2 <= config.num_hidden_layers-2, "second_to_last_layer_d2 must be between 0 and num_hidden_layers-2"
        assert self.args.first_layer_d2 <= self.args.second_to_last_layer_d2, "first_layer_d2 must be lower than or equal to second_to_last_layer_d2"
        assert not(self.args.skip_1_d1 and self.args.fc1_d1_layer_output_size==0), "residual layer skipping fc1_d1 cannot be set while fc1_d1 is disabled (fc1_d1_layer_output_size==0)"
        assert not(self.args.skip_1_d2 and self.args.fc1_d2_layer_output_size==0), "residual layer skipping fc1_d2 cannot be set while fc1_d2 is disabled (fc1_d2_layer_output_size==0)"
        assert not(self.args.skip_2_d1 and self.args.fc2_layer_output_size==0), "residual layer skipping fc1_d1 and fc2 cannot be set while fc2 is disabled (fc2_layer_output_size==0)"
        assert not(self.args.skip_2_d2 and self.args.fc2_layer_output_size==0), "residual layer skipping fc1_d2 and fc2 cannot be set while fc2 is disabled (fc2_layer_output_size==0)"

        # add args to config so that it is logged in wandb
        # FIXME: there should be a better way to do that (override BertConfig?)
        config.rbert_args = self.args.__dict__
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels
        self.layers_d1 = list(range(self.args.first_layer_d1, self.args.last_layer_d1+1))
        self.layers_d2 = [l for l in (range(self.args.first_layer_d2, self.args.second_to_last_layer_d2+1))]

        self.entity_pooling_mode = 'avg' if self.args.entity_mention.startswith('avg_pooling') else 'max'

        # fc layer applied to depth-1 attentions
        fc1_d1_layer_input_size = len(self.layers_d1) * config.num_attention_heads * 2
        if self.args.fc1_d1_layer_output_size > 0:
            fc1_d1_layer_output_size = self.args.fc1_d1_layer_output_size
            self.fc1_d1_layer = FCLayer(fc1_d1_layer_input_size, fc1_d1_layer_output_size, self.args.dropout_rate)
        else:
            fc1_d1_layer_output_size = fc1_d1_layer_input_size

        # fc layer applied to depth-2 attentions
        fc1_d2_layer_input_size = len(self.layers_d2) * config.num_attention_heads * config.num_attention_heads * 2
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
    def entity_attentions_pooling(attentions, e_mask, entity_pooling_mode, role):
        """
        Pool (avg or max depending on entity_pooling_mode) the attention vectors (H_i ~ H_j)
        :param attentions: [batch_size, max_seq_length, max_seq_length, layers_count, heads_count]
        :param e_mask: [batch_size, max_seq_length]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param entity_pooling_mode: which pool function to use
        :param role: role of the layer on wich we want to pool (possible values: 'attending' and 'attended')
        :return: [batch_size, max_seq_length, layers_count, heads_count]
        """
        dim_to_squeeze, dim_to_keep = (1, 2) if role == 'attending' else (2, 1)
        e_mask_unsqueezed = e_mask.unsqueeze(2).unsqueeze(2).unsqueeze(dim_to_keep)  # [batch_size, max_seq_length, 1, 1, 1] if attending
                                                                                    # [batch_size, 1, max_seq_length, 1, 1] if attended
        masked_attentions = e_mask_unsqueezed.float() * attentions # [batch_size, max_seq_length, max_seq_length, layers_count, heads_count]

        if entity_pooling_mode == 'max':
            pooled_vector = torch.max(masked_attentions, dim=dim_to_squeeze).values # [batch_size, max_seq_length, layers_count, heads_count]
        else: # avg pooling
            length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, 1]
            sum_vector = torch.sum(masked_attentions, dim=dim_to_squeeze) # [batch_size, max_seq_length, layers_count, heads_count]
            pooled_vector = sum_vector.float() / length_tensor.float() # [batch_size, max_seq_length, layers_count, heads_count]

        return pooled_vector

    @staticmethod
    def entity_to_entity_attentions_depth1(attentions, attending_mask, attended_mask, entity_pooling_mode):
        """
        Select the attention heads from e1 to e2 pooling over all tokens for each entity.
        :param attentions: [batch_size, max_seq_length, max_seq_length, layers_d1_count, heads_count]
        :param attending_mask: [batch_size, max_seq_length]
                e.g. attending_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param attended_mask: [batch_size, max_seq_length]
                e.g. attended_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param entity_pooling_mode: which pool function to use
        :return: [batch_size, layers_d1_count, heads_count]
        """
        attending_mask_unsqueezed = attending_mask.unsqueeze(2).unsqueeze(2).unsqueeze(2) # [batch_size, max_seq_length, 1, 1, 1]
        attended_mask_unsqueezed = attended_mask.unsqueeze(2).unsqueeze(2).unsqueeze(2) # [batch_size, max_seq_length, 1, 1, 1]
        
        masked_attentions = attended_mask_unsqueezed.float() * (attending_mask_unsqueezed.float() * attentions) # [batch_size, max_seq_length, max_seq_length, layers_d1_count, heads_count]

        if entity_pooling_mode == 'max':
            pooled_vector = masked_attentions.max(dim=1).values.max(dim=1).values # [batch_size, layers_d1_count, heads_count]
        else: # avg pooling
            length_attending_entity = (attending_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1]
            length_attended_entity = (attended_mask != 0).sum(dim=1).unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1]
            length_non_zeros = length_attending_entity * length_attended_entity # [batch_size, 1, 1, 1]
            sum_vector = masked_attentions.sum(dim=1).sum(dim=1) # [batch_size, layers_d1_count, heads_count]
            pooled_vector = sum_vector.float() / length_non_zeros.float() # [batch_size, layers_d1_count, heads_count]

        return pooled_vector

    @staticmethod
    def entity_to_entity_attentions_depth2(attentions, attending_mask, attended_mask, entity_pooling_mode):
        """
        Select the attention heads from attending to attended averaging over all tokens for each entity.
        :param attentions: [batch_size, max_seq_length, max_seq_length, (layers_d2_count+1), heads_count]
        :param attending_mask: [batch_size, max_seq_length]
                e.g. attending_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param attended_mask: [batch_size, max_seq_length]
                e.g. attended_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :param entity_pooling_mode: which pool function to use
        :return: [batch_size, max_seq_length, layers_d2_count, heads_count, heads_count]
        """
        attending_avg = RBERT.entity_attentions_pooling(attentions[:, :, :, 1:], attending_mask, entity_pooling_mode, role='attending') # batch_size, max_seq_length, layers_d2_count, heads_count
        attended_avg = RBERT.entity_attentions_pooling(attentions[:, :, :, :-1], attended_mask, entity_pooling_mode, role='attended') # batch_size, max_seq_length, layers_d2_count, heads_count
        attending_avg_unsqueezed = attending_avg.unsqueeze(-1) # batch_size, max_seq_length, layers_d2_count, heads_count, 1
        attended_avg_unsqueezed = attended_avg.unsqueeze(-2) # batch_size, max_seq_length, layers_d2_count, 1, heads_count
        e_to_e_attentions = torch.matmul(attending_avg_unsqueezed, attended_avg_unsqueezed) # batch_size, max_seq_length, layers_d2_count, heads_count, heads_count
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
        attentions_d1 = attentions[self.layers_d1] # layers_d1_count, batch_size, heads_count, max_seq_length, max_seq_length
        attentions_d2 = attentions[self.layers_d2 + [self.args.second_to_last_layer_d2+1]] # (layers_d2_count+1), batch_size, heads_count, max_seq_length, max_seq_length

        # permute dimensions
        attentions_d1 = attentions_d1.permute(1, 3, 4, 0, 2) # batch_size, max_seq_length, max_seq_length, layers_d1_count, heads_count
        attentions_d2 = attentions_d2.permute(1, 3, 4, 0, 2) # batch_size, max_seq_length, max_seq_length, (layers_d2_count+1), heads_count

        # depth-1 attentions between entities
        e1_to_e2_attentions_d1 = self.entity_to_entity_attentions_depth1(attentions_d1, e1_mask, e2_mask, self.entity_pooling_mode) # batch_size, layers_d1_count, heads_count
        e2_to_e1_attentions_d1 = self.entity_to_entity_attentions_depth1(attentions_d1, e2_mask, e1_mask, self.entity_pooling_mode) # batch_size, layers_d1_count, heads_count

        # depth-2 attentions between entities
        e1_to_e2_attentions_d2 = self.entity_to_entity_attentions_depth2(attentions_d2, e1_mask, e2_mask, self.entity_pooling_mode) # batch_size, max_seq_length, layers_d2_count, heads_count, heads_count
        e2_to_e1_attentions_d2 = self.entity_to_entity_attentions_depth2(attentions_d2, e2_mask, e1_mask, self.entity_pooling_mode) # batch_size, max_seq_length, layers_d2_count, heads_count, heads_count

        # fc layer applied to depth-1 attentions
        e1_to_e2_attentions_d1 = e1_to_e2_attentions_d1.reshape(e1_to_e2_attentions_d1.shape[0], -1) # batch_size, layers_d1_count*heads_count
        e2_to_e1_attentions_d1 = e2_to_e1_attentions_d1.reshape(e2_to_e1_attentions_d1.shape[0], -1) # batch_size, layers_d1_count*heads_count
        attentions_d1 = torch.cat((e1_to_e2_attentions_d1, e2_to_e1_attentions_d1), 1) # batch_size, layers_d1_count*heads_count*2
        if self.args.fc1_d1_layer_output_size > 0:
            fc1_output_d1 = self.fc1_d1_layer(attentions_d1) # batch_size, fc1_d1_layer_output_size
        else:
            fc1_output_d1 = attentions_d1 # batch_size, layers_d1_count*heads_count*2 (batch_size, fc1_d1_layer_output_size)

        # fc layer applied to depth-2 attentions with max pooling along tokens
        e1_to_e2_attentions_d2 = e1_to_e2_attentions_d2.reshape(e1_to_e2_attentions_d2.shape[0], e1_to_e2_attentions_d2.shape[1], -1) # batch_size, max_seq_length, layers_d2_count*heads_count*heads_count
        e2_to_e1_attentions_d2 = e2_to_e1_attentions_d2.reshape(e2_to_e1_attentions_d2.shape[0], e2_to_e1_attentions_d2.shape[1], -1) # batch_size, max_seq_length, layers_d2_count*heads_count*heads_count
        attentions_d2 = torch.cat((e1_to_e2_attentions_d2, e2_to_e1_attentions_d2), 2) # batch_size, max_seq_length, layers_d2_count*heads_count*heads_count*2
        if self.args.fc1_d2_layer_output_size > 0:
            fc1_output_d2 = self.fc1_d2_layer(attentions_d2) # batch_size, max_seq_length, fc1_d2_layer_output_size
        else:
            fc1_output_d2 = attentions_d2 # batch_size, max_seq_length, layers_d2_count*heads_count*heads_count*2 (batch_size, max_seq_length, fc1_d2_layer_output_size)
        fc1_output_d2 = torch.amax(fc1_output_d2, dim=1) # batch_size, fc1_d2_layer_output_size

        # get pooled depth-2 attentions if it needs to be used in a skip connection
        if self.args.skip_1_d2 or self.args.skip_2_d2:
            pooled_attentions_d2 = torch.amax(attentions_d2, dim=1) # batch_size, layers_d2_count*heads_count*heads_count*2

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
                if self.config.rbert_args['label_smoothing_epsilon'] == 0.0:
                    loss_fct = nn.CrossEntropyLoss()
                else:
                    loss_fct = LabelSmoothingCrossEntropy(self.config.rbert_args['label_smoothing_epsilon'])
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits
