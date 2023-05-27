from utils.modeling_norbert import NorbertForTokenClassification


model = NorbertForTokenClassification.from_pretrained(
    "ltg/norbert3-small", 
    num_labels=15,
    num_hidden_layers=12,
    hidden_dropout_prob=0.1,
)

# print(model)
print(model.base_model)

print(model.base_model.embedding)
print(model.base_model.transformer)