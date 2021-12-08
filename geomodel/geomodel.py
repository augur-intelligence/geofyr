import torch

class GeoModel():
    def __init__(self, model_string, tokenizer_string, tokenizer_class, max_seq_length):
        self.model_string = model_string
        self.tokenizer_string = tokenizer_string
        self.tokenizer_class = tokenizer_class
        self.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_string)
        self.model = torch.load(
            self.model_string,
            map_location=torch.device('cpu'))
        self.max_seq_lengt = max_seq_length

    def forward(self, text):
        self.current_text = text
        tokenized_text = self.tokenizer(
            self.current_text,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=self.max_seq_lengt)

        self.model.eval()
        with torch.no_grad():
            self.output = self.model.forward(
                input_ids=tokenized_text['input_ids'],
                attention_mask=tokenized_text['attention_mask'],
                output_hidden_states=True)

    def predict_point(self):
        return self.output['logits'].numpy().squeeze()

    def predict_areas(self, num_layers):
        return self.output['logits'].numpy().squeeze()