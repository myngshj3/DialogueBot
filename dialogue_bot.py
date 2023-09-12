import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LineByLineTextDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForMaskedLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments


class DialogueBot:

    def __init__(self):
        self._config = None
        self._model = None
        self._tokenizer = None
        self.configure()

    @property
    def config(self):
        return self._config
    
    @property
    def model(self) -> AutoModelForCausalLM:
        return self._model
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def configure(self):
        with open("config.json", "r", encoding="utf-8") as f:
            self._config = json.load(f)

    def reconfigure(self):
        self.configure()

    def load_model(self):
        model_name = self._config["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self._tokenizer = tokenizer
        self._model = model
        #self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        #self._model = T5ForConditionalGeneration.from_pretrained(model_name)

    def have_dialogue(self, input_text: str) -> str:
        print("question:", input_text)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt",
                                          max_length=self._config["max_input_length"], truncation=True)
        output = self.model.generate(input_ids, max_length=self._config["max_input_length"],
                                     num_return_sequences=1, pad_token_id=50256)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        #output = self.model.generate(input_ids,
        #                             max_length=self._config["max_output_length"],
        #                             num_return_sequences=1)
        #response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(response + "\n")
            print("output is written.")
        return response
    
    def learn(self, datasets):
        if torch.cuda.is_available():
            model = self.model.to("cuda")
            print("using cuda")
        else:
            print("No GPU. Learning is unavailable.")
            return

        inputs = self.tokenizer([pair[0] for pair in datasets], return_tensors='pt', padding=True, truncation=True)
        labels = self.tokenizer([pair[1] for pair in datasets], return_tensors='pt', padding=True, truncation=True).input_ids

        if torch.cuda.is_available():
            inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
            labels = labels.to("cuda")

        optimizer = torch.optim.Adam(model.parameters())
        epochs = 10

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    def console_dialogue_loop(self):
        while True:
            input_text: str = input("Input question:")
            response: str = self.have_dialogue(input_text)
            print("Bot:", response)

    def train_by_bigtext(self, file_path):
        # データセットの読み込みと前処理
        # テキストファイルからデータを読み込む
        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=file_path,  # テキストファイルのパス
            block_size=self.config["dataset_block_size"],  # ブロックサイズ（トークン数）
        )
        # データコレーターの設定
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # トレーニングアーギュメントの設定
        training_args = TrainingArguments(
            output_dir=self.config["training_output_dir"],
            overwrite_output_dir=self.config["overwrite_output_dir"],
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            save_steps=self.config["training_save_steps"],
            save_total_limit=self.config["training_save_total_limit"],
        )

        # トレーナーの設定
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # ファインチューニングの実行
        trainer.train()

        # ファインチューニングされたモデルの保存
        trainer.save_model()
