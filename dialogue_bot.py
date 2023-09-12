import os
import json
import numpy as np
import subprocess
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./model")
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
    
    def read_markdown_file(self, file_path):
        if file_path.endsWith(".DS_Store"):
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            print("Reading", file_path)
            content = f.read()
        return content
    
    def read_datasets(self, dataset_folder_path):
        entries = ""
        dataset_filenames = os.listdir(dataset_folder_path)
        for filename in dataset_filenames:
            file_path = dataset_folder_path + filename
            print("Processing", file_path)
            markdown_content = self.read_markdown_file(file_path)
            for line in markdown_content.split("\n"):
                if len(line) > 5:
                    entries += line + "\n\n"

        n = len(entries)
        print("Total input entry length", n)
        return entries, n
    
    def learn_from_datasets(self, dataset_folder_path):
        entries, n = self.read_datasets(dataset_folder_path)
        trainEntries = entries[:int(n*0.9)]
        evalEntries = entries[:int(n*0.1)] # 0.9?
        print("Train entries:", trainEntries, ", Eval entries:", evalEntries)

        max_length = 512
        trainTokens = []
        train_text_segments = [trainEntries[i:i+max_length] for i in range(0, len(trainEntries), max_length)]
        for segment in train_text_segments:
            train_segment_token = self.tokenizer.encode(segment, add_special_tokens=True)
            trainTokens.extend(train_segment_token)

        evalTokens = []
        eval_text_segments = [evalEntries[i:i+max_length] for i in range(0, len(evalEntries), max_length)]
        for segment in eval_text_segments:
            eval_segment_tokens = self.tokenizer.encode(segment, add_special_tokens=True)
            evalTokens.extend(eval_segment_tokens)

        print(len(trainTokens), "used for training;", len(evalTokens), "used for eval")

        train_ids = np.array(trainTokens)
        eval_ids = np.array(evalTokens)
        newFolderPath = os.path.join(os.path.dirname(__file__), "TrainingSet")
        if not os.path.exists(newFolderPath):
            os.makedirs(newFolderPath)
        train_ids.tofile(os.path.join(newFolderPath, "train.bin"))
        eval_ids.tofile(os.path.join(newFolderPath, "eval.bin"))

        subprocess.run(["python",
                        "--datasets=TrainingSet",
                        "--n_layer=4",
                        "--n_head=4",
                        "--n_embd=64",
                        "--compile=False",
                        "--eval_iters=1",
                        "--block_size=64",
                        "--batch_size=8",
                        "--device=mps",
                        "--eval_interval=100"])

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
