import json
import re
from tqdm import tqdm

class EquasionTokenizer:
	def __init__(self, vocab_file="tokens.json", dataset_file="data.txt"):
		self.token_to_id = {}
		self.id_to_token = {}
		self.vocab_file = vocab_file
		self.data_file = dataset_file

	def build_vocab(self):
		with open("data.txt", "r") as file:
			equasions = [line.strip() for line in file.readlines()]
		split_equasions = []
		for equasion in equasions:
			split_equasions.append(self.split_equasion(equasion))
		all_tokens = [token for equation in split_equasions for token in equation]
		unique_tokens = set(all_tokens)
		self.token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
		self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
		with open(self.vocab_file, "w") as vocab_file:
			json.dump({
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token
            }, vocab_file, indent=4)
		
	def split_equasion(self, equasion):
		pattern = r"<.{3}>|."

		tokens = re.findall(pattern, equasion)

		tokens = [token for token in tokens if token]

		while len(tokens) < 30:
			tokens.append("<PAD>")
		return tokens
	def load_vocab(self):
		with open(self.vocab_file, "r") as vocab_file:
			vocab = json.load(vocab_file)
			self.token_to_id = vocab["token_to_id"]
			self.id_to_token = vocab["id_to_token"]
		print("Vocabulary loaded successfully.")
	def encode(self, equasion):
		tokens = self.split_equasion(equasion)
		return [self.token_to_id.get(token) for token in tokens]
	def decode(self, tokens):
		return [self.id_to_token.get(token) for token in tokens]
	
if __name__ == "__main__":
	tokenizer = EquasionTokenizer()
	tokenizer.load_vocab()
	with open("data.txt", "r") as file, open("data.jsonl", "w") as jsonl_file:
		for line in tqdm(file, desc="Tokeinizing Lines", unit="line"):
			line = line.strip()
			encoded = tokenizer.encode(line)
			json.dump(encoded, jsonl_file)
			jsonl_file.write("\n")

