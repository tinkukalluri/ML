import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)


word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])


# we padded each sentence to have exactly 250 char max length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


model = keras.Sequential()
# embedding and gloabal averafepolling1D is not ur regular
# ANN layers they are totally different watch the video below to understand
# https://www.youtube.com/watch?v=qpb_39IjZA0&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj&index=6
# https://www.youtube.com/watch?v=5PL0TmQhItY
# https://embeddings.macheads101.com/word?word=pizza this website use word embedding
# that we used in this model...
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# print(model.summary())  # prints a summary of the model

model.compile(optimizer="adam" , loss="binary_crossentropy" , metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
# model.save("model.h5")  # name it whatever you want but end with .h5
model = keras.models.load_model("model.h5")

# results = model.evaluate(test_data, test_labels)
# print(results)


# predict= model.predict([test_data[0]])






def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded


def input_review(s="really loved the movie the actors did a great job the production and everything was "):
	my_review = s
	nline = my_review.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
	print("nline::", nline)
	encoded_data = review_encode(nline)
	encode = keras.preprocessing.sequence.pad_sequences([encoded_data], value=word_index["<PAD>"], padding="post",
														maxlen=250)  # make the data 250 words long
	predict = model.predict(encode)
	print(decode_review(encode[0]))
	print(encode)
	print(predict[0])


def input_file():
	with open("test.txt", encoding="utf-8") as f:
		for line in f.readlines():
			nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
			encode = review_encode(nline)
			encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
			predict = model.predict(encode)
			print(line)
			print(encode)
			print(predict[0])



input_review()
input_review("hated it pathetic film")