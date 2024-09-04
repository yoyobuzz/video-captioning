import os
import joblib
import random
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras import Input
from tqdm.notebook import tqdm
import keras

class Vid_cap_Train(object):
    """
    Initialize the parameters for the model
    """

    def __init__(self, config):
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.epochs = config.epochs
        self.latent_dim = config.latent_dim
        self.validation_split = config.validation_split
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.time_steps_decoder = None
        self.x_data = {}

        # processed data
        self.tokenizer = None

        # models
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path

    def preprocessing(self):
        files = os.listdir("Data/Training Videos/feat")
        for i in range(len(files)):
            files[i] = files[i][:-8]

        BASE_DIR = "Data"
        TRAIN_FEATURE_DIR = os.path.join(BASE_DIR,"Training Videos/feat")
        features = {}
        for file in files:
            f = np.load(os.path.join(TRAIN_FEATURE_DIR, file + ".avi.npy"), allow_pickle=True)
            features[file] = f


        # create mapping of image to captions
        mapping = {}
        # process lines
        for line in tqdm(captions_doc.split('\n')):
            #split the line by comma(,)
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            caption[0] = caption[0][1:]
            #remove extension from image id
            image_id = image_id.split('.')[0]
            # convert caption list to string
            caption = " ".join(caption)
            # create list if needed
            if image_id in files:
                if image_id not in mapping:
                    mapping[image_id] = []
                mapping[image_id].append(caption)

        all_captions = []
        for key in mapping:
            for caption in mapping[key]:
                all_captions.append(caption)

        # Train Test Split
        image_ids = list(mapping.keys())
        #print(image_ids)
        random.shuffle(image_ids)
        split = int(len(image_ids)*validation_split)
        training_list = image_ids[split:]
        validation_list = image_ids[:split]



        # tokenizer
        training_captions = []

        for id in training_list:
            for caption in mapping[id]:
                training_captions.append(caption)

        tokenizer = Tokenizer(num_words= num_decoder_tokens)
        tokenizer.fit_on_texts(training_captions)
        vocab_size =len(tokenizer.word_index)+1

        return training_list, validation_list, mapping, tokenizer, features



    def data_loader(training_list,mapping,features,tokenizer,self):
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []
        videoId = []
        videoSeq = []
        for id in training_list:
            captions = mapping[id]
            for caption in captions:
                videoId.append(id)
                videoSeq.append(caption)

        train_sequences = tokenizer.texts_to_sequences(videoSeq)
        train_sequences = np.array(train_sequences, dtype=object)
        train_sequences = pad_sequences(train_sequences, padding='post', truncating='post',
                                        maxlen=max_length)

        file_size = len(train_sequences)
        n = 0
        for i in range(epochs):
            for idx in range(0, file_size):
                n += 1
                encoder_input_data.append(features[videoId[idx]])
                y = to_categorical(train_sequences[idx], num_decoder_tokens)
                decoder_input_data.append(y[:-1])
                decoder_target_data.append(y[1:])
                if n == batch_size or idx == file_size-1:
                    encoder_input = np.array(encoder_input_data)
                    decoder_input = np.array(decoder_input_data)
                    decoder_target = np.array(decoder_target_data)
                    encoder_input_data = []
                    decoder_input_data = []
                    decoder_target_data = []
                    n = 0
                    yield ([encoder_input, decoder_input], decoder_target)



    def train_model(self):
        """
        an encoder decoder sequence to sequence model
        reference : https://arxiv.org/abs/1505.00487
        """
        encoder_inputs = Input(shape=(time_steps_encoder, num_encoder_tokens), name="encoder_inputs")
        encoder = LSTM(latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]


        decoder_inputs = Input(shape=(time_steps_decoder, num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='relu', name='decoder_relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        #   model.summary()
        
        training_list, validation_list, mapping, tokenizer, features = self.preprocessing()

        train = data_loader(training_list,mapping,features,tokenizer)
        valid = data_loader(validation_list,mapping,features,tokenizer,max_length,epochs,num_decoder_tokens,batch_size)

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')

        # Run training
        opt = keras.optimizers.Adam(lr=0.0003)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                        factor=0.1, patience=5, verbose=0,
                                                        mode="auto")
        model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')

        validation_steps = len(validation_list)//batch_size
        steps_per_epoch = len(training_list)//batch_size

        model.fit(train, validation_data=valid, validation_steps=validation_steps,
                    epochs=epochs, steps_per_epoch=steps_per_epoch,
                    callbacks=[reduce_lr, early_stopping])

        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        #encoder_model.summary()
        #decoder_model.summary()

        # saving the models
        encoder_model.save(os.path.join(save_model_path, 'encoder_model.h5'))
        decoder_model.save_weights(os.path.join(save_model_path, 'decoder_model_weights.h5'))
        with open(os.path.join(save_model_path, 'tokenizer' + str(num_decoder_tokens)), 'wb') as file:
            joblib.dump(tokenizer, file)


if __name__ == "__main__":
    video_to_text = Vid_cap_Train(config)
    video_to_text.train_model()
