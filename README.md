# Session 11 


Follow the similar strategy as we did in our baby-steps-code (Links to an external site.), but replace GRU with LSTM. In your code you must:

1. Perform 1 full feed forward step for the encoder manually
2. Perform 1 full feed forward step for the decoder manually.
3. You can use any of the 3 attention mechanisms that we discussed. 


# Encoder 

![image](https://user-images.githubusercontent.com/73247157/126877566-fb24ebcf-1c4e-4bda-a7f4-11e4d3478e64.png)


    embedded_input = embedding(input_tensor[0].view(-1, 1))
    output, (encoder_hidden,encoder_cell) = lstm(embedded_input, (encoder_hidden,encoder_cell))
    encoder_outputs[0] += output[0,0]


    embedded_input = embedding(input_tensor[1].view(-1, 1))
    output, (encoder_hidden,encoder_cell) = lstm(embedded_input, (encoder_hidden,encoder_cell))
    encoder_outputs[1] += output[0,0]
    
 
 Bahdanau attention
 
 ![image](https://user-images.githubusercontent.com/73247157/126877623-16d39525-fb27-4b70-bac7-8437234ea501.png)
 
 
1. Producing the Encoder Hidden States - Encoder produces hidden states of each element in the input sequence
2. Calculating Alignment Scores between the previous decoder hidden state and each of the encoder’s hidden states are calculated (Note: The last encoder hidden state can be used as the first hidden state in the decoder)
3. Softmaxing the Alignment Scores - the alignment scores for each encoder hidden state are combined and represented in a single vector and subsequently softmaxed
4. Calculating the Context Vector - the encoder hidden states and their respective alignment scores are multiplied to form the context vector
5. Decoding the Output - the context vector is concatenated with the previous decoder output and fed into the Decoder RNN for that time step along with the previous decoder hidden state to produce a new output
6.The process (steps 2-5) repeats itself for each time step of the decoder until an token is produced or output is past the specified maximum length
 
 source: https://blog.floydhub.com/attention-mechanism/


# Decoder 

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    output_size = output_lang.n_words
    embedding = nn.Embedding(output_size, 256).to(device)
    embedded = embedding(decoder_input)

    attn_applied ,attn_weights = getattentioncontext(decoder_hidden ,  encoder_outputs , attn_weight_layer = attn_weight_layer)

    input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)
    input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
    lstm = nn.LSTM(256, 256).to(device)
    input_to_lstm = input_to_lstm.unsqueeze(0)
    output, (decoder_hidden, decoder_cell) = lstm(input_to_lstm, (decoder_hidden, decoder_cell))
    output_word_layer = nn.Linear(256, output_lang.n_words).to(device)
    output = F.relu(output)
    output = F.softmax(output_word_layer(output[0]), dim = 1)
    top_value, top_index = output.data.topk(1)
    predicted_sentence.append(output_lang.index2word[top_index.item()])
    output_lang.index2word[top_index.item()],attn_weights
    
    decoder_input = torch.tensor([[nextwordidx]], device=device)
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    output_size = output_lang.n_words
    embedding = nn.Embedding(output_size, 256).to(device)
    embedded = embedding(decoder_input)


    attn_applied ,attn_weights = getattentioncontext(decoder_hidden ,  encoder_outputs , attn_weight_layer = attn_weight_layer)

    input_to_lstm_layer = nn.Linear(256 * 2, 256).to(device)
    input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
    lstm = nn.LSTM(256, 256).to(device)
    input_to_lstm = input_to_lstm.unsqueeze(0)
    output, (decoder_hidden, decoder_cell) = lstm(input_to_lstm, (decoder_hidden, decoder_cell))
    output_word_layer = nn.Linear(256, output_lang.n_words).to(device)
    output = F.relu(output)
    output = F.softmax(output_word_layer(output[0]), dim = 1)
    top_value, top_index = output.data.topk(1)
    predicted_sentence.append(output_lang.index2word[top_index.item()])
    output_lang.index2word[top_index.item()],attn_weights
