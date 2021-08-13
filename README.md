# Session 14 - BERT and BART

Team_members - Bharath,Dinesh 


# Task 1 

BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre- train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a re- sult, the pre-trained BERT model can be fine- tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task- specific architecture modifications.

BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. 

## Training Logs

    ***** Running training *****
      Num examples = 144262
      Num Epochs = 1
      Batch size = 16
      Total optimization steps = 45079
    Epoch: 100%
    1/1 [25:26<00:00, 1526.68s/it]
    Iteration: 100%
    1803/1803 [25:26<00:00, 1.19it/s]
    /usr/local/lib/python3.7/dist-packages/pytorch_transformers/optimization.py:166: UserWarning: This overload of add_ is deprecated:
        add_(Number alpha, Tensor other)
    Consider using one of the following signatures instead:
        add_(Tensor other, *, Number alpha) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:1025.)
      exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
    Train loss: 1.7028252339065075


## Output

    question       >> What form of poetry was developed in the Yuan?
    model's answer >> the qu
    
    question       >> Which American show changed the views of Romanians during the cold war?
    model's answer >> Dallas
    
    question       >>  What entity has taken the view that the goals of free trade are underpinned by the aims to improve people's well being?
    model's answer >> the Court of Justice
    
    question       >> Who received the first steam engine patent?
    model's answer >> The Spanish inventor Jerónimo de Ayanz y Beaumont
    
    question       >> On what date did the first railway trip in the world occur?
    model's answer >> 21 February 1804
    
    question       >> What method is used to intuitively assess or quantify the amount of resources required to solve a computational problem?
    model's answer >> mathematical models of computation

    question       >> What are leukocytes the first arm of?
    model's answer >> the innate immune system

    question       >> Where was an elected assembly to be set up, under the terms of the Scotland Act of 1978?
    model's answer >> Edinburgh

# Task 2
## Training Logs

    ======== Epoch 1 / 4 ========
    Training...
      Batch    40  of    241.    Elapsed: 0:00:13.
      Batch    80  of    241.    Elapsed: 0:00:27.
      Batch   120  of    241.    Elapsed: 0:00:40.
      Batch   160  of    241.    Elapsed: 0:00:54.
      Batch   200  of    241.    Elapsed: 0:01:08.
      Batch   240  of    241.    Elapsed: 0:01:22.

      Average training loss: 0.50
      Training epcoh took: 0:01:22

    Running Validation...
      Accuracy: 0.78
      Validation Loss: 0.48
      Validation took: 0:00:03

    ======== Epoch 2 / 4 ========
    Training...
      Batch    40  of    241.    Elapsed: 0:00:14.
      Batch    80  of    241.    Elapsed: 0:00:28.
      Batch   120  of    241.    Elapsed: 0:00:42.
      Batch   160  of    241.    Elapsed: 0:00:56.
      Batch   200  of    241.    Elapsed: 0:01:11.
      Batch   240  of    241.    Elapsed: 0:01:25.

      Average training loss: 0.30
      Training epcoh took: 0:01:26

    Running Validation...
      Accuracy: 0.81
      Validation Loss: 0.46
      Validation took: 0:00:03

    ======== Epoch 3 / 4 ========
    Training...
      Batch    40  of    241.    Elapsed: 0:00:15.
      Batch    80  of    241.    Elapsed: 0:00:29.
      Batch   120  of    241.    Elapsed: 0:00:44.
      Batch   160  of    241.    Elapsed: 0:00:58.
      Batch   200  of    241.    Elapsed: 0:01:13.
      Batch   240  of    241.    Elapsed: 0:01:28.

      Average training loss: 0.19
      Training epcoh took: 0:01:28

    Running Validation...
      Accuracy: 0.82
      Validation Loss: 0.50
      Validation took: 0:00:03

    ======== Epoch 4 / 4 ========
    Training...
      Batch    40  of    241.    Elapsed: 0:00:15.
      Batch    80  of    241.    Elapsed: 0:00:29.
      Batch   120  of    241.    Elapsed: 0:00:44.
      Batch   160  of    241.    Elapsed: 0:00:58.
      Batch   200  of    241.    Elapsed: 0:01:13.
      Batch   240  of    241.    Elapsed: 0:01:28.

      Average training loss: 0.13
      Training epcoh took: 0:01:28

    Running Validation...
      Accuracy: 0.83
      Validation Loss: 0.62
      Validation took: 0:00:03

    Training complete!
    Total training took 0:05:56 (h:mm:ss)
    Let's view the summary of the training process.



## Sample Outputs


    sentence  > jack hates sue and is loved by mary .
    predicted < acceptable
    true cls  = acceptable

    sentence  > everyone relies on someone . it ' s unclear who .
    predicted < acceptable
    true cls  = acceptable

    sentence  > the defendant denied the accusation .
    predicted < acceptable
    true cls  = acceptable

    sentence  > i ' m creating a committee . kim – you ' re in charge .
    predicted < acceptable
    true cls  = acceptable

    sentence  > vote for you !
    predicted < acceptable
    true cls  = unacceptable

    sentence  > i like bill ' s yellow shirt , but not max ' s .
    predicted < acceptable
    true cls  = acceptable

    sentence  > chocolate eggs were hidden from each other by the children .
    predicted < acceptable
    true cls  = unacceptable

    sentence  > the paper was written by john up .
    predicted < acceptable
    true cls  = unacceptable

    sentence  > the bucket was kicked by pat .
    predicted < acceptable
    true cls  = acceptable

    sentence  > i never know which papers sandy has read , but i usually know how many .
    predicted < acceptable
    true cls  = acceptable


# Task 3

## BART

BART is a language representation model that can be used for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BART is a denoising autoencoder for pretraining sequence-to-sequence models. 
BART is trained by (1) corrupting text with an arbitrary noising function
                   (2) learning a model to reconstruct the original text.
                   
BART uses a standard Transformer architecture (Encoder-Decoder) like the original Transformer model used for neural machine translation but also incorporates some changes from BERT (only uses the encoder) and GPT (only uses the decoder). 


# Pre-Training BART
BART is pre-trained by minimizing the cross-entropy loss between the decoder output and the original sequence.

## Masked Language Modeling (MLM)

MLM models such as BERT are pre-trained to predict masked tokens. This process can be broken down as follows:
Replace a random subset of the input with a mask token [MASK]. (Adding noise/corruption)
The model predicts the original tokens for each of the [MASK] tokens. (Denoising)
Importantly, BERT models can “see” the full input sequence (with some tokens replaced with [MASK]) when attempting to predict the original tokens. This makes BERT a bidirectional model, i.e. it can “see” the tokens before and after the masked tokens.

![image](https://user-images.githubusercontent.com/73247157/129408701-39f65f0b-517a-459c-b572-6195b19e6512.png)

This is suited for tasks like classification where you can use information from the full sequence to perform the prediction. However, it is less suited for text generation tasks where the prediction depends only on the previous words.

## Autoregressive Models
Models used for text generation, such as GPT2, are pre-trained to predict the next token given the previous sequence of tokens. This pre-training objective results in models that are well-suited for text generation, but not for tasks like classification.


![image](https://user-images.githubusercontent.com/73247157/129408746-4a380d75-ef27-45cc-92b3-a71064ce27a0.png)


## BART Sequence-to-Sequence
BART has both an encoder (like BERT) and a decoder (like GPT), essentially getting the best of both worlds.
The encoder uses a denoising objective similar to BERT while the decoder attempts to reproduce the original sequence (autoencoder), token by token, using the previous (uncorrupted) tokens and the output from the encoder.

![image](https://user-images.githubusercontent.com/73247157/129408840-f3fbefe8-0f40-4f89-8405-27aa602d0a80.png)



A significant advantage of this setup is the unlimited flexibility of choosing the corruption scheme; including changing the length of the original input. Or, in fancier terms, the text can be corrupted with an arbitrary noising function.
The corruption schemes used in the paper are summarized below.
Token Masking — A random subset of the input is replaced with [MASK] tokens, like in BERT.
Token Deletion — Random tokens are deleted from the input. The model must decide which positions are missing (as the tokens are simply deleted and not replaced with anything else).
Text Infilling — A number of text spans (length can vary) are each replaced with a single [MASK] token.
Sentence Permutation — The input is split based on periods (.), and the sentences are shuffled.
Document Rotation — A token is chosen at random, and the sequence is rotated so that it starts with the chosen token.

## Training Log

![Screenshot 2021-08-14 at 12 17 34 AM](https://user-images.githubusercontent.com/73247157/129405262-34f6dae8-b745-400c-91ca-e3548fc316c5.png)


# Output Logs:

    Text  > Why don't people look things up on the internet before asking Quora?
    Pred  < Why do so many people ask questions on Quora?
    Truth = Why don't Quora people just look up the answer on Google?

    Text  > What is an example of moral conflict?
    Pred  < What is a good example of moral conflict?
    Truth = What is a moral conflict? What are examples of this?

    Text  > Are the orbits of the planets synchronized on the same plane?
    Pred  < Paraphrase: Are the orbits of the planets synchronized on the same plane?
    Truth = Are the orbits of each planets coplanar?

    Text  > For what reasons have you been edit blocked on Quora?
    Pred  < What do you think is the reason you have been blocked on Quora?
    Truth = If you've never been edit blocked on Quora, to what do you attribute that?

    Text  > I do masturbation twice in a day. Does masturbation affects men's sperm count or does it cause infertility?
    Pred  < Paraphrase: I do masturbation twice a day. Does masturbation affect men's sperm count or does it cause infertility?
    Truth = Does masturbation cause infertility?

    Text  > What would happen if I went back in time and killed hitler as a baby?
    Pred  < What would happen if I went back in time and killed hitler as a baby?
    Truth = If I had a time machine, what would happen if I went back in time to assassinate Hitler?

    Text  > How many bones are in a shark's body?
    Pred  < How many bones are in a shark's body?
    Truth = How many bones do sharks have?

    Text  > Is it safe to lose 75 pound in 2 months?
    Pred  < Par it safe to lose 75 pounds in 2 months?
    Truth = What are some ideas on how to lose 75 pounds in 2 months?

    Text  > Should we buy unboxed phones?
    Pred  < Is should I buy unboxed phones?
    Truth = Should we buy unboxed mobiles?

    Text  > English cooking dominated early national cooking ; but as new immigrants arrived from other countries, other colonial soups gained popularity.
    Pred  < Paraphrase: English cooking dominated early national cooking ; but as new immigrants arrived from other countries, other colonial soups gained popularity.
    Truth = Cooking dominated early national cuisine, but as new immigrants arrived from other countries, other colonial soups gained popularity.

    Text  > If you want to guess the average IQ of people on Quora, what would it be?
    Pred  < Paraphrase: If you want to guess the average IQ of people on Quora, what would it be?
    Truth = What is the average IQ on Quora?

    Text  > Have you ever seen any ghost?
    Pred  < What you have ever seen a ghost?
    Truth = Has anyone ever seen real ghosts/scientifically unexplainable things?

    Text  > How can I overcome lack of motivation?
    Pred  < How do I overcome lack of motivation?
    Truth = How does one overcome extreme laziness and lack of motivation?

    Text  > How exactly does banning Rs 500 and Rs 1000 notes curb the problem of black money?
    Pred  < How will banning 500 and 1000 rupee notes help in curbing black money?
    Truth = How banning 500 and 1000 rs note will help fight corruption in India?

    Text  > What was the craziest thing you have done?
    Pred  < What is the craziest thing you have ever done?
    Truth = What is the craziest thing you have ever done in your lifetime?

    Text  > How could I get rid of Facebook addiction?
    Pred  < How do I get rid of Facebook addiction?
    Truth = What should I do to get rid of addictions?
