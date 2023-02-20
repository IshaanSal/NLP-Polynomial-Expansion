# NLP-Polynomial-Expansion
Implementation of a deep learning model that learns to expand single variable polynomials, where the model takes the factorized sequence as inputs and predicts the expanded sequence. { EX: (h+8)*(7*h-3)=7*h^2+53*h-24 }

Taking into consideration the Natural Language Processing (NLP) objective of the project, my overall approach of handling this task came down to utilizing the artificial intelligence construct of Recurrent Neural Networks.
In essence, a recurrent neural network (RNN) is a type of neural network designed to process sequential data, such as time series and typically natural language text. While feedforward networks process input data in a static manner, RNN’s provide more dynamic capabilities in the form of processing input sequences with varying lengths. A significant advantage of these sequence-to-sequence networks is their ability to manage data passed “one step at a time,” meaning they take into account the current input, as well as the previous inputs as well, ultimately resulting in a more optimized training model.

**Implementation Methodology:**
Data Organization - Converting raw data into usable arrays ||
Tokenization - Split arrays into understandable units ||
Vectorization - Convert tokens into machine-readable vectors ||
Model Training - Training the RNN to predict the expanded form based on the factored

_Data Organization:_
-The provided function “load_file()” effectively created the necessary iterables that could be used further.

_Tokenization:_
-Tokenization is a common process in natural language processing as it allows for programmers to convert large chunks of sentences and string data into more processable and comprehensive bits that can be assigned meaning. In this case, both the factored and expanded expressions were tokenized, to identify each number, set of parentheses, operators, as well as functions like sine and cosine. These tokens were then internally stored into a dictionary with a unique ID assigned to each. This allows for the network to distinguish between the elements of an inputted factored form better.

_Vectorization:_
-Tokenizing the data does create more detailed information to be inputted into the neural network, however, most machine learning algorithms can only be inputted numerical data to operate on and train from. As a result an additional step of converting all the tokens, which are still string characters, to their respective numerical representation is quite necessary. Also known as word embedding, this process generally makes training the model more efficient.

_Model Training:_
-Following the previous steps, this final step involves actually training the model with the given information. Providing the model with a sample input, the RNN works through the provided tokens sequentially to predict the expanded output.
