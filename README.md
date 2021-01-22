# Projects of OpenAI Request for Research 2.0

Experiments on projects proposed by OpenAI's [Requests for Research 2.0](https://blog.openai.com/requests-for-research-2/). 

## Projects

### Xor-lstm

Train an LSTM to solve the XOR problem: that is, given a sequence of bits, determine its parity. The LSTM should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end.

#### Run

run `xor-lstm.ipynb` with jupytor notebook.

#### Network

Given 2 binary number x and y, xor(x,y)=(x-y)*(x-y). 
Sequential xor can be approximated by 1 unit LSTM + 1 unit Dense.

#### reference:

- https://github.com/rinkesh2131998/XOR-RNN/blob/master/XOR_rnn.ipynb
- https://github.com/mitchellvitez/lstm-xor/blob/master/lstm_xor.py
- https://vitez.me/lstm-xor


### Snake

single player snake game.

#### Run

TODO

