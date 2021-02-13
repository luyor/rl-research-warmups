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

train: `python snake/snake.py`
evaluate: `python snake/eval.py`

#### Environment Settings

- **state space**: 
    - grid size 15x15x3
    - grid color of [0,255]x3 represents body, head, food, space

- **action space**:
    - 4 discrete actions {1,2,3,4} represents moving 4 direction
    - snake cannot go opposite direction of last action (go backward)

- **reward**:
    - if snake head collide with food, get reward 1
    - if snake head move out of map or collide with body, get reward -1 and episode ends
    - otherwise, get reward 0

#### Agent Network

| no. | type | kernel size | stride | channel | activation |
|-----|------|-------------|--------|---------|------------|
| 1   | conv | 3           | 2      | 64      | relu       |
| 2   | conv | 3           | 1      | 64      | relu       |
| 3   | conv | 3           | 1      | 64      | relu       |
| 4   | fc   |             |        | 128     | relu       |
| 5   | fc   |             |        | 128     | relu       |


#### Problems

1. sparse reward, reward is non-zero only when food is eaten or snake dies
2. snake may stuck in infinite loop
3. experience of late game is rare

#### reference:
- [RL environment](https://github.com/grantsrb/Gym-Snake)
- [Implementations of different methods](https://github.com/gsurma/slitherin)
- [One RL implementation](https://github.com/Platun0v/snake-gym/blob/master/rl/agent.py)