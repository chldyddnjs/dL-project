import tensorflow as tf

tf.set_random_seed(777)

sentence=("if you want to build a ship, don't drum up people together to "
        "collect wood and don't assign them tasks and work, but rather " 
        "teach them to long for the endless immensity of the sea.") 
char_set =list(set(sentence)) # char to index
char_dic = {w:i for i,w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10
learning_rate = 0.1

dataX =[]
dataY =[]
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i+1: i + sequence_length + 1]
   # print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)
