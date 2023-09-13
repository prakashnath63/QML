import tensorflow as tf 
from tensorflow import keras
import tensorflow_quantum as tfq
import numpy as np
import cirq
import sympy


class B_PQC_layer(keras.layers.Layer):
    def __init__(self, upstream_symbols, managed_symbols):
        super().__init__()
        self.all_symbols = upstream_symbols + managed_symbols
        self.upstream_symbols = upstream_symbols
        self.managed_symbols = managed_symbols
        self.batch_size = 64

  

    def build(self, input_shape):
        self.managed_weights = tf.Variable(
            initial_value=tf.keras.initializers.RandomUniform(0, 2 * np.pi)(shape=(1, len(self.managed_symbols)),dtype=tf.float32),
            trainable=True
            )
        
    @tf.autograph.experimental.do_not_convert
    def Circuit_create(self, batch_size):
        qubits = cirq.LineQubit.range(15)
        num_variables = 120
        variable_names = [f"u{i}" for i in range(num_variables)]
        params = sympy.symbols(variable_names)

        circuit = cirq.Circuit()  
        for i in range(15):
            circuit += cirq.ry(params[i] * np.pi).on(qubits[i])
        for i in range(15):
            circuit += cirq.rz(params[15+i] * np.pi).on(qubits[i])
       
        for i in range(14):
            circuit +=cirq.CNOT(qubits[i],qubits[i+1])   
            
        for i in range(15):
            circuit += cirq.ry(params[30+i] * np.pi).on(qubits[i])
        for i in range(15):
            circuit += cirq.rz(params[45+i] * np.pi).on(qubits[i])
 
        for i in range(15):
            circuit += cirq.ry(params[60+i] * np.pi).on(qubits[i])
        for i in range(15):
            circuit += cirq.rz(params[75+i] * np.pi).on(qubits[i])
       
        for i in range(14):
            circuit +=cirq.CNOT(qubits[i],qubits[i+1])   
            
        for i in range(15):
            circuit += cirq.ry(params[90+i] * np.pi).on(qubits[i])
        for i in range(15):
            circuit += cirq.rz(params[105+i] * np.pi).on(qubits[i])
            
        Moperators = [cirq.Z(qubits[i]) for i in range(15)]
        tinput=tfq.convert_to_tensor([circuit for _ in range(batch_size)])
        return tinput,Moperators

      
    def call(self, inputs):
        # inputs are: circuit tensor, upstream values
        tinput,Moperators=self.Circuit_create(self.batch_size)
        upstream_shape = tf.gather(tf.shape(tinput), 0)
        tiled_up_weights = tf.tile(self.managed_weights, [upstream_shape, 1])
        joined_params = tf.concat([inputs, tiled_up_weights], 1)
        output=tfq.layers.Expectation()(tinput,
                                        operators=Moperators,
                                        symbol_names=self.all_symbols,
                                        symbol_values=joined_params
                                        )
        
        return output
    

class HybridModel(keras.Model):

    def __init__(self):
        super(HybridModel, self).__init__()

        self.fc1 = keras.layers.Dense(1024, name="embedding")
        self.fc2 = keras.layers.Dense(256, activation="relu", name="dense_1")
        self.fc3 = keras.layers.Dense(384, activation="relu", name="dense_2")
        self.fc4 = keras.layers.Dense(192, activation="relu", name="dense_3")
        self.fc5 = keras.layers.Dense(60, activation="relu", name="dense_4")
        self.fc6 = keras.layers.Dense(7, activation="softmax", name="output")

        num_variables = 60
        input1 = [f"u{i}" for i in range(num_variables)]
        input2 = [f"u{i}" for i in range(num_variables, num_variables + 60)]
        self.PQC1 = B_PQC_layer(input1, input2)

    def call(self, inputs):

        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.PQC1(x)

        return self.fc6(x)