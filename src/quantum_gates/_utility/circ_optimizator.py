"""
This file implement a class that represents an optimization algorithm for the output of the circuit class. 
It takes the list of gate and qubit on which the gate is applied and compacts the list so there are lesser matrix to calculate in the backend
Pay attention that it is designend mainly to work with BinaryCircuit and BinaryBackend

"""

import numpy as np

class Optimizator(object):

    def __init__(self, level_opt:int, circ_list: list, qubit_list: list):
        if level_opt > 4 or level_opt < 0:
            raise ValueError(f"Not exist the level {level_opt} of optimizaion. They are from 0 to 4")
        self.circ_list = circ_list
        self.level_opt = level_opt
        self.qubit_list = qubit_list

    def optimize(self) -> list:

        if len(self.circ_list) <= 2:
            level = 0
        elif len(self.qubit_list) == 1:
            level = 0
        else:
            level = self.level_opt

        if level == 0:
            return self.circ_list

        if level == 1:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            return result_1
        
        elif level == 2:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            result_2 = self.opt_level_2(gate_list=result_1)
            result_3 = self.opt_level_3(gate_list=result_2)
            return result_2
        
        elif level == 3:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            result_2 = self.opt_level_2(gate_list=result_1)
            result_3 = self.opt_level_3(gate_list=result_2)
            return result_3

        elif level == 4:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            result_2 = self.opt_level_2(gate_list=result_1)
            result_3 = self.opt_level_3(gate_list=result_2)
            result_4 = self.opt_level_4(gate_list=result_3)
            return result_4

    def opt_level_1(self, gate_list: list) -> list:
        result_1 = []

        while gate_list:
            if len(gate_list[0][1]) == 1: # check if the current element is a 1q gate
                qubit = gate_list[0][1]
                c = 1
                while len(gate_list[c][1]) == 1 and qubit == gate_list[c][1]: # count how many successive gate have the same qubit
                    c += 1
                if c > 1:
                    gates = []
                    for i in range(c): #store in a list all the matrices of the gates involved
                        gates.append(gate_list[i][0])
                    # calculate the gate
                    gate = np.identity(2)
                    for j in range(c):
                        gate = gates[j] @ gate
                    result_1.append([gate,qubit]) # append the gate with the qubit
                    gate_list = gate_list[c:]
                else: # if there is only 1 consequent 2q gate with the same qubit just happend the 
                    result_1.append(gate_list[0])
                    gate_list = gate_list[c:]
            else:
                result_1.append(gate_list[0])
                gate_list = gate_list[1:] 
            if len(gate_list) == 1:
                result_1.append(gate_list[0])
                gate_list.remove(gate_list[0])
        
        return result_1
    
    def opt_level_2(self, gate_list:list) -> list:

        q2_gate = 0
        result_2 = []
        
        for i in gate_list: #count the number of two qubit gate in the result
            if len(i[1]) == 2:
                q2_gate += 1

        if q2_gate > 0: # check that exists almost one two qubit gate 
            for i in range(q2_gate): # iterate over the number of  two qubit gate
                
                # create the snip
                _snip = []
                counter = 0
                while len(gate_list[counter][1]) != 2:

                    _snip.append(gate_list[counter])
                    counter += 1
                _snip.append(gate_list[counter]) # append the 2q gate
                if len(gate_list) > counter+1: # assert that there are gate after the one considered
                    if len(gate_list[counter+1][1]) == 1: # append the right after gate if it is a 1 q gate
                        _snip.append(gate_list[counter+1])
                        if len(gate_list[counter+2][1]) == 1: # if it's appended the first closer whatch if append the second after gate if it is a 1 q gate
                            _snip.append(gate_list[counter+2])
                # the snip now is done

                # remove from result the snipped
                gate_list = gate_list[len(_snip):]            

                # process the snip
                processed_snip = self.proces_snipped(snip = _snip)
                
                # append to result_2 the processed snipped
                result_2 += processed_snip

            # process the last part of the gate_list which are only single qubit gates.
            result_2 += gate_list
        else:
            result_2 = gate_list
        return result_2
    
    def opt_level_3(self, gate_list:list) -> list:

        result_3 = []

        while gate_list:
            if len(gate_list[0][1]) == 2 and len(gate_list) > 1: # check if the current element is a 2q gate
                qubit = gate_list[0][1]
                c = 1
                while len(gate_list[c][1]) == 2 and qubit == gate_list[c][1]: # count how many successive gate have the same qubit
                    c += 1
                    if c >= len(gate_list): # if the list of gate_list is finished break the loop
                        break
                        
                if c > 1:
                    gates = []
                    for i in range(c): #store in a list all the matrices of the gates involved
                        gates.append(gate_list[i][0])
                    # calculate the gate
                    gate = np.identity(4)
                    for j in range(c):
                        gate = gates[j] @ gate
                    result_3.append([gate,qubit]) # append the gate with the qubit
                    gate_list = gate_list[c:]
                else: # if there is only 1 consequent 2q gate with the same qubit just happend the 
                    result_3.append(gate_list[0])
                    gate_list = gate_list[c:]
            else:
                result_3.append(gate_list[0])
                gate_list = gate_list[1:]
            if len(gate_list) == 1:
                result_3.append(gate_list[0])
                gate_list.remove(gate_list[0])
        
        return result_3
    
    def opt_level_4(self, gate_list:list) -> list:

        last_part = []
        l = 0
        gate_list = gate_list[::-1] # reverse the list of the gate
        reorder_qubit_list = list(range(len(self.qubit_list)))

        # check if there is almost one two qubit gate
        for item in gate_list:
            if len(item[1]) == 2:
                q2_gate_check = True
                break
            else: 
                q2_gate_check = False

        if q2_gate_check: # there is almost one two qubit gate
            while len(gate_list[l][1]) == 1: # append in a list the last gate after the 2q gate
                last_part.append(gate_list[l])
                l+=1

            gate_list = gate_list[::-1] # re-reverse the list of the gate 
            length_last_part = len(last_part) # calculate the length of the last part

            if l != length_last_part:
                raise ValueError("Mismatch between length of last part and gates in it")

            if l > 1: # check if there almost two element in the last part
                last_part = last_part[::-1] # reverse the list to obtain the right order
                result_4 = gate_list[:-l] # add all the gate before the last part a result_4
                for q_i in reorder_qubit_list: # scan all the used qubit 
                    if len(last_part) > 1:
                        indices = [index for index, element in enumerate(last_part) if element[1][0] == q_i] # find the indices of the element for the qubit q_i
                        if len(indices) > 1:
                            gate = np.identity(2)
                            for ind in indices:
                                gate = last_part[ind][0] @ gate
                            result_4.append([gate,[q_i]])
                        elif len(indices) == 1:
                            ind = indices[0]
                            result_4.append([last_part[ind][0],[q_i]])
                        last_part = [element for i, element in enumerate(last_part) if i not in indices] # remove from the last part the used items in this iteration
                    elif len(last_part) == 1:
                        result_4.append(last_part[0])
                        break
                    elif len(last_part) == 0:
                        break
            else:
                result_4 = gate_list
        else: # there is only one qubit gate
            last_part = gate_list
            length_last_part = len(last_part) # calculate the length of the last part
            if length_last_part > 1: # check if there almost two element in the last part
                last_part = last_part[::-1] # reverse the list to obtain the right order
                result_4 = []
                for q_i in reorder_qubit_list: # scan all the used qubit 
                    if len(last_part) > 1:
                        indices = [index for index, element in enumerate(last_part) if element[1][0] == q_i] # find the indices of the element for the qubit q_i
                        if len(indices) > 1:
                            gate = np.identity(2)
                            for ind in indices:
                                gate = last_part[ind][0] @ gate
                            result_4.append([gate,[q_i]])
                        else: 
                            ind = indices[0]
                            result_4.append([last_part[ind][0],[q_i]])
                        last_part = [element for i, element in enumerate(last_part) if i not in indices] # remove from the last part the used items in this iteration
                    elif len(last_part) == 1:
                        result_4.append(last_part[0])
                        break
                    elif len(last_part) == 0:
                        break
            else:
                result_4 = last_part[::-1]
        return result_4
    
    def proces_snipped(self, snip: list) -> list:
        loc = 0 # index of the 2q gates in the snip
        n_elem = len(snip) -1 # number of elements in the snip starting from 0
        proces_snip = [] 

        # find the index of the 2q gates in the snipped
        for i,item in enumerate(snip):
            if len(item[1]) ==  2:
                loc = i

        #before the two qubit gate
        if loc - 2 >= 0: # check there are almost two gate before the 2q gate
            #print("before 1")
            if snip[loc-2][1][0] == snip[loc][1][0] and snip[loc-1][1][0] == snip[loc][1][1]: # check if the two gates before are compatible with the 2q gate
                #print("before 1.1")
                gate = snip[loc][0] @ np.kron(snip[loc-2][0],snip[loc-1][0])
                qubits = snip[loc][1]
                if loc - 2 > 0: # check if there are gate before the ones used 
                    for i in range(loc-2):# append in the process snip all the gate before the that ones used before
                        proces_snip.append(snip[i])
                    proces_snip.append([gate,qubits])
                else:
                    proces_snip.append([gate,qubits])
            elif snip[loc-2][1][0] == snip[loc][1][1] and snip[loc-1][1][0] == snip[loc][1][0]: # check if the two gates before are compatible with the 2q gate with inverted qubit
                #print("before 1.2")
                gate = snip[loc][0] @ np.kron(snip[loc-1][0],snip[loc-2][0])
                qubits = snip[loc][1]
                if loc - 2 > 0: # check if there are gate before the ones used 
                    for i in range(loc-2):# append in the process snip all the gate before the that ones used before
                        proces_snip.append(snip[i])
                    proces_snip.append([gate,qubits])
                else:
                    proces_snip.append([gate,qubits])
            elif snip[loc-1][1][0] == snip[loc][1][0]: # the closest gate is compatible with control qubit, but not the second closer
                #print("before 1.3")
                gate = snip[loc][0] @ np.kron(snip[loc-1][0],np.identity(2))
                qubits = snip[loc][1]
                for i in range(loc-1):
                    proces_snip.append(snip[i]) # append in the process snip all the gate before the that ones used before
                proces_snip.append([gate,qubits])
            elif snip[loc-1][1][0] == snip[loc][1][1]: # the closest gate is compatible with target qubit, but not the second closer
                #print("before 1.4")
                gate = snip[loc][0] @ np.kron(np.identity(2),snip[loc-1][0])
                qubits = snip[loc][1]
                for i in range(loc-1):
                    proces_snip.append(snip[i]) # append in the process snip all the gate before the that ones used before
                proces_snip.append([gate,qubits])
            else: # no one of the item before match with the 2q gate
                #print("before 1.5")
                for i in range(loc+1):
                    proces_snip.append(snip[i])
        elif loc -1 >= 0: # check if there is one gate before the 2q gate
            #print("before 2")
            if snip[loc-1][1][0] == snip[loc][1][0]: # the closer gate is compatible with control qubit
                #print("before 2.1")
                gate = snip[loc][0] @ np.kron(snip[loc-1][0],np.identity(2))
                qubits = snip[loc][1]
                if loc - 1 > 0:
                    for i in range(loc-1):# append in the process snip all the gate before the that ones used before
                        proces_snip.append(snip[i])
                    proces_snip.append([gate,qubits])
                else:
                    proces_snip.append([gate,qubits])

            elif snip[loc-1][1][0] == snip[loc][1][1]: # the closer gate is compatible with target qubit
                #print("before 2.2")
                gate = snip[loc][0] @ np.kron(np.identity(2),snip[loc-1][0])
                qubits = snip[loc][1]
                if loc - 1 > 0:
                    for i in range(loc-1):# append in the process snip all the gate before the that ones used before
                        proces_snip.append(snip[i])
                    proces_snip.append([gate,qubits])
                else:
                    proces_snip.append([gate,qubits])
            else: # the closer gate is neither compatible with the targer nor the control qubit
                #print("before 2.3")
                for i in range(loc+1):
                    proces_snip.append(snip[i])
        else: # if there aren't items before the 2q qubit just append the item
            #print("before 3")
            proces_snip.append(snip[0]) 
            if len(snip[0][1]) != 2:
                raise ValueError(f"Expected an item that act on 2 qubit, gate with {snip[0][1]} qubit found")

        #after the two qubit gate
        if loc + 2 <= n_elem: # check if there are two items before the 2q gate
            #print("after 1")
            if snip[loc+2][1][0] == snip[loc][1][0] and snip[loc+1][1][0] == snip[loc][1][1]: # check if the two gates before are compatible with the 2q gate
                #print("after 1.1")
                gate = np.kron(snip[loc+2][0],snip[loc+1][0]) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits])

            elif snip[loc+2][1][0] == snip[loc][1][1] and snip[loc+1][1][0] == snip[loc][1][0]: # check if the two gates before are compatible with the 2q gate with inverted qubit
                #print("after 1.2")
                gate = np.kron(snip[loc+1][0],snip[loc+2][0]) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits])

            elif snip[loc+2][1][0] == snip[loc][1][0] and snip[loc+1][1][0] != snip[loc][1][1]: #the second closest gate is compatible with the control qubti not the first gate
                #print("after 1.3")
                gate = np.kron(snip[loc+2][0],np.identity(2)) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits])
                proces_snip.append(snip[loc+1])

            elif snip[loc+2][1][0] == snip[loc][1][1] and snip[loc+1][1][0] != snip[loc][1][0]: #the second closest gate is compatible with the control qubti not the first gate
                #print("after 1.4")
                gate = np.kron(np.identity(2),snip[loc+2][0]) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits])
                proces_snip.append(snip[loc+1])

            elif snip[loc+1][1][0] == snip[loc][1][0]: # the closest gate is compatible with control qubit, not the second
                #print("after 1.5")
                gate = np.kron(snip[loc+1][0],np.identity(2)) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits])
                proces_snip.append(snip[loc+2])

            elif snip[loc+1][1][0] == snip[loc][1][1]: # the closest gate is compatible with target qubit, not the second
                #print("after 1.6")
                gate = np.kron(np.identity(2),snip[loc+1][0]) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits])
                proces_snip.append(snip[loc+2]) 
            else: # nor the first or the second after is compatible with the 2q gate
                #print("after 1.7")
                proces_snip.append(snip[loc+1])
                proces_snip.append(snip[loc+2])
        elif loc+1 <= n_elem:  # check if there is one item before the 2q gate
            #print("after 2")
            if snip[loc+1][1][0] == snip[loc][1][0]: # the closest gate is compatible with control qubit
                #print("after 2.1")
                gate = np.kron(snip[loc+1][0],np.identity(2)) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits]) 

            elif snip[loc-1][1][0] == snip[loc][1][1]: # the closest gate is compatible with target qubit
                #print("after 2.2")
                gate = np.kron(np.identity(2),snip[loc+1][0]) @ proces_snip[-1][0]
                qubits = proces_snip[-1][1]
                proces_snip = proces_snip[:-1]
                proces_snip.append([gate,qubits])
            else: # the closest 
                #print("after 2.2")
                proces_snip.append(snip[loc+1])
        
        return proces_snip
    



    