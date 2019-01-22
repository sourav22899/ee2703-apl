#!/usr/bin/env python3

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('filename',help = 'Mention the file name to parse.')
# args = parser.parse_args()

import sys

def main():
    # Checks if the number of arguments is correct, else exits the script.
    if len(sys.argv) != 2:
        print("Incorrect format.The correct format is $ python3 solution.py filename")
        sys.exit()
    else:
        filename = sys.argv[1]

    # Checks if the file exists, else exits the script.
    try:
        f = open(filename,'r')
    except IOError:
        print('Could not read the file:{0}'.format(filename))
        sys.exit()        

    lines = f.read().splitlines()
    f.close()
    # print(lines)

    def ExtractCircuit(lines):
        """
        Extracts the circuit part from the netlist file.

        Args:
            A list of lines of the parsed file.
        
        Returns:
            A list describing the circuit removing all the comments and other parts.
        """
        circuit,flag = [],0
        for line in lines:
            if line == '.circuit':
                circuit.append(line)
                flag = 1
            elif line == '.end' and flag == 1:
                circuit.append(line)
                flag = 0
            elif flag == 1:
                circuit.append(line)

        final_circuit = []   
        for line in circuit:
            words = line.split()
            newline = []
            for word in words:
                if word[0] == '#':
                    break
                newline.append(word)
            newline = ' '.join(newline)
            final_circuit.append(newline)
        return final_circuit

    final_circuit = ExtractCircuit(lines)
    rev_final_circuit = reversed(final_circuit)
    for line in rev_final_circuit:
        if line != '.circuit' and line != '.end':
            words = reversed(line.split())
            print (' '.join(words))   

if __name__ == "__main__":
    main()