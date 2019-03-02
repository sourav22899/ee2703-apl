#!/usr/bin/env python3

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('filename',help = 'Mention the file name to parse.')
# args = parser.parse_args()

import sys
import os

def main():
    # Checks if the number of arguments is correct, else exits the script.
    if len(sys.argv) != 2:
        print("Incorrect format.The correct format is $ python3 assign1.py filename")
        sys.exit()
    else:
        filename = sys.argv[1]

    # Checks if the file exists, else exits the script.
    try:
        f = open(filename,'r')
    except IOError:
        print('Could not read the file:{0}'.format(filename))
        sys.exit()        

    if os.stat(filename).st_size == 0:
        print('File is empty.')
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
        final_circuit,flag,circuit = [],0,[]
        for line in lines:
            words = line.strip().split()
            newline = []
            for word in words:
                if word[0] == '#':
                    break
                newline.append(word)
            newline = ' '.join(newline)
            circuit.append(newline)
        
        for line in circuit:
            if line == '.circuit':
                final_circuit.append(line)
                flag = 1
            elif line == '.end' and flag == 1:
                final_circuit.append(line)
                flag = 0
            elif flag == 1:
                final_circuit.append(line)
        
        if len(final_circuit) == 0:
            print('The netlist file doesnot contain ".circuit" and/or ".end" line.')
            sys.exit()

        if final_circuit[-1] != '.end':
            print('The netlist file doesnot contain ".end" line.')
            sys.exit()

        return final_circuit

    final_circuit = ExtractCircuit(lines)
    rev_final_circuit = reversed(final_circuit)
    for line in rev_final_circuit:
        if line != '.circuit' and line != '.end':
            words = reversed(line.split())
            print (' '.join(words))   

if __name__ == "__main__":
    main()