import sys
with open(sys.argv[1], 'r') as infile, \
     open(sys.argv[2], 'w') as ofile:
    for line in infile:
        line = line.strip().split()
        ofile.write('0')
        for i in range(1, len(line)):
            ofile.write(' 0')
        ofile.write('\n')
