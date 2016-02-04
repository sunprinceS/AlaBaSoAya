import sys
with open(sys.argv[1], 'r') as dparents, \
     open(sys.argv[2], 'r') as labels, \
     open(sys.argv[3], 'w') as outfile:
    for (line1, line2) in zip(dparents, labels):
        tree = line1.strip().split(' ')
        sent = line2.strip()
        label = 0
        if sent == 'positive':
            label = 1
        elif sent == 'negative':
            label = -1
        for i in range(len(tree)):
            node = tree[i]
            if int(node) == 0:
                outfile.write(str(label))
            else:
                outfile.write('0')
            if i == len(tree) - 1:
                outfile.write('\n')
            else:
                outfile.write(' ')
