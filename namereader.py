fhand = open('dino_names.txt')
fout = open('dino_train.csv', 'w+')

# maxlen = 0

# alpha_list = []

data_count = 0

for dino_name in fhand:
    dino_lower = dino_name.lower()
    dino_lower = dino_lower.rstrip()

    # if maxlen < len(dino_lower):
    #     maxlen = len(dino_lower)

    for i in range(0, len(dino_lower)+1):
        # line = dino_lower[0:j]
        if i < len(dino_lower):
            if i == 0:
                fout.write('*'+','+dino_lower[0]+'\n')
            else:
                fout.write('*' + dino_lower[:i]+','+dino_lower[i]+'\n')
        if i == len(dino_lower)-1:
            fout.write('*' + dino_lower+','+'$'+'\n')
            data_count += 1
            break

#     for alphabet in dino_lower:
#         if alphabet not in alpha_list:
#             alpha_list.append(alphabet)

fhand.close()
fout.close()
# print(f'Length of the largest string is {maxlen} characters')
# print(alpha_list)
# print(len(alpha_list))
# print(f'We have {data_count} samples')