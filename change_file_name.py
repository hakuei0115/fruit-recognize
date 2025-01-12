import os

for path in os.listdir('training'):
    i = 1
    for file in os.listdir(f'training/{path}'):
        os.rename(f'training/{path}/{file}', f'training/{path}/{path}_{i}.jpg')
        i += 1