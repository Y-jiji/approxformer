import torch as t

if __name__ == '__main__':
    print(t.nn.Transformer().generate_square_subsequent_mask(10))