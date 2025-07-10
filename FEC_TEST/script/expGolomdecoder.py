class ExpGolombDecoder:
    """指数哥伦布编码解码器"""
    def __init__(self, data):
        self.data = data
        self.pos = 0  # 当前字节中的位位置（0-7）
        self.index = 0  # 当前字节索引

    def read_bit(self):
        if self.index >= len(self.data):
            return 0
        bit = (self.data[self.index] >> (7 - self.pos)) & 0x1
        self.pos += 1
        if self.pos >= 8:
            self.pos = 0
            self.index += 1
        return bit

    def read_bits(self, n):
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def read_ue(self):
        # 解码无符号指数哥伦布编码
        leading_zeros = 0
        while self.read_bit() == 0:
            leading_zeros += 1
        return (1 << leading_zeros) - 1 + self.read_bits(leading_zeros)

code = b'\x28'

decoder = ExpGolombDecoder(code)

print(decoder.read_ue())