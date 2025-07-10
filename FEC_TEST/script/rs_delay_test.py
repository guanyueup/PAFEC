from reedsolo import RSCodec
import time
from concurrent.futures import ProcessPoolExecutor

raw_data = b'\x00\x00\x00\x01\x67\x64\x00\x1F\xAC\xD9\x40\x1F\xE8\x00\x00\x03\x00\x00\x00\x01'*70
red_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]



def encode(rr, pack_len):
    with open('rs_delay.csv', 'a') as f:
        cur_data = pack_len * raw_data
        #print(f"rr: {rr}, data_len: {len(cur_data)}")
        encoder = RSCodec(nsym=int(len(cur_data) * rr),nsize=len(cur_data))
        start = time.perf_counter()
        encoded_data = encoder.encode(cur_data)
        end = time.perf_counter()
        #print(f"Time taken: {(end - start)*1000:.3f} ms")
        encode_time = (end - start)*1000
        
        # Simulate packet loss
        for i in range(int(len(encoded_data) * rr * 0.5)):
            encoded_data[i] = 0
        
        # Decode
        start = time.perf_counter()
        decoded_data = encoder.decode(encoded_data)
        end = time.perf_counter()
        decode_time = (end - start)*1000
        #print(f"Time taken to decode: {(end - start)*1000:.3f} ms")
        f.write(f"{rr}, {pack_len}, {encode_time:.3f} , {decode_time:.3f} \n")
    


for rr in red_rate:
    rr_list = []
    len_list = []
    for pack_len in range(5, 50, 5):
        rr_list.append(rr)
        len_list.append(pack_len)
    with ProcessPoolExecutor(max_workers=18) as executor:
        executor.map(encode, rr_list, len_list)